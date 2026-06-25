"""SP1 Stage 1 — build the corpus-wide LLM concept layer (C).

Open concept extraction (Haiku) over the ~3,476 audit paragraphs, lemmatized and
filtered with spaCy Spanish into a ranked candidate-lemma table. The per-paragraph
lemma layer (data/concept_layer.parquet) feeds the coherence validation and SP2/SP3.

Bulk LLM cache lives in data/discovery_cache/ (gitignored). Re-runs are free.

Usage:
    python src/concept_layer.py --smoke    # 50 paragraphs, sanity-check
    python src/concept_layer.py            # full corpus
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import concept_extract as ce

ROOT = ce.ROOT
DATA = ce.DATA
RES = ROOT / "results"
DISCOVERY_CACHE = DATA / "discovery_cache"
SPACY_MODEL = "es_core_news_sm"
KEEP_POS = {"NOUN", "ADJ", "VERB"}

try:  # make piped stdout UTF-8-safe on Windows (audit.load_paragraphs prints "->")
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    try:
        return spacy.load(SPACY_MODEL, disable=["ner", "parser"])
    except OSError as e:
        raise SystemExit(
            f"spaCy model '{SPACY_MODEL}' not installed. Run:\n"
            f"    python -m spacy download {SPACY_MODEL}"
        ) from e


@lru_cache(maxsize=1)
def _proper_noun_tokens() -> frozenset[str]:
    """Lowercased, accent-stripped tokens of every name in persons.csv, to drop
    cited-person names that survive POS filtering."""
    names = pd.read_csv(DATA / "persons.csv")["name"].dropna().tolist()
    toks: set[str] = set()
    for n in names:
        for part in str(n).split():
            c = ce.canonicalize(part)
            if len(c) >= 3:
                toks.add(c)
    return frozenset(toks)


def lemmatize_terms(terms: list[str]) -> list[str]:
    """Map a paragraph's extracted concept phrases to filtered content lemmas:
    keep NOUN/ADJ/VERB, drop stopwords/punct/numbers/proper-noun tokens, lowercase
    + accent-strip via canonicalize. Returns deduped lemmas in first-seen order."""
    if not terms:
        return []
    proper = _proper_noun_tokens()
    out: list[str] = []
    seen: set[str] = set()
    for doc in _nlp().pipe(terms):
        for tok in doc:
            if tok.pos_ not in KEEP_POS or tok.is_stop or not tok.is_alpha:
                continue
            lemma = ce.canonicalize(tok.lemma_)
            if len(lemma) < 3 or lemma in proper or lemma in seen:
                continue
            seen.add(lemma)
            out.append(lemma)
    return out


from concurrent.futures import ThreadPoolExecutor, as_completed


def build_layer(df: pd.DataFrame, max_workers: int = 2) -> pd.DataFrame:
    """For each paragraph: lean Haiku terms extraction (cached, parallelized, with
    rate-limit retry) -> filtered lemmas (spaCy, single-threaded). Row order preserved.
    max_workers is low + retry/backoff so the run stays under the output-tokens/min cap."""
    import time
    texts = df["text"].tolist()
    terms_by_i: list[list[str] | None] = [None] * len(texts)

    def _extract(i: int):
        for attempt in range(6):
            try:
                return i, ce.extract_open_terms(texts[i], "haiku", cache_dir=DISCOVERY_CACHE)
            except Exception as e:  # noqa: BLE001 - retry only on rate limits
                if "RateLimit" not in type(e).__name__ or attempt == 5:
                    raise
                time.sleep(30 * (attempt + 1))  # back off to let the per-minute window reset

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_extract, i) for i in range(len(texts))]
        for fut in as_completed(futs):
            i, terms = fut.result()
            terms_by_i[i] = terms
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(texts)} paragraphs extracted")

    rows = []
    for (_, r), terms in zip(df.iterrows(), terms_by_i):
        rows.append({
            "para_id": r["para_id"],
            "persons": list(r["persons"]),
            "terms": terms,
            "lemmas": lemmatize_terms(terms),
        })
    return pd.DataFrame(rows)


def _load_paragraphs():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import audit
    df, _ = audit.load_paragraphs()
    return df


def main(smoke: bool = False) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    df = _load_paragraphs()
    if smoke:
        df = df.head(50).reset_index(drop=True)
        print(f"SMOKE: {len(df)} paragraphs")
    layer = build_layer(df)
    out = DATA / ("concept_layer_smoke.parquet" if smoke else "concept_layer.parquet")
    layer.to_parquet(out, index=False)
    if not smoke:
        import json
        from datetime import datetime, timezone
        from importlib.metadata import version
        (DATA / "lexicon_discovery_provenance.json").write_text(json.dumps({
            "model": ce.MODELS["haiku"],
            "mode": "open_terms (lean, terms-only)",
            "temperature": ce.TEMPERATURE,
            "n_paragraphs": int(len(layer)),
            "n_unique_lemmas": int(layer["lemmas"].explode().dropna().nunique()),
            "spacy_model": SPACY_MODEL,
            "spacy_version": version("spacy"),
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "note": "Per-paragraph concept layer (lean Haiku terms-only). Bulk LLM cache "
                    "in data/discovery_cache/ is gitignored but rebuildable.",
        }, indent=2), encoding="utf-8")
    n_lemmas = layer["lemmas"].apply(len)
    print(f"wrote {out}: {len(layer)} paragraphs, "
          f"mean lemmas/para={n_lemmas.mean():.2f}, "
          f"empty paras={int((n_lemmas == 0).sum())}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
