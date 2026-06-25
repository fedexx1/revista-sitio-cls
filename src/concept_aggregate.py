"""SP1 Stage 1 (cont.) — aggregate the concept layer into a ranked candidate table.

Columns: lemma, n_instances, n_paragraphs, tfidf_rank, example_forms,
example_para_ids, enrichment_score, in_current_seeds, keep.

enrichment_score = track-2 log-odds that a lemma's paragraphs cite a Blanchot-circle
author vs the rest. Inspection only; NOT the curation criterion.
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import concept_extract as ce
import concept_layer as cl
from lexicons import BLANCHOT_CIRCLE, BLANCHOT_ID, BLANCHOT_SEEDS

DATA = ce.DATA
RES = ce.ROOT / "results"


def _tfidf_ranks(layer: pd.DataFrame) -> dict[str, int]:
    """No-LLM baseline: tf-idf over the per-paragraph lemma bags. Returns
    lemma -> rank (1 = highest summed tf-idf). Compared against the LLM candidates."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [" ".join(ls) for ls in layer["lemmas"]]
    vec = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False)
    X = vec.fit_transform(docs)
    scores = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    order = sorted(range(len(terms)), key=lambda i: scores[i], reverse=True)
    return {terms[i]: rank + 1 for rank, i in enumerate(order)}


def _term_lemma_map(layer: pd.DataFrame) -> dict[str, list[str]]:
    """Lemmatize each unique surface term exactly once: term -> list[lemma]."""
    unique_terms = sorted({t for terms in layer["terms"] for t in terms})
    return {t: cl.lemmatize_terms([t]) for t in unique_terms}


def aggregate(layer: pd.DataFrame) -> pd.DataFrame:
    layer = layer.reset_index(drop=True)
    circle = set(BLANCHOT_CIRCLE) | {BLANCHOT_ID}
    seeds = {ce.canonicalize(s) for s in BLANCHOT_SEEDS}
    term2lemmas = _term_lemma_map(layer)

    in_circle = [any(p in circle for p in row["persons"]) for _, row in layer.iterrows()]
    n_circle = sum(in_circle)
    n_rest = len(layer) - n_circle

    n_instances: Counter = Counter()
    paras_with: dict[str, set] = {}
    forms: dict[str, Counter] = {}
    circle_count: Counter = Counter()
    for idx, row in layer.iterrows():
        para_lemmas: set[str] = set()
        for term in row["terms"]:
            for lemma in term2lemmas.get(term, []):
                n_instances[lemma] += 1
                paras_with.setdefault(lemma, set()).add(idx)
                forms.setdefault(lemma, Counter())[term] += 1
                para_lemmas.add(lemma)
        if in_circle[idx]:
            for lemma in para_lemmas:
                circle_count[lemma] += 1

    tfidf_rank = _tfidf_ranks(layer)

    rows = []
    for lemma, n_inst in n_instances.items():
        paras = paras_with[lemma]
        n_par = len(paras)
        c_in = circle_count.get(lemma, 0)
        c_out = n_par - c_in
        odds_lemma = (c_in + 0.5) / (c_out + 0.5)          # smoothed log-odds
        odds_base = (n_circle + 0.5) / (n_rest + 0.5)
        enrichment = math.log(odds_lemma / odds_base)
        rows.append({
            "lemma": lemma,
            "n_instances": n_inst,
            "n_paragraphs": n_par,
            "tfidf_rank": tfidf_rank.get(lemma, ""),
            "example_forms": "; ".join(f for f, _ in forms[lemma].most_common(3)),
            "example_para_ids": "; ".join(layer.iloc[i]["para_id"] for i in sorted(paras)[:3] if layer.iloc[i]["para_id"]),
            "enrichment_score": round(enrichment, 3),
            "in_current_seeds": lemma in seeds,
            "keep": "",
        })
    return pd.DataFrame(rows).sort_values("n_paragraphs", ascending=False).reset_index(drop=True)


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    layer = pd.read_parquet(DATA / "concept_layer.parquet")
    cand = aggregate(layer)
    cand.to_csv(RES / "lexicon_candidates.csv", index=False, encoding="utf-8-sig")
    print(f"wrote {RES / 'lexicon_candidates.csv'}: {len(cand)} lemmas")
    print(cand.head(15).to_string(index=False))
    print(f"current seeds flagged in candidates: {int(cand['in_current_seeds'].sum())}")


if __name__ == "__main__":
    main()
