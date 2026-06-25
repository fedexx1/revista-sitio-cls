"""SP1 Stage 3 — build curated Blanchot seed sets from the tagged candidate CSV.

Reads results/lexicon_candidates.csv (keep in {core, shared}), expands each kept lemma
to the surface word-forms that occur in the corpus and lemmatize to it (so audit's
whole-word regex matches inflections), keeps editor-added canonical terms STRICTLY
SEPARATE (core_manual vs core_discovered), and writes data/curated_seeds.json.
Committed BEFORE the re-audit (pre-registration).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit
import concept_extract as ce
import concept_layer as cl

DATA = ce.DATA
RES = ce.ROOT / "results"
MANUAL_CORE = ["afuera", "dehors", "neutro"]  # canonical Blanchot terms ADDED BY HAND (LLM missed them)


def _lemma_to_wordforms() -> dict[str, set[str]]:
    """Over the corpus paragraph texts, map lemma -> {surface word-forms}. Lets a kept
    lemma (e.g. 'errancia') expand to every inflected form actually present."""
    df, _ = audit.load_paragraphs()
    nlp = cl._nlp()
    out: dict[str, set[str]] = defaultdict(set)
    for doc in nlp.pipe(df["text"].tolist(), batch_size=64):
        for tok in doc:
            if tok.is_alpha and not tok.is_stop:
                out[ce.canonicalize(tok.lemma_)].add(tok.text.lower())
    return out


def build() -> dict:
    cand = pd.read_csv(RES / "lexicon_candidates.csv", encoding="utf-8-sig", keep_default_na=False)
    tier = {row["lemma"]: str(row["keep"]).strip().lower()
            for _, row in cand.iterrows() if str(row["keep"]).strip()}
    bad = {v for v in tier.values() if v not in {"core", "shared"}}
    if bad:
        raise SystemExit(f"keep column must be 'core' or 'shared'; found {bad}. Tag the CSV first.")

    l2w = _lemma_to_wordforms()

    def forms(lemma: str) -> list[str]:
        c = ce.canonicalize(lemma)
        return sorted({c} | {ce.canonicalize(w) for w in l2w.get(c, set())})

    core_lemmas = [l for l, t in tier.items() if t == "core"]
    shared_lemmas = [l for l, t in tier.items() if t == "shared"]
    core_discovered = sorted({f for l in core_lemmas for f in forms(l)})
    core_manual = sorted({f for m in MANUAL_CORE for f in forms(m)})
    core = sorted(set(core_discovered) | set(core_manual))
    shared = sorted({f for l in shared_lemmas for f in forms(l)})

    seeds = {
        "core_discovered": core_discovered,
        "core_manual": core_manual,
        "core": core,
        "shared": shared,
        "manual_core_terms": sorted(ce.canonicalize(m) for m in MANUAL_CORE),
        "core_lemmas": sorted(core_lemmas),
        "shared_lemmas": sorted(shared_lemmas),
    }
    (DATA / "curated_seeds.json").write_text(json.dumps(seeds, ensure_ascii=False, indent=2), encoding="utf-8")
    return seeds


if __name__ == "__main__":
    s = build()
    print("core lemmas (discovered):", s["core_lemmas"])
    print("shared lemmas:", s["shared_lemmas"])
    print("manual core (hand-added, NOT discovered):", s["manual_core_terms"])
    print(f"forms — core_discovered: {len(s['core_discovered'])}, "
          f"core_manual: {len(s['core_manual'])}, shared: {len(s['shared'])}")
