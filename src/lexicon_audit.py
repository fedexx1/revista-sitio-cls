"""SP1 Stage 3 — Blanchot re-audit across curated seed sets.

Runs audit.blanchot_probe (+ stratified no-Blanchot variant) with five seed sets and
tabulates the A<->B Cliff's delta vs the A<->D baseline:
  1. baseline control = BLANCHOT_SEEDS (must reproduce Step 1 ~ +0.427)
  2. core - discovered only  = LLM-discovered, human-tagged 'core' surface forms
  3. core - discovered + manual = adds the hand-injected afuera/dehors/neutro
  4. core + shared = full curated lexicon
  5. track-2 ceiling = top enrichment-ranked lemmas (labeled over-fit upper bound)
Reuses cached embeddings (0 Modal cost).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit
import concept_extract as ce
from lexicons import BLANCHOT_SEEDS

DATA = ce.DATA
RES = ce.ROOT / "results"
CEILING_N = 30  # top enrichment-ranked lemmas for the track-2 ceiling


def _ceiling_seeds() -> list[str]:
    cand = pd.read_csv(RES / "lexicon_candidates.csv", encoding="utf-8-sig", keep_default_na=False)
    top = cand[cand["n_paragraphs"] >= 5].nlargest(CEILING_N, "enrichment_score")
    return sorted({ce.canonicalize(l) for l in top["lemma"]})


def _probe(df, emb, seeds):
    rng = np.random.default_rng(audit.SEED)  # fixed seed per run for comparability
    rows, groups, _ = audit.blanchot_probe(df, emb, rng, seeds=seeds)
    ab = next(r for r in rows if r["pair_class"] == "A↔B")
    ab_s = next(r for r in rows if r["pair_class"] == "A↔B (no Blanchot)")
    strat = ab_s["cliffs_delta_vs_AD"] if ab_s["cliffs_delta_vs_AD"] is not None else float("nan")
    return {"n_A": groups["n_A"], "n_seeds": len(seeds),
            "delta": ab["cliffs_delta_vs_AD"], "median_AB": ab["median"], "delta_strat": strat}


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    audit.verify_lexicon_ids()
    df, _ = audit.load_paragraphs()
    emb = audit.get_embeddings(df)
    cur = json.loads((DATA / "curated_seeds.json").read_text(encoding="utf-8"))

    sets = {
        "baseline (BLANCHOT_SEEDS)": list(BLANCHOT_SEEDS),
        "core — discovered only": cur["core_discovered"],
        "core — discovered + manual (added by us)": cur["core"],
        "core + shared": sorted(set(cur["core"]) | set(cur["shared"])),
        "track-2 ceiling (top enrichment)": _ceiling_seeds(),
    }
    res = {name: _probe(df, emb, seeds) for name, seeds in sets.items()}
    for name, r in res.items():
        print(f"{name:42s} n_seeds={r['n_seeds']:4d} n_A={r['n_A']:5d} "
              f"delta={r['delta']:+.3f} strat={r['delta_strat']:+.3f}")

    rows_md = "\n".join(
        f"| {name} | {r['n_seeds']} | {r['n_A']} | {r['delta']:+.3f} | {r['delta_strat']:+.3f} |"
        for name, r in res.items()
    )
    md = f"""# Blanchot re-audit — curated lexicon (SP1 Stage 3)

Does a data-derived, human-curated lexicon strengthen / preserve / dilute the Step 1
Blanchot shadow probe? Cliff's δ is the A↔B (vocabulary ↔ seven-author circle) effect vs
the A↔D baseline; stratified δ removes paragraphs naming Blanchot. Seeds were curated on
theoretical grounds (NOT the enrichment signal) and committed before this audit ran.

| seed set | n seeds (forms) | n_A | Cliff's δ | stratified δ |
|---|---:|---:|---:|---:|
{rows_md}

Step 1 reference: BLANCHOT_SEEDS gave δ ≈ +0.427 (the baseline row is the regression control).

**Provenance note (important).** `afuera`, `dehors`, `neutro` were **added by hand by the
editors** — the LLM did not surface them. They are NOT part of the data-driven discovery.
The two core rows isolate their effect: *core — discovered only* uses solely the
LLM-discovered, human-tagged core lemmas; *core — discovered + manual* adds our three terms.
The gap between those rows is the contribution of the hand-injected vocabulary, and must be
read as such — no row here is "purely discovered" except *core — discovered only*.

## Reading

- If **core — discovered only** holds (δ near/above baseline, surviving stratification), the
  Blanchot-specific effect stands on discovered vocabulary alone — the strongest claim.
- If the effect appears only once the **manual** terms are added (gap between the two core
  rows), then it is expert-injected, not discovered — state that explicitly; the claim
  becomes "expert-curated, LLM-assisted," not "the LLM rediscovered Blanchot's lexicon."
- If only **core + shared** holds, the cluster reflects the broader poststructuralist
  footprint (escritura/lectura/espacio/muerte), not Blanchot specifically.
- If the curated sets **dilute** δ vs baseline, the original 15 seeds did specific work / the
  effect is fragile to lexicon breadth.
- The **track-2 ceiling** shows how high δ goes when seeds are picked by the enrichment signal
  itself; it is NOT evidence for the thesis — it bounds the over-fitting.
"""
    (RES / "lexicon_expansion.md").write_text(md, encoding="utf-8")
    print(f"wrote {RES / 'lexicon_expansion.md'}")


if __name__ == "__main__":
    main()
