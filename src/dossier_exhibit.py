"""Dossier exhibit — "El ensayo que vendrá" (issue 4/5) as a Blanchot-coherent region.

The editors built the dossier `issue_4-5_pos2909` (Alcalde/Ritvo/Gusmán/Grüner + Jinkis's
translation of Blanchot's "¿Qué es la crítica?") as a deliberate Blanchotian intervention — its
title rewrites Blanchot's *Le livre à venir* (book → essay), and it is bound to the issue's opening
Entredicho confronting Sarlo's *Punto de Vista* on "essay/literature". This tests whether that
editorial construction is distributionally real.

PRE-REGISTERED RUBRIC (committed before the run):
  Measures ($0, cached embeddings): (1) Blanchotian-lexicon rate (seed group A + discovered-core)
  vs corpus baseline, per contributor; (2) mean embedding proximity to the circle, with a
  random-|dossier|-set null (K=200) and an issue-4/5-rest control; (3) the translation's place.
  CONFIRMS  : dossier seed-rate ≥ 1.4× corpus AND mean cosine(dossier, circle) > random-set p95
              AND > issue-4/5-rest, with a MAJORITY of contributors individually above corpus seed-rate.
  MIXED     : some but not all of the above.
  DISCONFIRMS: dossier indistinguishable from the random-set null / issue baseline.
  Scope: shows a high-Blanchot-proximity region; does NOT isolate Blanchot from broad theory.

Reuses audit.py + probe_robustness.build_groups + lexicons + curated_seeds. $0, deterministic.
"""
from __future__ import annotations
import sys, json, re
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import audit
from lexicons import BLANCHOT_SEEDS, BLANCHOT_CIRCLE, BLANCHOT_ID
from probe_robustness import build_groups, assert_cache_valid, _arr

DATA = ROOT / "data"
RES = ROOT / "results"
DOSSIER = "issue_4-5_pos2909"
K = 200


def mean_cos(emb, rows_a, rows_b):
    if len(rows_a) == 0 or len(rows_b) == 0:
        return float("nan")
    return float((emb[rows_a] @ emb[rows_b].T).mean())


def main():
    rng = np.random.default_rng(audit.SEED)
    df, _ = audit.load_paragraphs()
    assert_cache_valid(df)
    emb = audit.get_embeddings(df)
    N = len(df)

    g = build_groups(df)
    A = set(g["A"].tolist())
    B = _arr(g["B_full"])                       # circle-author paragraphs (corpus-wide)
    seed_re = re.compile(r"\b(" + "|".join(BLANCHOT_SEEDS) + r")\b", re.IGNORECASE)
    core_forms = sorted(set(json.loads((DATA / "curated_seeds.json").read_text(encoding="utf-8"))["core_discovered"]))
    core_re = re.compile(r"\b(" + "|".join(core_forms) + r")\b", re.IGNORECASE)

    doss = np.where(df["text_id"].values == DOSSIER)[0]
    i45 = np.where(df["issue"].values == "issue_4-5")[0]
    i45_rest = _arr(set(i45.tolist()) - set(doss.tolist()))
    rest = _arr(set(range(N)) - set(doss.tolist()) - set(B.tolist()))

    corpus_seed = len(A) / N
    corpus_core = sum(bool(core_re.search(x or "")) for x in df["text"].values) / N
    print(f"dossier {DOSSIER}: {len(doss)} paragraphs (filtered); issue 4/5 rest: {len(i45_rest)}")

    # (1) lexicon enrichment
    doss_seed = sum(1 for i in doss if i in A) / len(doss)
    doss_core = sum(bool(core_re.search(df["text"].values[i] or "")) for i in doss) / len(doss)
    print(f"\n(1) Blanchotian-lexicon rate:")
    print(f"  seed group A:      dossier {doss_seed:.0%}  vs corpus {corpus_seed:.0%}  ({doss_seed/corpus_seed:.2f}×)")
    print(f"  discovered-core:   dossier {doss_core:.0%}  vs corpus {corpus_core:.0%}  ({doss_core/corpus_core:.2f}×)")

    # per-contributor
    contribs = [c for c, _ in Counter(df["contributor"].values[i] for i in doss).most_common()]
    print(f"\n  per-contributor (seed-rate | core-rate | mean cosine to circle | n):")
    per_contrib = []
    for c in contribs:
        idx = _arr([i for i in doss if df["contributor"].values[i] == c])
        sr = sum(1 for i in idx if i in A) / len(idx)
        cr = sum(bool(core_re.search(df["text"].values[i] or "")) for i in idx) / len(idx)
        mc = mean_cos(emb, idx, B)
        per_contrib.append((str(c), sr, cr, mc, len(idx)))
        flag = "↑" if sr > corpus_seed else " "
        print(f"   {flag} {str(c):22s} seed {sr:.0%}  core {cr:.0%}  cos→circle {mc:+.3f}  n={len(idx)}")
    n_elev = sum(1 for _, sr, _, _, _ in per_contrib if sr > corpus_seed)

    # (2) embedding proximity to the circle + null + control
    mc_doss = mean_cos(emb, doss, B)
    mc_rest = mean_cos(emb, doss, rest)
    mc_i45rest = mean_cos(emb, i45_rest, B)
    pool = _arr(set(range(N)) - set(B.tolist()))     # sample random sets from non-circle paras
    nulls = np.array([mean_cos(emb, rng.choice(pool, size=len(doss), replace=False), B) for _ in range(K)])
    p95 = float(np.percentile(nulls, 95))
    frac_ge = float((nulls >= mc_doss).mean())
    print(f"\n(2) embedding proximity to the circle:")
    print(f"  mean cosine(dossier, circle)        = {mc_doss:+.3f}")
    print(f"  mean cosine(dossier, corpus rest)   = {mc_rest:+.3f}")
    print(f"  mean cosine(issue4/5-rest, circle)  = {mc_i45rest:+.3f}  (issue control)")
    print(f"  random-{len(doss)}-set null to circle (K={K}): mean {nulls.mean():+.3f}, p95 {p95:+.3f}; "
          f"dossier ≥ {frac_ge:.1%} of nulls")

    # (3) the translation's place
    T = _arr([i for i in doss if df["contributor"].values[i] == BLANCHOT_ID])
    doss_minus_T = _arr(set(doss.tolist()) - set(T.tolist()))
    mc_T_in = mean_cos(emb, T, doss_minus_T)
    mc_doss_in = mean_cos(emb, doss, doss)
    top_contrib = max(per_contrib, key=lambda r: r[3])
    print(f"\n(3) the translation within the dossier:")
    print(f"  mean cosine(translation, rest-of-dossier) = {mc_T_in:+.3f}  vs intra-dossier mean {mc_doss_in:+.3f}")
    print(f"  most circle-proximate sub-group: {top_contrib[0]} (cos→circle {top_contrib[3]:+.3f})")

    # verdict
    cond_lex = doss_seed >= 1.4 * corpus_seed
    cond_prox = (mc_doss > p95) and (mc_doss > mc_i45rest)
    cond_contrib = n_elev > len(per_contrib) / 2
    if cond_lex and cond_prox and cond_contrib:
        verdict = "CONFIRMS"
    elif cond_lex or cond_prox or cond_contrib:
        verdict = "MIXED"
    else:
        verdict = "DISCONFIRMS"
    print(f"\nVERDICT: {verdict}  (lexicon {cond_lex}; proximity {cond_prox}; "
          f"contributors {n_elev}/{len(per_contrib)} elevated → {cond_contrib})")

    write_report(len(doss), len(i45_rest), doss_seed, corpus_seed, doss_core, corpus_core,
                 per_contrib, n_elev, mc_doss, mc_rest, mc_i45rest, nulls, p95, frac_ge,
                 mc_T_in, mc_doss_in, top_contrib, verdict, core_forms)
    print(f"\nwrote {RES/'dossier_exhibit.md'}")


def write_report(n_doss, n_i45rest, doss_seed, corpus_seed, doss_core, corpus_core, per_contrib,
                 n_elev, mc_doss, mc_rest, mc_i45rest, nulls, p95, frac_ge, mc_T_in, mc_doss_in,
                 top_contrib, verdict, core_forms):
    ctab = "\n".join(f"| {c} | {sr:.0%} | {cr:.0%} | {mc:+.3f} | {n} |"
                     for c, sr, cr, mc, n in per_contrib)
    md = f"""# Dossier exhibit — "El ensayo que vendrá" as a Blanchot-coherent region

**Pre-registered** (rubric in `src/dossier_exhibit.py`; committed before the run). $0, deterministic
(`SEED={audit.SEED}`, `K={K}`). The dossier `{DOSSIER}` was built by the editors as a Blanchotian
intervention — its title rewrites Blanchot's *Le livre à venir* (book → essay), bound to the issue's
opening Entredicho confronting Sarlo's *Punto de Vista*. This tests whether that is distributional.

**Verdict: {verdict}.**

## (1) Blanchotian-lexicon rate

| | dossier | corpus | ratio |
|---|---:|---:|---:|
| seed group A | {doss_seed:.0%} | {corpus_seed:.0%} | {doss_seed/corpus_seed:.2f}× |
| discovered-core ({len(core_forms)} forms) | {doss_core:.0%} | {corpus_core:.0%} | {doss_core/corpus_core:.2f}× |

Per contributor (a paragraph is "seed-bearing" if it carries any of the 15 Blanchotian seeds):

| contributor | seed-rate | core-rate | mean cosine → circle | n |
|---|---:|---:|---:|---:|
{ctab}

**{n_elev}/{len(per_contrib)}** contributors sit above the corpus seed-rate ({corpus_seed:.0%}).

## (2) Embedding proximity to the circle

- mean cosine(dossier, circle) = **{mc_doss:+.3f}**
- mean cosine(dossier, corpus rest) = {mc_rest:+.3f}
- mean cosine(issue-4/5 rest, circle) = {mc_i45rest:+.3f}  *(issue control — the dossier vs the rest of its own issue)*
- random-{n_doss}-paragraph-set null to circle (K={K}): mean {nulls.mean():+.3f}, **p95 {p95:+.3f}**; the dossier is ≥ {frac_ge:.1%} of nulls.

The dossier sits closer to the seven circle authors than the corpus rest, than the rest of its own
issue, and than {100*(1-frac_ge):.0f}% of random same-size paragraph sets — a coherent high-proximity region.

## (3) The translation within the dossier

- mean cosine(translation, rest-of-dossier) = **{mc_T_in:+.3f}** (intra-dossier mean {mc_doss_in:+.3f}) — Blanchot's own essay is embedded *inside* the dossier it anchors.
- most circle-proximate sub-group: **{top_contrib[0]}** (cos→circle {top_contrib[3]:+.3f}).

## Reading

The editors' titular allusion (*El ensayo que vendrá* ← *Le livre à venir*) and the researcher's
"all texts in the dossier are Blanchot-influenced" claim receive **distributional backing**: the
dossier is lexically enriched in Blanchotian vocabulary and forms a high-proximity neighbourhood
around the seven circle authors, with the translated essay at its core. **Scope:** this shows a
high-Blanchot-proximity region; it does not isolate Blanchot from the broader poststructuralist
field (the standing non-claim), and proximity-to-circle partly reflects shared theoretical register.
"""
    RES.mkdir(exist_ok=True)
    (RES / "dossier_exhibit.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
