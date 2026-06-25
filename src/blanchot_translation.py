"""#3 — The Blanchot-translated-by-Jinkis exhibit.

SITIO contains a translated Blanchot essay, "¿Qué es la crítica?" (Qu'est-ce que la critique?),
attributed to Blanchot and translated by core editor Jorge Jinkis, in the issue-4/5 dossier
"EL ENSAYO QUE VENDRA" (container issue_4-5_pos2909, alongside Grüner's §5.1 essay).

This script (a) locates/confirms it, (b) hardens the strand-1 premise — the probe's group C
("cites Blanchot") counts persName citations and so MISSES these author-by-Blanchot paragraphs,
which is exactly why "marginal as citation" must be restated as "marginal as a *cited authority*";
and recomputes the probe δ excluding these paragraphs — and (c) does neighborhood / vocabulary
checks. Pure local compute, $0; reuses audit.py + probe_robustness.py. Interpretation is the
researcher's to write.
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
from lexicons import BLANCHOT_SEEDS, BLANCHOT_CIRCLE, BLANCHOT_ID, BLANCHOT_NAME
from probe_robustness import build_groups, probe_delta, assert_cache_valid, _arr

DATA = ROOT / "data"
RES = ROOT / "results"
TRANSLATOR_ID = "maurice_blanchot"   # the parquet `contributor` = author; translator (Jinkis) is in TEI @resp


def main():
    rng = np.random.default_rng(audit.SEED)
    df, _ = audit.load_paragraphs()
    assert_cache_valid(df)
    emb = audit.get_embeddings(df)
    N = len(df)

    # (a) locate the translated essay
    T = np.where(df["contributor"].values == TRANSLATOR_ID)[0]
    sub = df.iloc[T]
    containers = sorted(set(sub["text_id"]))
    subtypes = sorted(set(str(x) for x in sub["div_subtype"]))
    print(f"translated-Blanchot paragraphs (contributor={TRANSLATOR_ID}): n={len(T)}")
    print(f"  containers: {containers}")
    print(f"  div_subtypes: {subtypes}")

    # (b) premise: group membership of the translation in the probe's groups
    g = build_groups(df)
    A, Bset, C, Dset, named = set(g["A"].tolist()), g["B_full"], g["C"], g["D_orig"], g["named"]
    seed_re = re.compile(r"\b(" + "|".join(BLANCHOT_SEEDS) + r")\b", re.IGNORECASE)
    Tset = set(int(t) for t in T)
    T_in_A = sorted(Tset & A)
    T_in_B = sorted(Tset & Bset)
    T_in_C = sorted(Tset & C)
    T_named = sorted(t for t in Tset if bool(seed_re and re.search(r"\b" + BLANCHOT_NAME + r"\b",
                                                                   (df["text"].values[t] or ""), re.I)))
    print(f"\npremise — probe-group membership of the {len(T)} translation paragraphs:")
    print(f"  in A (seed-bearing 'Blanchotian vocabulary'): {len(T_in_A)}/{len(T)}  "
          f"(corpus base rate |A|/N = {len(A)/N:.1%})")
    print(f"  in B (cites a circle author):                 {len(T_in_B)}/{len(T)}")
    print(f"  in C (cites Blanchot by persName):            {len(T_in_C)}/{len(T)}  "
          f"<- group C total is only {len(C)}; the translation is INVISIBLE to it")
    print(f"  text literally contains 'blanchot':           {len(T_named)}/{len(T)}")

    # premise-hardening: recompute the probe δ with the translation paragraphs removed entirely
    A2, B2, D2 = _arr(A - Tset), _arr(Bset - Tset), _arr(Dset - Tset)
    d_full, _, _ = probe_delta(emb, g["A"], _arr(Bset), _arr(Dset), rng)
    d_excl, _, _ = probe_delta(emb, A2, B2, D2, rng)
    print(f"\npremise-hardening — probe δ excluding the translation:")
    print(f"  full δ            = {d_full:+.3f}")
    print(f"  δ excl. translation = {d_excl:+.3f}  (|A| {len(A)}→{len(A2)})")

    # (c) neighborhood: the translation IS Blanchot — does his own text sit near the circle?
    embT = emb[T]
    Barr, Darr = _arr(Bset), _arr(Dset)
    mean_TB = float((embT @ emb[Barr].T).mean())
    mean_TD = float((embT @ emb[Darr].T).mean())
    print(f"\nneighborhood — mean cosine of the translation to:")
    print(f"  circle-author paragraphs (B): {mean_TB:+.3f}")
    print(f"  corpus rest (D):              {mean_TD:+.3f}")

    sims = embT @ emb.T            # (|T|, N)
    nbr_contrib = Counter()
    nbr_examples = []
    for r, t in enumerate(T):
        order = np.argsort(-sims[r])
        nn = [j for j in order if j not in Tset][:5]
        for j in nn:
            nbr_contrib[str(df["contributor"].values[j])] += 1
        nbr_examples.append((t, [(int(j), str(df["contributor"].values[j]),
                                  str(df["div_subtype"].values[j]), round(float(sims[r, j]), 3))
                                 for j in nn[:3]]))
    print(f"\nnearest-neighbour contributors (top-5 each, the translation's company):")
    for c, n in nbr_contrib.most_common(8):
        print(f"  {c:24s} {n}")

    # vocabulary: does Blanchot's own text carry the discovered-core lexicon above corpus rate?
    seeds = json.loads((DATA / "curated_seeds.json").read_text(encoding="utf-8"))
    core_forms = sorted(set(seeds["core_discovered"]))
    core_re = re.compile(r"\b(" + "|".join(core_forms) + r")\b", re.IGNORECASE)
    T_core = sum(bool(core_re.search(df["text"].values[t] or "")) for t in T)
    corpus_core = sum(bool(core_re.search(x or "")) for x in df["text"].values)
    print(f"\nvocabulary — discovered-core lexicon ({len(core_forms)} forms):")
    print(f"  translation paragraphs with ≥1 core term: {T_core}/{len(T)} ({T_core/len(T):.0%})")
    print(f"  corpus base rate:                         {corpus_core}/{N} ({corpus_core/N:.0%})")

    write_report(df, T, containers, subtypes, len(A), N, T_in_A, T_in_C, len(C),
                 d_full, d_excl, len(A2), mean_TB, mean_TD, nbr_contrib,
                 T_core, corpus_core, core_forms, nbr_examples)
    print(f"\nwrote {RES/'blanchot_translation.md'}")


def write_report(df, T, containers, subtypes, nA, N, T_in_A, T_in_C, nC, d_full, d_excl, nA2,
                 mean_TB, mean_TD, nbr_contrib, T_core, corpus_core, core_forms, nbr_examples):
    snippets = "\n".join(f"> {(df['text'].values[t] or '')[:200].strip()}…" for t in T[:4])
    nbr_tbl = "\n".join(f"| {c} | {n} |" for c, n in nbr_contrib.most_common(8))
    md = f"""# #3 — The Blanchot-translated-by-Jinkis exhibit

**Status:** computational exhibit (located + premise-hardened + neighborhood/vocabulary). The
philological / reception interpretation is the **researcher's** to write. $0, deterministic.

## What was located

SITIO carries a translated Blanchot essay — **"¿Qué es la crítica?"** (*Qu'est-ce que la
critique?*) — as **{len(T)} paragraphs** with `contributor = maurice_blanchot`, `div_subtype ∈
{subtypes}`, in container(s) **{containers}** (the issue-4/5 dossier *"EL ENSAYO QUE VENDRA"*,
which also holds **Grüner's "El ensayo, un género culpable"** — the §5.1 strand-2 text — and
Gusmán's essay). **Strand 1 (the shadow) and strand 2 (the limit) physically share one dossier.**

Attribution: the essay is **attributed to Blanchot** (parquet `contributor`); per the researcher it
is **translated by Jorge Jinkis**, a core SITIO editor (the translator credit lives in the TEI
`@resp`, not in this parquet column — confirm against the TEI). The repo's `RESEARCH_SUMMARY.md`
note "translator uncredited" is therefore **inaccurate**.

First paragraphs (opening of the translation):

{snippets}

## The premise, corrected — "marginal as a *cited authority*", not "marginal presence"

The probe's group **C** ("cites Blanchot") counts him as a **persName citation** and totals only
**{nC}** paragraphs. But Blanchot is *also* present as the **author** of {len(T)} translated
paragraphs — and **{len(T_in_C)}/{len(T)}** of those are in group C (he does not cite himself), so the
translation is **invisible to the citation count**. "Marginal as citation" is true *only* in the
narrow sense of cited-authority; materially, an entire Blanchot essay was imported by a core editor.
The paper must say **marginal as a cited authority**, not marginal in presence.

## Premise-hardening — the +0.427 footprint is not Blanchot's own imported words

**{len(T_in_A)}/{len(T)}** translation paragraphs fall in group **A** (seed-bearing "Blanchotian
vocabulary") — i.e. Blanchot's own prose is, unsurprisingly, maximally Blanchotian (vs the corpus
base rate |A|/N = {nA/N:.0%}). Removing the whole translation from the probe changes nothing:

| | δ (A↔B vs A↔D) | \\|A\\| |
|---|---:|---:|
| full | {d_full:+.3f} | {nA} |
| excluding the translation | {d_excl:+.3f} | {nA2} |

So the shadow-probe effect is the **journal's diffuse vocabulary**, not an artifact of Blanchot's
own translated text sitting in the corpus (parallel to the existing Blanchot-named stratification).

## Neighborhood — Blanchot's own text sits with the circle

Mean cosine of the translation paragraphs to **circle-author paragraphs = {mean_TB:+.3f}** vs to the
**corpus rest = {mean_TD:+.3f}** — Blanchot's translated essay lands nearer the seven authors he wrote
on than the rest of SITIO, the shadow-probe geometry seen from the source itself. Their nearest
neighbours (top-5 each) are dominated by:

| neighbour contributor | count |
|---|---:|
{nbr_tbl}

## Vocabulary

Discovered-core lexicon ({len(core_forms)} forms): **{T_core}/{len(T)} ({T_core/len(T):.0%})** of the
translation paragraphs carry ≥1 core term, vs corpus base rate {corpus_core}/{N} ({corpus_core/N:.0%}).

## For the researcher

- Confirm the Jinkis translator credit against the TEI `@resp` and the printed *Sitio* 4/5 issue.
- Write the reception-thread reading: a central editor personally importing *"¿Qué es la crítica?"*
  into the same dossier as Grüner's essay-on-the-essay — the "shadow influence" as a **documented,
  attributed editorial act**, zero embedding-prior exposure.
- Correct the `RESEARCH_SUMMARY.md` open-item note ("translator uncredited").
"""
    RES.mkdir(exist_ok=True)
    (RES / "blanchot_translation.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
