"""SP1 Stage 2 — validate the concept layer.

(a) Concept-embedding coherence: do paragraphs sharing an extracted concept lemma sit
    closer in embedding space than random pairs? (persName sanity check, generalized.)
    Reuses audit.py's pair-sampling + Cliff's delta. Robust C<->E relationship claim.
(b) tf-idf overlap: how much do the LLM candidates and a no-LLM frequency baseline agree?

Reuses cached data/embeddings.npy (no Modal cost).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit
import concept_extract as ce

DATA = ce.DATA
RES = ce.ROOT / "results"


def coherence(df: pd.DataFrame, layer: pd.DataFrame, emb: np.ndarray,
              rng: np.random.Generator) -> dict:
    """df and emb are aligned (audit order). layer has a `lemmas` column keyed by
    para_id. Build shared-concept pairs (paragraphs sharing >=1 lemma) and compare
    their cosines to random pairs."""
    # layer is positionally aligned with df/emb (all from audit.load_paragraphs(), same
    # order). para_id is None for ~3,429/3,476 paragraphs, so a para_id join collapses
    # them to one entry — use positional alignment instead.
    assert len(layer) == len(df) and (
        df["para_id"].fillna("\x00").values == layer["para_id"].fillna("\x00").values
    ).all(), "concept layer is not aligned with audit paragraph order"
    lemmas = [list(ls) for ls in layer["lemmas"]]

    inv: dict[str, list[int]] = {}
    for i, ls in enumerate(lemmas):
        for l in set(ls):
            inv.setdefault(l, []).append(i)

    shared: set[tuple[int, int]] = set()
    for l, idxs in inv.items():
        if len(idxs) < 2:
            continue
        s = sorted(idxs)
        for a in range(len(s)):
            for b in range(a + 1, len(s)):
                shared.add((s[a], s[b]))
    shared_arr = (np.array(sorted(shared), dtype=np.int64)
                  if shared else np.empty((0, 2), dtype=np.int64))
    if len(shared_arr) > audit.N_PAIRS:
        sel = rng.choice(len(shared_arr), audit.N_PAIRS, replace=False)
        shared_arr = shared_arr[sel]
    base = audit.sample_pairs(list(range(len(df))), audit.N_PAIRS, rng)

    c_shared = audit.cosines(emb, shared_arr)
    c_base = audit.cosines(emb, base)
    return {
        "n_shared_pairs": int(len(c_shared)),
        "n_baseline_pairs": int(len(c_base)),
        "median_shared": float(np.median(c_shared)) if len(c_shared) else float("nan"),
        "median_baseline": float(np.median(c_base)) if len(c_base) else float("nan"),
        "cliffs_delta": audit.cliffs_delta(c_shared, c_base),
    }


def graded_coherence(df: pd.DataFrame, layer: pd.DataFrame, emb: np.ndarray,
                     rng: np.random.Generator, n_pairs: int = 200_000) -> dict:
    """Threshold-free coherence: across random paragraph pairs, does a higher fraction
    of shared concepts (lemma-set Jaccard) predict higher cosine? Lets rare/specific
    overlap count, unlike the binary 'share >=1' test."""
    from scipy.stats import spearmanr
    assert len(layer) == len(df), "concept layer is not aligned with audit paragraph order"
    lemsets = [set(ls) for ls in layer["lemmas"]]
    n = len(df)
    a = rng.integers(0, n, n_pairs)
    b = rng.integers(0, n, n_pairs)
    mask = a != b
    a, b = a[mask], b[mask]
    cos = audit.cosines(emb, np.stack([a, b], axis=1))
    jac = np.empty(len(a))
    for k in range(len(a)):
        sa, sb = lemsets[a[k]], lemsets[b[k]]
        union = len(sa | sb)
        jac[k] = (len(sa & sb) / union) if union else 0.0
    rho, p = spearmanr(jac, cos)
    bins = [(-0.001, 0.0), (0.0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01)]
    table = []
    for lo, hi in bins:
        sel = (jac > lo) & (jac <= hi)
        if int(sel.sum()) > 0:
            label = "0 (no shared concept)" if hi == 0.0 else f"({lo:.2f}, {hi:.2f}]"
            table.append((label, int(sel.sum()), float(np.median(cos[sel]))))
    return {"n_pairs": int(len(a)), "spearman_rho": float(rho),
            "spearman_p": float(p), "cos_by_jaccard": table}


def tfidf_overlap(top_n: int = 100) -> dict:
    """Jaccard of the top-N LLM candidate lemmas vs the top-N tf-idf lemmas."""
    cand = pd.read_csv(RES / "lexicon_candidates.csv", encoding="utf-8-sig")
    llm_top = set(cand.sort_values("n_paragraphs", ascending=False)["lemma"].head(top_n))
    tfidf = cand[cand["tfidf_rank"].notna()].copy()
    tfidf["tfidf_rank"] = tfidf["tfidf_rank"].astype(int)
    tf_top = set(tfidf.sort_values("tfidf_rank")["lemma"].head(top_n))
    inter = llm_top & tf_top
    union = llm_top | tf_top
    return {
        "top_n": top_n,
        "jaccard": round(len(inter) / len(union), 3) if union else 0.0,
        "llm_only_examples": sorted(llm_top - tf_top)[:10],
    }


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(audit.SEED)
    df, _ = audit.load_paragraphs()
    emb = audit.get_embeddings(df)
    layer = pd.read_parquet(DATA / "concept_layer.parquet")

    coh = coherence(df, layer, emb, rng)
    grad = graded_coherence(df, layer, emb, rng)
    ov = tfidf_overlap()
    print("coherence(binary):", coh)
    print("coherence(graded):", grad)
    print("tfidf overlap:", ov)

    delta = coh["cliffs_delta"]
    rho = grad["spearman_rho"]
    zero_n = next((n for lbl, n, _ in grad["cos_by_jaccard"] if lbl.startswith("0")), 0)
    share_frac = (1 - zero_n / grad["n_pairs"]) if grad["n_pairs"] else 0.0
    p_str = "< 1e-300" if grad["spearman_p"] == 0.0 else f"{grad['spearman_p']:.1e}"

    if delta >= 0.33:
        binary_read = (f"This is a strong positive effect (delta {delta:+.3f}, vs Step 1 references "
                       "Blanchot +0.427 and persName +0.852). Sharing even one extracted concept is "
                       f"selective — only {share_frac:.1%} of random pairs share any — so "
                       "shared-concept pairs sitting closer is a real signal, not an artifact of "
                       "ubiquitous vocabulary.")
    elif delta >= 0.15:
        binary_read = (f"This is a modest positive effect (delta {delta:+.3f}); shared-concept pairs are "
                       f"somewhat closer than random ({share_frac:.1%} of random pairs share a concept).")
    else:
        binary_read = (f"This is ~0 (delta {delta:+.3f}): shared-concept pairs are indistinguishable from "
                       "random, indicating the concept layer is too broad to discriminate.")

    if rho >= 0.15:
        verdict = (f"Specific concept overlap tracks embedding similarity (graded rho {rho:+.3f}), and the "
                   f"binary test agrees (Cliff's delta {delta:+.3f}). The concept layer coheres with "
                   "distributional semantics at corpus scale, so SP2/SP3 may use C as a (graded) semantic "
                   "signal — noting common concepts carry less discriminative weight than rare ones.")
    elif rho >= 0.05:
        verdict = (f"Weak graded coherence (rho {rho:+.3f}; binary delta {delta:+.3f}): specific overlap "
                   "helps a little, but the layer is mostly coarse. SP2/SP3 should treat C as a rough "
                   "topical signal, not a fine one.")
    else:
        verdict = (f"No meaningful graded coherence (rho {rho:+.3f}; binary delta {delta:+.3f}): the open "
                   "concept layer does not align with distributional semantics beyond chance at finer "
                   "granularity. SP2/SP3 should rely on C as raw vocabulary, not a semantic signal.")

    bin_rows = "\n".join(f"| {lbl} | {n:,} | {med:.4f} |" for lbl, n, med in grad["cos_by_jaccard"])
    md = f"""# Concept layer validation (SP1 Stage 2)

## (a) Concept-embedding coherence — binary (share >=1 concept lemma)

Do paragraphs sharing >=1 extracted concept lemma sit closer in embedding space than random
pairs? (persName sanity check generalized.)

| | shared-concept | random |
|---|---:|---:|
| median cosine | {coh['median_shared']:.4f} | {coh['median_baseline']:.4f} |
| n pairs | {coh['n_shared_pairs']:,} | {coh['n_baseline_pairs']:,} |

**Cliff's delta = {delta:+.3f}.** {binary_read}

## (b) Concept-embedding coherence — graded (degree of concept overlap)

Across {grad['n_pairs']:,} random paragraph pairs, does a higher *fraction* of shared concepts
(lemma-set Jaccard) predict higher cosine? Threshold-free; lets rare, specific overlap count.

**Spearman rho(Jaccard, cosine) = {rho:+.3f}** (p = {p_str}).

Median cosine by concept-overlap (Jaccard) bin:

| Jaccard overlap | n pairs | median cosine |
|---|---:|---:|
{bin_rows}

## (c) LLM candidates vs no-LLM tf-idf baseline

Top-{ov['top_n']} overlap (Jaccard) between the LLM concept candidates and a tf-idf frequency
baseline over the same lemmatized corpus: **{ov['jaccard']}**.

High overlap -> the LLM mostly rediscovers what frequency alone would surface. LLM-only
examples: {', '.join(ov['llm_only_examples']) or '(none)'}.

## Reading for SP2/SP3

{verdict}
"""
    (RES / "concept_layer_validation.md").write_text(md, encoding="utf-8")
    print(f"wrote {RES / 'concept_layer_validation.md'}")


if __name__ == "__main__":
    main()
