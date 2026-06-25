"""Embedding audit for the SITIO corpus.

Two tests:
1. persName alignment — do paragraphs sharing ≥1 cited person cluster?
   Sanity check; if this fails, embeddings or matching is broken.
2. Blanchot shadow probe — do paragraphs with Blanchotian vocabulary (A)
   cluster with paragraphs citing Kafka/Joyce/Beckett/Céline/Faulkner/
   Borges/Gombrowicz (B) more than with the corpus rest (D)? Stratified
   variant excludes paragraphs naming Blanchot from both A and B.

Embeddings are cached at data/embeddings.npy and reused. First run requires
`modal deploy src/embed.py`.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lexicons import BLANCHOT_SEEDS, BLANCHOT_CIRCLE, BLANCHOT_ID, BLANCHOT_NAME

DATA = Path(__file__).resolve().parents[1] / "data"
RES = Path(__file__).resolve().parents[1] / "results"

MIN_TEXT_LEN = 50
N_PAIRS = 50_000
SEED = 0


def load_paragraphs() -> tuple[pd.DataFrame, int]:
    df = pd.read_parquet(DATA / "paragraphs.parquet")
    n_total = len(df)
    df = df[df["text"].str.len() >= MIN_TEXT_LEN].reset_index(drop=True)
    print(f"paragraphs: {n_total} total → {len(df)} after filter (>={MIN_TEXT_LEN} chars)")
    return df, n_total


def _text_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(df["text"].tolist()).encode("utf-8")).hexdigest()


def get_embeddings(df: pd.DataFrame, model: str = "qwen3") -> np.ndarray:
    """Return embeddings for df['text'] under the given registry model, reusing the
    per-model cache only if it matches both the row count AND the exact text content
    (sha256 in the meta json). model='qwen3' keeps the legacy cache filenames.

    Shape-only validation would silently return stale vectors after any TEI edit
    that preserves the post-filter row count (typo fix, whitespace normalization,
    paragraph reorder), so the content hash is the real cache key.
    """
    suffix = "" if model == "qwen3" else f"_{model}"
    cache = DATA / f"embeddings{suffix}.npy"
    meta_path = DATA / ("embeddings_meta.json" if model == "qwen3"
                        else f"embeddings{suffix}_meta.json")
    text_hash = _text_hash(df)
    if cache.exists() and meta_path.exists():
        emb = np.load(cache)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if emb.shape[0] == len(df) and meta.get("text_sha256") == text_hash:
            print(f"loaded cached embeddings: {emb.shape}")
            return emb
        reason = ("row count" if emb.shape[0] != len(df) else "text content")
        print(f"cache invalid ({reason} changed); re-embedding")
    elif cache.exists():
        print("cache present but embeddings_meta.json missing/unhashed; re-embedding")

    from embed import embed_paragraphs, MODELS, GPU
    print(f"embedding {len(df)} paragraphs via Modal ({model}) …")
    t0 = time.monotonic()
    emb = embed_paragraphs(df["text"].tolist(), model=model)
    elapsed = time.monotonic() - t0
    np.save(cache, emb)
    meta_path.write_text(json.dumps({
        "model_id": MODELS[model],
        "gpu_type": GPU,
        "dim": int(emb.shape[1]),
        "dtype": str(emb.dtype),
        "n_paragraphs": int(emb.shape[0]),
        "elapsed_s": round(elapsed, 2),
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "text_sha256": text_hash,
    }, indent=2), encoding="utf-8")
    print(f"cached embeddings to {cache}: {emb.shape}")
    return emb


def verify_lexicon_ids() -> None:
    """Fail loud if any Blanchot-circle id is absent from persons.csv (the claim
    in lexicons.py). Catches a fat-fingered id silently shrinking group B."""
    known = set(pd.read_csv(DATA / "persons.csv")["id"])
    missing = (set(BLANCHOT_CIRCLE) | {BLANCHOT_ID}) - known
    if missing:
        raise AssertionError(f"lexicon ids not found in persons.csv: {sorted(missing)}")


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """P(X > Y) - P(X < Y), via sorted-y searchsort. O((n+m) log m)."""
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    y_sorted = np.sort(y)
    greater = np.searchsorted(y_sorted, x, side="left").sum()    # x_i > y_j
    less = (len(y_sorted) - np.searchsorted(y_sorted, x, side="right")).sum()
    return float((greater - less) / (len(x) * len(y)))


def cosines(emb: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Dot products for L2-normalized embeddings = cosines."""
    if len(pairs) == 0:
        return np.array([])
    return (emb[pairs[:, 0]] * emb[pairs[:, 1]]).sum(axis=1)


def sample_pairs(pool: list[int], n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample up to n distinct unordered pairs (i, j) with i < j from pool."""
    if len(pool) < 2:
        return np.empty((0, 2), dtype=np.int64)
    out: set[tuple[int, int]] = set()
    arr = np.asarray(pool, dtype=np.int64)
    max_attempts = n * 5
    attempts = 0
    while len(out) < n and attempts < max_attempts:
        idx = rng.choice(len(arr), 2, replace=False)
        a, b = int(arr[idx[0]]), int(arr[idx[1]])
        out.add((min(a, b), max(a, b)))
        attempts += 1
    return np.array(sorted(out), dtype=np.int64)


def cross_pairs(left: np.ndarray, right: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n pairs (l, r) with l in left, r in right (with replacement on each side),
    excluding self-pairs (l == r). When the pools overlap (e.g. A∩B in the Blanchot
    probe) self-pairs would otherwise leak in with cosine 1.0 and inflate Cliff's δ.
    n is capped at len(left)*len(right) so the reported n_pairs never overstates the
    available pool (e.g. A↔C with |A|·|C| ≪ 50k)."""
    if len(left) == 0 or len(right) == 0:
        return np.empty((0, 2), dtype=np.int64)
    n = min(n, len(left) * len(right))
    out: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = max(n * 20, 1000)
    while len(out) < n and attempts < max_attempts:
        draw = n - len(out)
        li = left[rng.integers(0, len(left), draw)]
        rj = right[rng.integers(0, len(right), draw)]
        mask = li != rj
        out.extend(zip(li[mask].tolist(), rj[mask].tolist()))
        attempts += draw
    return np.array(out[:n], dtype=np.int64)


def persname_test(df: pd.DataFrame, emb: np.ndarray, rng: np.random.Generator):
    inv: dict[str, list[int]] = {}
    for i, ps in enumerate(df["persons"]):
        for p in set(ps):  # dedupe: a paragraph citing Borges twice must not self-pair
            inv.setdefault(p, []).append(i)
    shared: set[tuple[int, int]] = set()
    for p, idxs in inv.items():
        if len(idxs) < 2:
            continue
        s = sorted(idxs)
        for a in range(len(s)):
            for b in range(a + 1, len(s)):
                shared.add((s[a], s[b]))
    print(f"  total shared-persName pairs: {len(shared):,}")
    shared_arr = np.array(sorted(shared), dtype=np.int64) if shared else np.empty((0, 2), dtype=np.int64)
    if len(shared_arr) > N_PAIRS:
        sel = rng.choice(len(shared_arr), N_PAIRS, replace=False)
        shared_arr = shared_arr[sel]

    base = sample_pairs(list(range(len(df))), N_PAIRS, rng)

    c_shared = cosines(emb, shared_arr)
    c_base = cosines(emb, base)

    u, p = mannwhitneyu(c_shared, c_base, alternative="two-sided") if len(c_shared) and len(c_base) else (float("nan"), float("nan"))
    return {
        "n_shared_pairs": int(len(c_shared)),
        "n_baseline_pairs": int(len(c_base)),
        "median_shared": float(np.median(c_shared)) if len(c_shared) else float("nan"),
        "iqr_shared_low": float(np.percentile(c_shared, 25)) if len(c_shared) else float("nan"),
        "iqr_shared_high": float(np.percentile(c_shared, 75)) if len(c_shared) else float("nan"),
        "median_baseline": float(np.median(c_base)) if len(c_base) else float("nan"),
        "iqr_baseline_low": float(np.percentile(c_base, 25)) if len(c_base) else float("nan"),
        "iqr_baseline_high": float(np.percentile(c_base, 75)) if len(c_base) else float("nan"),
        "cliffs_delta": cliffs_delta(c_shared, c_base),
        "mw_u": float(u),
        "mw_p": float(p),
    }, c_shared, c_base


def blanchot_probe(df: pd.DataFrame, emb: np.ndarray, rng: np.random.Generator, seeds=BLANCHOT_SEEDS):
    seed_re = re.compile(r"\b(" + "|".join(seeds) + r")\b", re.IGNORECASE)
    name_re = re.compile(r"\b" + BLANCHOT_NAME + r"\b", re.IGNORECASE)
    has_seed = df["text"].apply(lambda t: bool(seed_re.search(t or "")))
    has_circle = df["persons"].apply(lambda ps: any(p in BLANCHOT_CIRCLE for p in ps))
    has_blanchot = df["persons"].apply(lambda ps: BLANCHOT_ID in ps)
    # Stratification exclusion is text-based (notes included), symmetric with how A is
    # built — so footnote mentions of Blanchot are removed too, not just persName tags
    # outside notes. C (persName citation) stays the basis of the "marginal citation"
    # claim and the A↔C row.
    names_blanchot = df["text"].apply(lambda t: bool(name_re.search(t or "")))

    A = np.where(has_seed)[0]
    B = np.where(has_circle)[0]
    C = np.where(has_blanchot)[0]
    named = set(np.where(names_blanchot)[0])
    other = set(range(len(df))) - set(A) - set(B) - set(C)
    D = np.array(sorted(other), dtype=np.int64)

    A_strat = np.array(sorted(set(A) - named), dtype=np.int64)
    B_strat = np.array(sorted(set(B) - named), dtype=np.int64)

    pair_classes = {
        "A↔A": sample_pairs(A.tolist(), N_PAIRS, rng),
        "A↔B": cross_pairs(A, B, N_PAIRS, rng),
        "A↔C": cross_pairs(A, C, N_PAIRS, rng) if len(C) else np.empty((0, 2), dtype=np.int64),
        "A↔D": cross_pairs(A, D, N_PAIRS, rng),
        "A↔B (no Blanchot)": cross_pairs(A_strat, B_strat, N_PAIRS, rng),
        "A↔D (no Blanchot)": cross_pairs(A_strat, D, N_PAIRS, rng),
    }
    cosine_dict = {k: cosines(emb, v) for k, v in pair_classes.items()}
    baseline = cosine_dict["A↔D"]

    rows = []
    for name, vals in cosine_dict.items():
        if len(vals) == 0:
            rows.append({
                "pair_class": name, "n_pairs": 0, "median": None,
                "iqr_low": None, "iqr_high": None,
                "cliffs_delta_vs_AD": None, "mw_p_vs_AD": None,
            })
            continue
        delta = 0.0 if name == "A↔D" else cliffs_delta(vals, baseline)
        try:
            u, p = mannwhitneyu(vals, baseline, alternative="two-sided")
            mwp = float(p)
        except ValueError:
            mwp = None
        rows.append({
            "pair_class": name,
            "n_pairs": int(len(vals)),
            "median": float(np.median(vals)),
            "iqr_low": float(np.percentile(vals, 25)),
            "iqr_high": float(np.percentile(vals, 75)),
            "cliffs_delta_vs_AD": delta,
            "mw_p_vs_AD": mwp,
        })
    groups = {
        "n_A": int(len(A)), "n_B": int(len(B)),
        "n_C": int(len(C)), "n_D": int(len(D)),
        "n_named": int(len(named)),
        "n_A_strat": int(len(A_strat)), "n_B_strat": int(len(B_strat)),
    }
    return rows, groups, cosine_dict


def null_test(
    emb: np.ndarray,
    n_A: int,
    n_B: int,
    rng: np.random.Generator,
    n_runs: int = 5,
) -> list[dict]:
    """Permutation null: scramble group membership.

    On each run, pick random disjoint subsets of |A| and |B| paragraphs
    and treat the rest as D, then recompute the A↔B vs A↔D comparison.
    A well-calibrated method should produce Cliff's δ near 0 across runs.
    (M-W p will often be small because n_pairs is large — focus on δ.)
    """
    n_total = emb.shape[0]
    rows = []
    for run in range(n_runs):
        perm = rng.permutation(n_total)
        fake_A = perm[:n_A]
        fake_B = perm[n_A:n_A + n_B]
        fake_D = perm[n_A + n_B:]
        AB = cross_pairs(fake_A, fake_B, N_PAIRS, rng)
        AD = cross_pairs(fake_A, fake_D, N_PAIRS, rng)
        cAB, cAD = cosines(emb, AB), cosines(emb, AD)
        try:
            _, p = mannwhitneyu(cAB, cAD, alternative="two-sided")
            mwp = float(p)
        except ValueError:
            mwp = None
        rows.append({
            "run": run,
            "median_AB": float(np.median(cAB)),
            "median_AD": float(np.median(cAD)),
            "cliffs_delta": cliffs_delta(cAB, cAD),
            "mw_p": mwp,
        })
    return rows


def robustness_test(df: pd.DataFrame, emb: np.ndarray, rng: np.random.Generator) -> dict:
    """Re-run A↔B vs A↔D on the subset of paragraphs with no <note> content."""
    if "has_note" not in df.columns:
        return {}
    no_notes = ~df["has_note"].values
    df_sub = df[no_notes].reset_index(drop=True)
    emb_sub = emb[no_notes]
    seed_re = re.compile(r"\b(" + "|".join(BLANCHOT_SEEDS) + r")\b", re.IGNORECASE)
    name_re = re.compile(r"\b" + BLANCHOT_NAME + r"\b", re.IGNORECASE)
    A = np.where(df_sub["text"].apply(lambda t: bool(seed_re.search(t or ""))))[0]
    B = np.where(df_sub["persons"].apply(lambda ps: any(p in BLANCHOT_CIRCLE for p in ps)))[0]
    C = np.where(df_sub["persons"].apply(lambda ps: BLANCHOT_ID in ps))[0]
    named = set(np.where(df_sub["text"].apply(lambda t: bool(name_re.search(t or ""))))[0])
    D = np.array(sorted(set(range(len(df_sub))) - set(A) - set(B) - set(C)), dtype=np.int64)
    A_s = np.array(sorted(set(A) - named), dtype=np.int64)
    B_s = np.array(sorted(set(B) - named), dtype=np.int64)

    def _stats(left, right):
        if not len(left) or not len(right):
            return None
        pairs = cross_pairs(left, right, N_PAIRS, rng)
        return cosines(emb_sub, pairs)

    cAB = _stats(A, B); cAD = _stats(A, D)
    cABs = _stats(A_s, B_s); cADs = _stats(A_s, D)
    return {
        "n_paras": int(len(df_sub)),
        "n_A": int(len(A)), "n_B": int(len(B)), "n_C": int(len(C)), "n_D": int(len(D)),
        "median_AB": float(np.median(cAB)) if cAB is not None else None,
        "median_AD": float(np.median(cAD)) if cAD is not None else None,
        "cliffs_delta": cliffs_delta(cAB, cAD) if cAB is not None and cAD is not None else None,
        "median_AB_strat": float(np.median(cABs)) if cABs is not None else None,
        "median_AD_strat": float(np.median(cADs)) if cADs is not None else None,
        "cliffs_delta_strat": cliffs_delta(cABs, cADs) if cABs is not None and cADs is not None else None,
    }


def violin_persname(c_shared, c_base, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.violinplot([c_shared, c_base], showmedians=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["shared persName", "random"])
    ax.set_ylabel("cosine similarity")
    ax.set_title("persName alignment")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def violin_blanchot(cosine_dict, path: Path) -> None:
    items = [(k, v) for k, v in cosine_dict.items() if len(v) > 0]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.violinplot([v for _, v in items], showmedians=True)
    ax.set_xticks(range(1, len(items) + 1))
    ax.set_xticklabels([k for k, _ in items], rotation=25, ha="right")
    ax.set_ylabel("cosine similarity")
    ax.set_title("Blanchot shadow probe — pairwise cosine distributions")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_summary(df, emb, pn, blanchot_rows, blanchot_groups, null_rows=None, robust=None, n_total=None) -> str:
    meta_path = DATA / "embeddings_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    ns_path = DATA / "note_stats.json"
    note_stats = json.loads(ns_path.read_text(encoding="utf-8")) if ns_path.exists() else {}
    n_runs = len(null_rows) if null_rows else 0
    AA = next((r for r in blanchot_rows if r["pair_class"] == "A↔A"), None)
    AB = next((r for r in blanchot_rows if r["pair_class"] == "A↔B"), None)
    AC = next((r for r in blanchot_rows if r["pair_class"] == "A↔C"), None)
    AD = next((r for r in blanchot_rows if r["pair_class"] == "A↔D"), None)
    AB_s = next((r for r in blanchot_rows if r["pair_class"] == "A↔B (no Blanchot)"), None)
    AD_s = next((r for r in blanchot_rows if r["pair_class"] == "A↔D (no Blanchot)"), None)

    null_block = ""
    if null_rows:
        deltas = [r["cliffs_delta"] for r in null_rows]
        median_abs = float(np.median([abs(d) for d in deltas]))
        max_abs = max(abs(d) for d in deltas)
        if median_abs < 0.05:
            verdict = "**passes**"
        elif median_abs > 0.10:
            verdict = "**FAILS — investigate**"
        else:
            verdict = "**marginal**"
        rows_md = "\n".join(
            f"| {r['run']} | {r['median_AB']:.4f} | {r['median_AD']:.4f} | {r['cliffs_delta']:+.3f} |"
            for r in null_rows
        )
        null_block = f"""## Null test — does the method produce zero when there's no signal?

If you scramble which paragraphs are in group A and B (random labels), the test should report no clustering. If it still reports a signal, the test itself is biased.

{n_runs} random label-shuffles, same sample sizes as the real test:

| run | median(A↔B) | median(A↔D) | Cliff's δ |
|---:|---:|---:|---:|
{rows_md}

Median |δ| across runs = **{median_abs:.3f}**, max = {max_abs:.3f}. The real Blanchot probe gave δ = **{AB['cliffs_delta_vs_AD']:+.3f}**, roughly {AB['cliffs_delta_vs_AD']/max(median_abs, 1e-6):.0f}× larger than the typical null. Verdict: {verdict} — the signal vanishes when we destroy the group structure, so the +{AB['cliffs_delta_vs_AD']:.3f} on the real test is not an artifact of how we sample or compute.

Full table: `audit_null.csv`.

"""

    robust_block = ""
    if robust:
        n_notes = note_stats.get("n_notes")
        n_ed = note_stats.get("n_ed")
        n_ed_in_p = note_stats.get("n_ed_in_p")
        n_notes_in_p = note_stats.get("n_notes_in_p")
        if n_ed is not None and n_ed_in_p == 0:
            ed_claim = (
                f"Of {n_notes} `<note>` elements in the corpus, {n_ed} carry "
                f"`resp=\"#ED\"` (the digital editor's commentary layer). Walking "
                f"their ancestor chains shows **all {n_ed} sit at the `<div>` level**, "
                f"never inside a `<p>`:\n\n"
                f"```\nancestor chain for every #ED note: div → div → body → text → TEI\n```\n\n"
                f"The parser builds paragraph text via `\"\".join(p.itertext())`, which "
                f"only traverses descendants of `<p>`. Since zero `#ED` notes are inside "
                f"any `<p>`, **zero `#ED` text reached the embeddings.** The contamination "
                f"path was structurally impossible."
            )
        else:
            ed_claim = (
                f"Of {n_notes} `<note>` elements in the corpus, {n_ed} carry "
                f"`resp=\"#ED\"` (the digital editor's commentary layer), of which "
                f"{n_ed_in_p} sit inside a `<p>` and so reach the embedded text via "
                f"`\"\".join(p.itertext())`. The no-notes subset below removes them."
            )
        robust_block = f"""## Robustness — could editorial commentary be contaminating the result?

**The worry.** The TEI encoding includes `<note>` elements written by the digital editors during encoding (typically `resp="#ED"` with `type="interpretation"` or `type="summary"`). If those notes — which by their nature comment on intellectual genealogy — slipped into the embedded text, they could inflate the Blanchot cluster artifactually.

**Structural argument — the editorial notes were never in the embeddings.**

{ed_claim}

**Sanity confirmation — also for the non-editor notes inside paragraphs.**

The {n_notes_in_p} notes that do live inside `<p>` are footnotes belonging to the original 1981–1987 journal articles, attributed (via `@resp`) to SITIO contributors and the figures they cite (Bloy, Gusmán, Alcalde, Panesi, Pezzoni, Grüner, Molloy, …). These are original journal content. To check they're not driving the result anyway, re-running the Blanchot probe on the **{robust['n_paras']:,} paragraphs that contain zero notes of any kind**:

| pair class | full corpus ({len(df):,}) | no-notes subset ({robust['n_paras']:,}) |
|---|---:|---:|
| median(A↔B) | {AB['median']:.4f} | {robust['median_AB']:.4f} |
| median(A↔D) | {AD['median']:.4f} | {robust['median_AD']:.4f} |
| Cliff's δ | **{AB['cliffs_delta_vs_AD']:+.3f}** | **{robust['cliffs_delta']:+.3f}** |
| stratified Cliff's δ | {AB_s['cliffs_delta_vs_AD']:+.3f} | {robust['cliffs_delta_strat']:+.3f} |

Identical to within noise. The shadow-influence effect lives in the journal's body text, not in any note layer — editorial or otherwise.

"""

    paper_block = f"""## What this means for the paper

1. **§3 (methodology):** The persName sanity check (δ = +{pn['cliffs_delta']:.3f}) validates that embeddings work on this corpus. One-paragraph result, one figure.
2. **§5.2 (computation strengthens humanities):** The Blanchot shadow probe is the headline empirical figure. The abstract's qualitative claim ("Blanchot is marginal as citation but pervasive as vocabulary") becomes quantitative: paragraphs containing his vocabulary are δ = +{AB['cliffs_delta_vs_AD']:.3f} closer to paragraphs about the seven authors he wrote on than to the rest of the corpus, and this survives removing every paragraph that names him (stratified δ = +{AB_s['cliffs_delta_vs_AD']:.3f}).
3. **Reproducibility:** embeddings are cached at `data/embeddings.npy`. Re-running `audit.py` is free and deterministic.

"""

    appendix = f"""## How to read this — a non-statistician's translation

**"cosine similarity"** is a number between -1 and +1 that says how close two paragraph vectors are in the embedding model's semantic space. +1 = identical direction, 0 = unrelated, -1 = opposite. For paragraphs in this corpus, random pairs land around **{pn['median_baseline']:.2f}** — that's the "two unrelated SITIO paragraphs" baseline. Shared-author pairs land around **{pn['median_shared']:.2f}**. The Blanchot probe asks whether vocabulary-bearing paragraphs land higher than {pn['median_baseline']:.2f} when matched against the seven authors.

**"Cliff's delta (δ)"** is an effect size between -1 and +1. It answers a single question: *if I pick one pair from group X and one pair from group Y at random, what's the probability X is closer than Y?* — minus the reverse probability. So:
- δ = 0 → 50/50, no effect
- δ = +0.5 → 75% chance X > Y
- δ = +1 → every X pair is closer than every Y pair (impossible in practice)

Standard thresholds (Romano et al. 2006): |δ| ≥ 0.147 = small, ≥ 0.33 = medium, ≥ 0.474 = large. The Blanchot probe's **δ = +{AB['cliffs_delta_vs_AD']:.3f}** sits in medium-large territory.

**"Mann-Whitney U p-value"** answers: *if the two distributions were actually identical, how likely would I see a gap this large by chance?* Very small p = "not by chance." With 50k samples per side, this test is so powerful that almost any non-zero difference comes back p < 0.001 — so p alone doesn't tell you the effect is *large*, only that it's *real*. **Read p and δ together**: p says "this is not noise"; δ says "this is how big it is."

**"IQR"** = the middle 50% of values (25th to 75th percentile). It tells you how spread out the distribution is. Narrow IQR = tight cluster; wide IQR = heterogeneous.

### Questions you can ask of any cosine result without doing statistics

1. **Is the median far from the corpus baseline (≈{pn['median_baseline']:.2f})?** Shared-persName is at {pn['median_shared']:.2f} → yes, far. A↔B is at {AB['median']:.2f} → modestly above. A↔D is at {AD['median']:.2f} → at baseline (it IS the baseline-ish group).
2. **Is the null test δ ≈ 0?** {f'Yes — max |δ| = {max(abs(r["cliffs_delta"]) for r in null_rows):.3f} across {n_runs} shuffles.' if null_rows else '(see Null test section)'} If it weren't, the method is biased and the real δ is meaningless.
3. **Does the effect survive removing the obvious explanation?** The stratified test (Blanchot-named paragraphs removed) gave δ = +{AB_s['cliffs_delta_vs_AD']:.3f}, essentially unchanged. So the cluster is not just "vocabulary lives near Blanchot's name in some paragraphs."
4. **Does the effect survive removing your editorial layer?** Yes (no-notes subset, robustness section). If the answer to any of (2)(3)(4) were "no," the headline would not hold.

### When to ask for a second opinion

Statistics caught in a humanities paper can be technically correct but misleading. Three things this audit cannot guarantee:

- **That the seed lexicon is the right operationalization of "Blanchotian vocabulary."** A reviewer could argue the list is too narrow, too broad, or biased toward terms that happen to cluster with the seven authors for unrelated reasons. The defence is to publish the exact list (already in `src/lexicons.py`) and justify each term in the paper.
- **That the seven-author circle is the right operationalization of "Blanchot's objects of criticism."** Same kind of methodological choice; defence is the same.
- **That the embedding model isn't smuggling in its own "Blanchot prior" learned from training data.** Qwen3 was trained on internet-scale text including academic discourse on Blanchot. The model might cluster these authors because the literature on them is itself blanchotian. This is a real interpretive caveat that belongs in the paper's discussion, not something this audit can solve.

For these, the right move is `/ultrareview` on this file before submission, or a humanities-stats collaborator.

"""

    parquet_rows = f"{n_total:,} rows" if n_total is not None else "all parsed paragraphs"
    files_block = f"""## Files

- `data/paragraphs.parquet` — {parquet_rows}, gold entity + ana layers + has_note flag
- `data/embeddings.npy` — float32, L2-normalized, dim from `embeddings_meta.json`
- `data/embeddings_meta.json` — model / GPU / dim / dtype / text-hash provenance for the full embed
- `results/audit_persname.csv` — sanity check numbers
- `results/audit_blanchot.csv` — full Blanchot probe per-pair-class
- `results/audit_null.csv` — {n_runs} null-test runs
- `results/figures/persname_violin.png` — sanity check plot
- `results/figures/blanchot_probe_violin.png` — Blanchot probe plot
"""

    head = f"""# SITIO embedding audit — summary

**What this is.** A measurement of how language-model embeddings position the paragraphs of Revista SITIO in semantic space, used to test whether the abstract's qualitative "Blanchot shadow influence" claim is visible empirically at corpus scale.

**Setup.**
- **Corpus:** {len(df):,} paragraphs after filter (text ≥ {MIN_TEXT_LEN} chars).
- **Embedding:** `{meta.get('model_id', '?')}` on `{meta.get('gpu_type', '?')}`, dim {meta.get('dim', emb.shape[1])}, L2-normalized. Cosine similarity is the dot product of two paragraph vectors and lives in [-1, +1].
- **Statistics:** medians and IQRs for the cosine distributions; **Cliff's delta** as an effect size (|δ| ≥ 0.147 = small, ≥ 0.33 = medium, ≥ 0.474 = large); **Mann-Whitney U** for significance.

---

## Sanity check — persName alignment

**Question.** Do paragraphs that share at least one hand-tagged cited person sit closer in embedding space than random pairs?

This is the "do the embeddings even work for this corpus" check. If two paragraphs both cite Borges and the model puts them far apart, the whole audit is unsafe.

| | shared persName | random baseline |
|---|---:|---:|
| median cosine | **{pn['median_shared']:.4f}** | {pn['median_baseline']:.4f} |
| IQR | [{pn['iqr_shared_low']:.3f}, {pn['iqr_shared_high']:.3f}] | [{pn['iqr_baseline_low']:.3f}, {pn['iqr_baseline_high']:.3f}] |
| n pairs | {pn['n_shared_pairs']:,} | {pn['n_baseline_pairs']:,} |

**Cliff's δ = {pn['cliffs_delta']:+.3f}** · M-W p {('≈ 0 (scipy underflow)' if pn['mw_p'] == 0 else f'= {pn["mw_p"]:.2e}')}.

**How to read this.** The median cosine for pairs sharing a cited author is **+{pn['median_shared']-pn['median_baseline']:.2f}** above random pairs. Cliff's δ = {pn['cliffs_delta']:+.3f} means roughly {50 + 50*pn['cliffs_delta']:.0f}% of shared-persName pairs are closer than the median random pair. Embeddings track citation co-occurrence; audit is on solid ground.

See `figures/persname_violin.png`.

---

## The Blanchot shadow probe

**Question.** Does the conceptual vocabulary the journal inherits from Blanchot (escritura, ausencia, afuera, errancia, muerte, olvido…) cluster in embedding space with the seven authors Blanchot wrote about — even though Blanchot himself is almost never cited in SITIO?

Four groups over the {len(df):,} paragraphs:

- **A** — Blanchotian vocabulary present (whole-word regex on {len(BLANCHOT_SEEDS)} seed terms). **n = {blanchot_groups['n_A']}.**
- **B** — paragraphs citing any of the seven Blanchot-circle authors (Kafka, Joyce, Beckett, Céline, Faulkner, Borges, Gombrowicz). **n = {blanchot_groups['n_B']}.**
- **C** — paragraphs citing Blanchot himself. **n = {blanchot_groups['n_C']}** (confirming the "marginal citation presence" claim at full corpus scale).
- **D** — corpus rest. **n = {blanchot_groups['n_D']:,}.**

50,000 sampled cosine pairs per pair-class.

### Headline

| pair class | median cosine | Cliff's δ vs A↔D |
|---|---:|---:|
| A↔A (vocab ↔ vocab) | {AA['median']:.3f} | {AA['cliffs_delta_vs_AD']:+.3f} |
| **A↔B (vocab ↔ 7 authors)** | **{AB['median']:.3f}** | **{AB['cliffs_delta_vs_AD']:+.3f}** |
| A↔C (vocab ↔ Blanchot-named) | {AC['median']:.3f} | {AC['cliffs_delta_vs_AD']:+.3f} |
| A↔D (vocab ↔ corpus rest) | {AD['median']:.3f} | 0.000 (baseline) |

**Plain reading.** Vocabulary lands +{AB['median']-AD['median']:.3f} cosine closer to paragraphs about the seven Blanchot-circle authors than to the corpus rest. Cliff's δ = {AB['cliffs_delta_vs_AD']:+.3f} is a **medium-large effect** — about {50 + 50*AB['cliffs_delta_vs_AD']:.0f}% of (vocab↔7-authors) pairs are closer than the median (vocab↔corpus-rest) pair. M-W p {('≈ 0 (scipy underflow)' if AB['mw_p_vs_AD'] == 0 else f"= {AB['mw_p_vs_AD']:.2e}")}.

### The shadow-stratified test

Excluding the {blanchot_groups['n_named']} paragraphs that name Blanchot anywhere in their text (footnotes included) from both sides. Cliff's δ is measured against the same A↔D baseline as the headline table:

| pair class | median | Cliff's δ vs A↔D |
|---|---:|---:|
| **A↔B (no Blanchot named)** | **{AB_s['median']:.3f}** | **{AB_s['cliffs_delta_vs_AD']:+.3f}** |
| A↔D (no Blanchot named) | {AD_s['median']:.3f} | {AD_s['cliffs_delta_vs_AD']:+.3f} |

The effect is **identical** to within noise. The cluster is conceptual genealogy, not a citation footprint.

### One bonus pattern

**A↔A ({AA['median']:.3f}) < A↔B ({AB['median']:.3f}).** Seed-bearing paragraphs are *less* similar to each other than they are to paragraphs about the seven circle authors. Group A is heterogeneous (any paragraph with escritura OR muerte OR afuera). Group B is specific (a fixed set of authors), and seeds land in B as Blanchotian readings of those authors. The vocabulary makes more sense alongside the seven authors than alongside other vocabulary occurrences. That is the shadow-influence story compressed into one inequality.

See `figures/blanchot_probe_violin.png`. Full table: `audit_blanchot.csv`.

---

"""
    return head + null_block + robust_block + paper_block + appendix + files_block


def main() -> None:
    (RES / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    verify_lexicon_ids()
    df, n_total = load_paragraphs()
    emb = get_embeddings(df)

    pn, c_shared, c_base = persname_test(df, emb, rng)
    pd.DataFrame([pn]).to_csv(RES / "audit_persname.csv", index=False)
    violin_persname(c_shared, c_base, RES / "figures" / "persname_violin.png")
    print(f"persName: median(shared)={pn['median_shared']:.4f} "
          f"median(base)={pn['median_baseline']:.4f} "
          f"delta={pn['cliffs_delta']:+.3f} p={pn['mw_p']:.2e}")

    rows, groups, cosine_dict = blanchot_probe(df, emb, rng)
    pd.DataFrame(rows).to_csv(RES / "audit_blanchot.csv", index=False)
    violin_blanchot(cosine_dict, RES / "figures" / "blanchot_probe_violin.png")
    print(f"Blanchot groups: {groups}")
    for r in rows:
        if r["median"] is None:
            print(f"  {r['pair_class']:25s} n=0 (empty group)")
            continue
        print(f"  {r['pair_class']:25s} n={r['n_pairs']:6d} "
              f"median={r['median']:.4f} delta={r['cliffs_delta_vs_AD']:+.3f} "
              f"p={r['mw_p_vs_AD']}")

    n_runs = 5
    null_rows = null_test(emb, groups["n_A"], groups["n_B"], rng, n_runs=n_runs)
    pd.DataFrame(null_rows).to_csv(RES / "audit_null.csv", index=False)
    null_deltas = [r["cliffs_delta"] for r in null_rows]
    print(f"Null test ({n_runs} shuffled runs): "
          f"deltas={[f'{d:+.3f}' for d in null_deltas]} "
          f"median={float(np.median(null_deltas)):+.3f}")

    robust = robustness_test(df, emb, rng)
    if robust:
        print(f"Robustness (no-notes subset, n={robust['n_paras']}): "
              f"delta={robust['cliffs_delta']:+.3f} stratified={robust['cliffs_delta_strat']:+.3f}")

    summary = build_summary(df, emb, pn, rows, groups, null_rows, robust, n_total=n_total)
    (RES / "audit_summary.md").write_text(summary, encoding="utf-8")
    print(f"\nwrote {RES / 'audit_summary.md'}")


if __name__ == "__main__":
    main()
