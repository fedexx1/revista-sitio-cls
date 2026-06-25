"""SP2 — corpus cartography (concept x author + concept x time).

C x H maps over the cached concept layer and TEI persName/issue apparatus.
No embeddings: robust to the embedding prior by design. Pure local compute ($0).
See docs/superpowers/specs/2026-06-03-sp2-corpus-cartography-design.md
"""
from __future__ import annotations
import sys, math, json
from collections import Counter
from pathlib import Path

# Windows cp1252 stdout crashes on non-ASCII; force UTF-8 (handoff landmine).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, false_discovery_control, norm
import audit

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"

LEMMA_MIN_PARAS = 10
AUTHOR_MIN_PARAS = 5
MIN_SUPPORT = 3
ISSUE_ORDER = ["issue_1", "issue_2", "issue_3", "issue_4-5", "issue_6"]


def load_layers():
    """Return (df, lay) positionally aligned; assert row-equality."""
    df, _ = audit.load_paragraphs()
    lay = pd.read_parquet(DATA / "concept_layer.parquet")
    assert len(df) == len(lay), f"row count mismatch {len(df)} != {len(lay)}"
    assert (df["para_id"].fillna("X").values == lay["para_id"].fillna("X").values).all(), \
        "positional alignment broken — do NOT merge on para_id"
    return df, lay


def _row_iter(arr):
    """Return the string items of a numpy-object / list cell (possibly empty)."""
    if arr is None:
        return []
    return [x for x in arr]


def build_indexes(df, lay):
    """Per-lemma and per-person paragraph-index sets, restricted to thresholds.

    Returns dict with:
      lemma_rows:  {lemma: set(row_idx)}  for lemmas in >= LEMMA_MIN_PARAS paras
      person_rows: {person: set(row_idx)} for persons in >= AUTHOR_MIN_PARAS paras
      N: total paragraphs
    """
    N = len(df)
    lemma_rows: dict[str, set] = {}
    for i, cell in enumerate(lay["lemmas"].values):
        for lem in set(_row_iter(cell)):
            lemma_rows.setdefault(lem, set()).add(i)
    person_rows: dict[str, set] = {}
    for i, cell in enumerate(df["persons"].values):
        for p in set(_row_iter(cell)):
            person_rows.setdefault(p, set()).add(i)
    lemma_rows = {l: r for l, r in lemma_rows.items() if len(r) >= LEMMA_MIN_PARAS}
    person_rows = {p: r for p, r in person_rows.items() if len(r) >= AUTHOR_MIN_PARAS}
    return {"lemma_rows": lemma_rows, "person_rows": person_rows, "N": N}


def cell_stats(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Haldane-corrected log-odds ratio and two-sided Fisher exact p for a 2x2 cell.
        a = group & lemma,  b = group & not-lemma,
        c = not-group & lemma,  d = not-group & not-lemma.
    """
    log_odds = math.log(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))
    _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return log_odds, p


def add_fdr(rows: list[dict], pkey: str = "p", qkey: str = "q") -> list[dict]:
    """Annotate rows in place with Benjamini-Hochberg q-values across ALL rows."""
    if not rows:
        return rows
    q = false_discovery_control([r[pkey] for r in rows], method="bh")
    for r, qi in zip(rows, q):
        r[qkey] = float(qi)
    return rows


def build_author_map(df, lay, idx) -> pd.DataFrame:
    """One row per tested (lemma, person) cell with a >= MIN_SUPPORT. Background = all N paras."""
    N = idx["N"]
    lemma_rows, person_rows = idx["lemma_rows"], idx["person_rows"]
    n_lemma = {l: len(r) for l, r in lemma_rows.items()}
    rows = []
    for person, prows in person_rows.items():
        n_p = len(prows)
        # count thresholded lemmas occurring within this person's paragraphs
        cnt = Counter()
        for i in prows:
            for lem in set(_row_iter(lay["lemmas"].values[i])):
                if lem in lemma_rows:
                    cnt[lem] += 1
        for lem, a in cnt.items():
            if a < MIN_SUPPORT:
                continue
            b = n_p - a
            c = n_lemma[lem] - a
            d = N - n_p - c
            log_odds, p = cell_stats(a, b, c, d)
            rows.append({"lemma": lem, "person": person, "a": a,
                         "n_person_paras": n_p, "n_lemma_paras": n_lemma[lem],
                         "log_odds": round(log_odds, 3), "p": p})
    add_fdr(rows)
    if not rows:
        return pd.DataFrame(columns=["lemma", "person", "a", "n_person_paras",
                                     "n_lemma_paras", "log_odds", "p", "q", "significant"])
    out = pd.DataFrame(rows)
    out["significant"] = out["q"] < 0.05
    out = out.sort_values(["q", "log_odds", "lemma", "person"],
                          ascending=[True, False, True, True]).reset_index(drop=True)
    out["p"] = out["p"].round(6); out["q"] = out["q"].round(6)
    return out[["lemma", "person", "a", "n_person_paras", "n_lemma_paras",
                "log_odds", "p", "q", "significant"]]


def check_anchor(author_map: pd.DataFrame) -> dict:
    """Blanchot circle x SP1 core+shared lemmas should be positive (regression anchor)."""
    circle = set(audit.BLANCHOT_CIRCLE)
    seeds = json.loads((DATA / "curated_seeds.json").read_text(encoding="utf-8"))
    core = set(seeds["core_lemmas"]) | set(seeds["shared_lemmas"])  # full validated lexicon
    sub = author_map[author_map["person"].isin(circle) & author_map["lemma"].isin(core)]
    n_cells = len(sub)
    n_pos = int((sub["log_odds"] > 0).sum())
    n_sig = int(sub["significant"].sum())
    frac_pos = (n_pos / n_cells) if n_cells else 0.0
    ok = n_cells >= 3 and frac_pos >= 0.6 and n_sig >= 1
    return {"n_cells": n_cells, "n_pos": n_pos, "n_sig": n_sig,
            "frac_pos": round(frac_pos, 3), "ok": ok}


def trend_test(a_by_issue, n_by_issue, scores):
    """Cochran-Armitage trend z-stat and two-sided p for one lemma across ordered groups.
    a_by_issue: successes per issue; n_by_issue: group sizes; scores: ordinal 1..k.
    Returns (z, p, direction) where direction is 'rising'/'falling'/'flat'.
    """
    a = np.asarray(a_by_issue, float); n = np.asarray(n_by_issue, float); t = np.asarray(scores, float)
    N = n.sum(); A = a.sum()
    if N == 0 or A == 0 or A == N:
        return 0.0, 1.0, "flat"
    pbar = A / N
    tbar = (n * t).sum() / N
    U = (a * (t - tbar)).sum()
    V = pbar * (1 - pbar) * (n * (t - tbar) ** 2).sum()
    if V <= 0:
        return 0.0, 1.0, "flat"
    z = U / math.sqrt(V)
    p = 2 * (1 - norm.cdf(abs(z)))
    direction = "rising" if z > 0 else ("falling" if z < 0 else "flat")
    return z, p, direction


def build_time_trend(df, lay, idx) -> pd.DataFrame:
    lemma_rows = idx["lemma_rows"]
    issues = ISSUE_ORDER
    scores = list(range(1, len(issues) + 1))
    issue_of = df["issue"].values
    n_by_issue = [int((issue_of == iss).sum()) for iss in issues]
    rows = []
    for lem, lrows in lemma_rows.items():
        a_by = []
        for iss in issues:
            a_by.append(sum(1 for i in lrows if issue_of[i] == iss))
        z, p, direction = trend_test(a_by, n_by_issue, scores)
        rec = {"lemma": lem, "n_lemma_paras": len(lrows)}
        for iss, ai, ni in zip(issues, a_by, n_by_issue):
            key = "prop_" + iss.replace("-", "_")
            rec[key] = round(ai / ni, 4) if ni else 0.0
        rec.update({"direction": direction, "ca_stat": round(z, 3), "p": p})
        rows.append(rec)
    add_fdr(rows, qkey="q_trend")
    out = pd.DataFrame(rows)
    out["p"] = out["p"].round(6); out["q_trend"] = out["q_trend"].round(6)
    prop_cols = [c for c in out.columns if c.startswith("prop_")]
    return out[["lemma", "n_lemma_paras", *prop_cols, "direction", "ca_stat", "p", "q_trend"]] \
        .sort_values(["q_trend", "ca_stat", "lemma"], ascending=[True, False, True]).reset_index(drop=True)


def build_time_enrichment(df, lay, idx) -> pd.DataFrame:
    N = idx["N"]
    lemma_rows = idx["lemma_rows"]
    n_lemma = {l: len(r) for l, r in lemma_rows.items()}
    issue_of = df["issue"].values
    rows = []
    for iss in ISSUE_ORDER:
        irows = {i for i in range(N) if issue_of[i] == iss}
        n_i = len(irows)
        cnt = Counter()
        for i in irows:
            for lem in set(_row_iter(lay["lemmas"].values[i])):
                if lem in lemma_rows:
                    cnt[lem] += 1
        for lem, a in cnt.items():
            if a < MIN_SUPPORT:
                continue
            b = n_i - a
            c = n_lemma[lem] - a
            d = N - n_i - c
            log_odds, p = cell_stats(a, b, c, d)
            rows.append({"lemma": lem, "issue": iss, "a": a,
                         "n_issue_paras": n_i, "n_lemma_paras": n_lemma[lem],
                         "log_odds": round(log_odds, 3), "p": p})
    add_fdr(rows)
    if not rows:
        return pd.DataFrame(columns=["lemma", "issue", "a", "n_issue_paras",
                                     "n_lemma_paras", "log_odds", "p", "q", "significant"])
    out = pd.DataFrame(rows)
    out["significant"] = out["q"] < 0.05
    out["p"] = out["p"].round(6); out["q"] = out["q"].round(6)
    return out[["lemma", "issue", "a", "n_issue_paras", "n_lemma_paras",
                "log_odds", "p", "q", "significant"]] \
        .sort_values(["q", "log_odds", "lemma", "issue"],
                     ascending=[True, False, True, True]).reset_index(drop=True)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="SP2 corpus cartography")
    ap.add_argument("--smoke", action="store_true", help="quick subset sanity, no CSV writes")
    args = ap.parse_args(argv)
    df, lay = load_layers()
    idx = build_indexes(df, lay)
    print(f"rows {idx['N']} | lemmas {len(idx['lemma_rows'])} | persons {len(idx['person_rows'])}")

    author = build_author_map(df, lay, idx)
    anchor = check_anchor(author)
    print("anchor", anchor)
    assert anchor["ok"], "Blanchot circle x core anchor FAILED — harness wrong"
    trend = build_time_trend(df, lay, idx)
    enrich = build_time_enrichment(df, lay, idx)

    if args.smoke:
        print("SMOKE OK — author", len(author), "trend", len(trend), "enrich", len(enrich))
        return
    RES.mkdir(exist_ok=True)
    author.to_csv(RES / "concept_author.csv", index=False, encoding="utf-8-sig")
    trend.to_csv(RES / "concept_time_trend.csv", index=False, encoding="utf-8-sig")
    enrich.to_csv(RES / "concept_time_enrichment.csv", index=False, encoding="utf-8-sig")
    print("wrote 3 CSVs to", RES)


if __name__ == "__main__":
    main()
