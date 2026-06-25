"""SP2 (cont.) — concept × prosopography: concept×period + concept×region maps over C×H.

Reuses cartography's log-odds + Fisher + BH-FDR machinery. No embeddings ($0).
See docs/superpowers/specs/2026-06-03-sp2-prosopography-design.md
"""
from __future__ import annotations
import sys, argparse
from collections import Counter, defaultdict
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import audit
import parse
import cartography as cg

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"
PERIOD_ORDER = [name for _, _, name in parse.HISTORICAL_PERIODS] + ["Unknown"]


def load():
    """Return (df, lay, idx, proso, cited_ids). idx from cartography (lemma_rows>=10, N)."""
    df, _ = audit.load_paragraphs()
    lay = pd.read_parquet(DATA / "concept_layer.parquet")
    assert (df["para_id"].fillna("X").values == lay["para_id"].fillna("X").values).all(), \
        "positional alignment broken"
    idx = cg.build_indexes(df, lay)
    proso = parse.parse_prosopography(parse.TEI_DIR / "listPerson.xml",
                                      parse.TEI_DIR / "listPlaces2.xml").set_index("id")
    cited = set(p for ps in df["persons"].values if ps is not None for p in ps)
    assert all(c in proso.index for c in cited), "some cited person ids missing from prosopography"
    return df, lay, idx, proso, cited


def _bucket_rows(df, proso, dim):
    """row idx -> set, keyed by demographic bucket value of cited figures (paragraphs overlap buckets)."""
    dim_of = proso[dim].to_dict()
    bucket_rows = defaultdict(set)
    for i, ps in enumerate(df["persons"].values):
        if ps is None:
            continue
        for pid in set(ps):
            b = dim_of.get(pid)
            if b is not None:
                bucket_rows[b].add(i)
    return bucket_rows


def build_concept_bucket_map(df, lay, idx, proso, dim) -> pd.DataFrame:
    """concept×<dim> enrichment: per (lemma, bucket) 2x2 vs whole corpus, log-odds+Fisher+BH-FDR."""
    N = idx["N"]
    lemma_rows = idx["lemma_rows"]
    n_lemma = {l: len(r) for l, r in lemma_rows.items()}
    bucket_rows = _bucket_rows(df, proso, dim)
    rows = []
    for bucket, brows in bucket_rows.items():
        n_b = len(brows)
        cnt = Counter()
        for i in brows:
            for lem in set(cg._row_iter(lay["lemmas"].values[i])):
                if lem in lemma_rows:
                    cnt[lem] += 1
        for lem, a in cnt.items():
            if a < cg.MIN_SUPPORT:
                continue
            b = n_b - a
            c = n_lemma[lem] - a
            d = max(N - n_b - c, 0)  # buckets overlap (a paragraph can cite figures of several origins); clamp
            log_odds, p = cg.cell_stats(a, b, c, d)
            rows.append({"lemma": lem, dim: bucket, "a": a, "n_bucket_paras": n_b,
                         "n_lemma_paras": n_lemma[lem], "log_odds": round(log_odds, 3), "p": p})
    cg.add_fdr(rows)
    if not rows:
        return pd.DataFrame(columns=["lemma", dim, "a", "n_bucket_paras", "n_lemma_paras",
                                     "log_odds", "p", "q", "significant"])
    out = pd.DataFrame(rows)
    out["significant"] = out["q"] < 0.05
    out["p"] = out["p"].round(6); out["q"] = out["q"].round(6)
    return out[["lemma", dim, "a", "n_bucket_paras", "n_lemma_paras", "log_odds", "p", "q",
                "significant"]].sort_values(["q", "log_odds", "lemma", dim],
                                            ascending=[True, False, True, True]).reset_index(drop=True)


def build_profile(df, proso, cited) -> pd.DataFrame:
    """Descriptive cited-figure profile: distinct figures + paragraphs by period, region, origin."""
    rows = []
    for dim in ("period", "region"):
        bucket_rows = _bucket_rows(df, proso, dim)
        fig_by = defaultdict(set)
        for pid in cited:
            fig_by[proso.loc[pid, dim]].add(pid)
        for bucket in sorted(bucket_rows, key=lambda b: (PERIOD_ORDER.index(b) if dim == "period" and b in PERIOD_ORDER else 99, str(b))):
            rows.append({"dim": dim, "bucket": bucket,
                         "n_figures": len(fig_by.get(bucket, set())),
                         "n_paras": len(bucket_rows[bucket])})
    unknown = {pid for pid in cited if proso.loc[pid, "country"] is None}
    arg = {pid for pid in cited if proso.loc[pid, "country"] == "Argentina"}
    foreign = cited - arg - unknown
    for label, ids in (("Argentine", arg), ("Foreign", foreign), ("Unknown origin", unknown)):
        paras = set(i for i, ps in enumerate(df["persons"].values)
                    if ps is not None and (set(ps) & ids))
        rows.append({"dim": "origin", "bucket": label, "n_figures": len(ids), "n_paras": len(paras)})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="SP2 prosopography")
    ap.add_argument("--smoke", action="store_true", help="quick sanity, no writes")
    args = ap.parse_args(argv)
    df, lay, idx, proso, cited = load()
    print(f"persons {len(proso)} | cited {len(cited)} | lemmas {len(idx['lemma_rows'])}")
    period = build_concept_bucket_map(df, lay, idx, proso, "period")
    region = build_concept_bucket_map(df, lay, idx, proso, "region")
    profile = build_profile(df, proso, cited)
    if args.smoke:
        print("SMOKE OK — period", len(period), "region", len(region), "profile", len(profile))
        return
    DATA.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
    proso.reset_index().to_csv(DATA / "persons_prosopography.csv", index=False, encoding="utf-8-sig")
    period.to_csv(RES / "concept_period.csv", index=False, encoding="utf-8-sig")
    region.to_csv(RES / "concept_region.csv", index=False, encoding="utf-8-sig")
    profile.to_csv(RES / "prosopography_profile.csv", index=False, encoding="utf-8-sig")
    print("wrote persons_prosopography.csv + 3 result CSVs")


if __name__ == "__main__":
    main()
