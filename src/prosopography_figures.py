"""SP2 prosopography figures — distributions + concept×period/region heatmaps. No recompute."""
from __future__ import annotations
import sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import parse

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"; FIG = RES / "figures"
PERIOD_ORDER = [name for _, _, name in parse.HISTORICAL_PERIODS] + ["Unknown"]


def _dist_bar(dim, order, path, title):
    pr = pd.read_csv(RES / "prosopography_profile.csv", encoding="utf-8-sig")
    sub = pr[pr["dim"] == dim].set_index("bucket")
    cats = [b for b in order if b in sub.index] if order else list(sub.index)
    cats += [b for b in sub.index if b not in cats]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(cats))
    ax.bar(x, [sub.loc[b, "n_figures"] for b in cats], color="#3498db")
    ax.set_xticks(list(x)); ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("distinct cited figures"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _heatmap(csv, dim, col_order, path, title, top=25):
    d = pd.read_csv(RES / csv, encoding="utf-8-sig")
    sig = d[d["significant"]]
    lemmas = sig.nlargest(top, "log_odds")["lemma"].unique()
    sub = d[d["lemma"].isin(lemmas)]
    piv = sub.pivot_table(index="lemma", columns=dim, values="log_odds")
    cols = [c for c in (col_order or list(piv.columns)) if c in piv.columns]
    cols += [c for c in piv.columns if c not in cols]
    piv = piv[cols]
    vmax = np.nanmax(np.abs(piv.values)) if piv.size else 1.0
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * piv.shape[1] + 3), max(4, 0.35 * piv.shape[0] + 2)))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_title(title, fontsize=10); fig.colorbar(im, ax=ax, label="log-odds")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    _dist_bar("period", PERIOD_ORDER, FIG / "proso_period_dist.png", "Cited figures by period")
    _dist_bar("region", None, FIG / "proso_region_dist.png", "Cited figures by region")
    _heatmap("concept_period.csv", "period", PERIOD_ORDER, FIG / "proso_concept_period.png",
             "Top concept×period associations (log-odds)")
    _heatmap("concept_region.csv", "region", None, FIG / "proso_concept_region.png",
             "Top concept×region associations (log-odds)")
    print("wrote 4 figures to", FIG)


if __name__ == "__main__":
    main()
