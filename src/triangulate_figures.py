"""SP3 figures — reuse triangulate's functions; write δ bar chart + PCA/UMAP cluster scatters."""
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
import triangulate as t

RES = t.RES
FIG = RES / "figures"


def fig_recovery_bar():
    rec = pd.read_csv(RES / "sp3_recovery.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(rec))
    colors = ["#d62728" if p < 0.05 else "#999" for p in rec["p"]]
    ax.bar(x, rec["delta"], color=colors)
    ax.errorbar(x, rec["null_mean"], yerr=rec["null_sd"], fmt="o", color="k",
                capsize=4, label="permutation null (mean±sd)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(list(x)); ax.set_xticklabels(rec["layer"])
    ax.set_ylabel("Cliff's δ (same-label vs different-label pair similarity)")
    ax.set_title("ana_* recovery by layer (red = p<0.05)", fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG / "sp3_recovery_delta.png", dpi=150); plt.close(fig)


def _scatter(xy, labels, gold_mask, title, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xy[~gold_mask, 0], xy[~gold_mask, 1], c=labels[~gold_mask],
               cmap="tab20", s=5, alpha=0.5, linewidths=0)
    ax.scatter(xy[gold_mask, 0], xy[gold_mask, 1], c="black", s=40, marker="*",
               edgecolors="white", linewidths=0.5, label="ana_* gold (39)")
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_recovery_bar()
    df, lay, emb = t.load_all()
    _, labels = t.cluster_and_label(df, lay, emb)
    gold_mask = np.zeros(len(df), dtype=bool)
    gold_mask[t.gold_rows(df)] = True

    from sklearn.decomposition import PCA
    pca_xy = PCA(n_components=2, random_state=0).fit_transform(emb)
    _scatter(pca_xy, labels, gold_mask, "Embedding clusters (PCA) — gold overlaid",
             FIG / "sp3_clusters_pca.png")

    import umap
    # UMAP may warn "Spectral initialisation failed! Falling back to random initialisation!"
    # on this near-degenerate eigenspectrum — harmless; random_state=42 keeps the fallback deterministic.
    umap_xy = umap.UMAP(n_components=2, random_state=42).fit_transform(emb)
    _scatter(umap_xy, labels, gold_mask, "Embedding clusters (UMAP) — gold overlaid",
             FIG / "sp3_clusters_umap.png")
    print("wrote 3 figures to", FIG)


if __name__ == "__main__":
    main()
