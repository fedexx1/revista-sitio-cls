"""SP2 figures — read the committed cartography CSVs, write PNGs. No recompute."""
from __future__ import annotations
import sys, json
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
import audit

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; RES = ROOT / "results"; FIG = RES / "figures"
ISSUE_LABELS = ["issue_1", "issue_2", "issue_3", "issue_4-5", "issue_6"]


def _heatmap(piv, title, path):
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * piv.shape[1] + 3), max(4, 0.4 * piv.shape[0] + 2)))
    vmax = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_title(title, fontsize=10); fig.colorbar(im, ax=ax, label="log-odds")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_blanchot_tieback():
    am = pd.read_csv(RES / "concept_author.csv", encoding="utf-8-sig")
    circle = list(audit.BLANCHOT_CIRCLE)
    seeds = json.loads((DATA / "curated_seeds.json").read_text(encoding="utf-8"))
    lex = seeds["core_lemmas"] + seeds["shared_lemmas"]   # validated lexicon (matches the anchor)
    sub = am[am["person"].isin(circle) & am["lemma"].isin(lex)]
    piv = sub.pivot_table(index="lemma", columns="person", values="log_odds")
    _heatmap(piv, "Blanchot circle x validated lexicon (core+shared), log-odds — SP2 tie-back",
             FIG / "cartography_blanchot_tieback.png")


def fig_top_author_assoc(top=30):
    am = pd.read_csv(RES / "concept_author.csv", encoding="utf-8-sig")
    sig = am[am["significant"]].nlargest(top, "log_odds")
    sig = sig.assign(cell=sig["lemma"] + " | " + sig["person"]).sort_values("log_odds")
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(sig) + 1)))
    ax.barh(sig["cell"], sig["log_odds"], color="#3b6")
    ax.set_xlabel("log-odds"); ax.set_title(f"Top {top} FDR-significant concept x author associations", fontsize=10)
    ax.tick_params(axis="y", labelsize=7); fig.tight_layout()
    fig.savefig(FIG / "cartography_top_author_assoc.png", dpi=150); plt.close(fig)


def fig_time_trajectories(top=8):
    tr = pd.read_csv(RES / "concept_time_trend.csv", encoding="utf-8-sig")
    prop_cols = [c for c in tr.columns if c.startswith("prop_")]
    sig = tr.sort_values("q_trend")
    rising = sig[sig["direction"] == "rising"].head(top // 2)
    falling = sig[sig["direction"] == "falling"].head(top // 2)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(prop_cols))
    for _, r in pd.concat([rising, falling]).iterrows():
        ax.plot(x, [r[c] for c in prop_cols], marker="o",
                label=f"{r['lemma']} ({r['direction']})")
    ax.set_xticks(list(x)); ax.set_xticklabels(ISSUE_LABELS, rotation=45, ha="right")
    ax.set_ylabel("document-frequency proportion")
    ax.set_title("Most-significantly-trending lemmas, 1981->1987", fontsize=10)
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(FIG / "cartography_time_trajectories.png", dpi=150); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_blanchot_tieback(); fig_top_author_assoc(); fig_time_trajectories()
    print("wrote 3 figures to", FIG)


if __name__ == "__main__":
    main()
