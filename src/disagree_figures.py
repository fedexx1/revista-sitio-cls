"""T1 figure — hexbin of E cosine vs C Jaccard over eligible pairs."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import disagree

df, lay, emb = disagree.load_all()
S_e, S_c, inter_c, inter_h, sz_c = disagree.similarity_matrices(df, lay, emb)
ii, jj = disagree.eligible_pairs(df, sz_c)
fig, ax = plt.subplots(figsize=(6, 5))
hb = ax.hexbin(S_e[ii, jj], S_c[ii, jj], gridsize=60, bins="log", cmap="viridis")
ax.set_xlabel("E cosine (Qwen3)")
ax.set_ylabel("C Jaccard (lemmas)")
ax.set_title("Layer agreement over eligible pairs")
fig.colorbar(hb, label="log10(count)")
out = disagree.RES / "figures" / "disagree_hexbin.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)

# --- per-direction distributions with the top-10 exhibits marked ---
import pandas as pd
e = S_e[ii, jj]; c = S_c[ii, jj]
ic = inter_c[ii, jj]; ih = inter_h[ii, jj]
hot = pd.read_csv(disagree.RES / "disagreement_pairs.csv", encoding="utf-8-sig")
fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))

e1 = e[(c == 0) & (ih == 0)]
a1.hist(e1, bins=120, log=True, color="#4a6fa5")
for v in hot[hot["direction"] == "E_high_surface_zero"].head(10)["e_cos"]:
    a1.axvline(v, color="crimson", lw=0.8, alpha=0.8)
a1.set_title(f"Direction 1 — E cosine over surface-zero pairs (n={len(e1):,})")
a1.set_xlabel("E cosine"); a1.set_ylabel("pairs (log)")

e2 = e[ic >= disagree.MIN_SHARED_LEMMAS]
a2.hist(e2, bins=60, log=True, color="#4a6fa5")
for v in hot[hot["direction"] == "C_high_E_low"].head(10)["e_cos"]:
    a2.axvline(v, color="crimson", lw=0.8, alpha=0.8)
a2.set_title(f"Direction 2 — E cosine over ≥{disagree.MIN_SHARED_LEMMAS}-shared-lemma pairs (n={len(e2):,})")
a2.set_xlabel("E cosine"); a2.set_ylabel("pairs (log)")

out2 = disagree.RES / "figures" / "disagree_dist.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print("wrote", out2)
