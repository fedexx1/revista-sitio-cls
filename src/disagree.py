"""T1 — layer-disagreement hotspot map (E vs C vs H). $0 — cached artifacts only.

Where do the three descriptions of the corpus DISAGREE? Two hotspot rankings
(E-high/surface-zero; C-high/E-low), per-paragraph neighborhood disagreement, and
global E<->C / E<->H Spearman. See the spec (T1) and results/disagreement_report.md.
"""
from __future__ import annotations
import sys, re, argparse
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
import audit

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"

TOP_K_PAIRS = 200       # per direction, persisted to CSV
N_EXHIBIT = 10          # per direction, excerpted in the report
K_NEIGH = 10            # neighborhood size
MIN_SHARED_LEMMAS = 3   # direction-2 hard filter (spec)
EXCERPT_CHARS = 280


def load_all():
    """(df, lay, emb) positionally aligned — same contract as triangulate.load_all
    (duplicated 6-liner; house style avoids cross-module analysis imports)."""
    df, _ = audit.load_paragraphs()
    lay = pd.read_parquet(DATA / "concept_layer.parquet")
    assert (df["para_id"].fillna("X").values == lay["para_id"].fillna("X").values).all(), \
        "positional alignment broken — never join bulk corpus on para_id"
    emb = audit.get_embeddings(df)
    assert emb.shape[0] == len(df)
    return df, lay, emb


def _binary_csr(list_col, n_rows: int) -> sparse.csr_matrix:
    """Rows of token-lists -> deduped binary CSR (n_rows x vocab)."""
    vocab: dict[str, int] = {}
    indptr, indices = [0], []
    for row in list_col:
        toks = set() if row is None else {str(t) for t in row}
        for tok in sorted(toks):
            indices.append(vocab.setdefault(tok, len(vocab)))
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (np.ones(len(indices), dtype=np.int32), indices, indptr),
        shape=(n_rows, len(vocab)))


def similarity_matrices(df, lay, emb):
    """Dense pairwise matrices. n=3476 -> each (n,n) float32/int32 is ~48 MB; fine."""
    E = emb.astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    S_e = E @ E.T
    Xc = _binary_csr(lay["lemmas"].values, len(df))
    inter_c = np.asarray((Xc @ Xc.T).todense(), dtype=np.int32)
    sz_c = np.asarray(Xc.sum(axis=1)).ravel().astype(np.int32)
    union_c = sz_c[:, None] + sz_c[None, :] - inter_c
    with np.errstate(divide="ignore", invalid="ignore"):
        S_c = np.where(union_c > 0, inter_c / union_c, 0.0).astype(np.float32)
    Xh = _binary_csr(df["persons"].values, len(df))
    inter_h = np.asarray((Xh @ Xh.T).todense(), dtype=np.int32)
    return S_e, S_c, inter_c, inter_h, sz_c


def _text_codes(df) -> np.ndarray:
    """text_id -> integer codes (text_id is never None after the parse extension)."""
    codes = pd.factorize(df["text_id"])[0].astype(np.int64)
    assert (codes >= 0).all(), "text_id must be non-null after the parse extension"
    return codes


def eligible_pairs(df, sz_c):
    """Upper-triangle (i,j): different text, non-adjacent rows, both lemma sets non-empty."""
    n = len(df)
    ii, jj = np.triu_indices(n, k=1)
    div = _text_codes(df)
    ok = (div[ii] != div[jj]) & (np.abs(ii - jj) > 1) & (sz_c[ii] > 0) & (sz_c[jj] > 0)
    return ii[ok], jj[ok]


def _text_titles() -> dict[str, str]:
    """text_id -> TEI <head> title (best-effort; empty string when the div has no head).
    Read-only TEI access via parse; keys mirror parse._text_identity_div's synthesis."""
    import parse
    out: dict[str, str] = {}
    for issue in parse.ISSUES:
        root = parse.get_root(parse.TEI_DIR / f"{issue}.xml")
        for d in root.iter(f"{{{parse.NS['tei']}}}div"):
            xid = d.get(parse.XML_ID)
            key = xid if xid else f"{issue}_pos{int(d.sourceline or 0)}"
            heads = d.xpath("./tei:head", namespaces=parse.NS)
            t = re.sub(r"\s+", " ", "".join(heads[0].itertext())).strip() if heads else ""
            if key not in out or (t and not out[key]):
                out[key] = t
    return out


def _pair_rows(df, idx_pairs, S_e, S_c, inter_c, inter_h, direction, titles):
    rows = []
    for i, j in idx_pairs:
        tid_a, tid_b = df["text_id"].values[i], df["text_id"].values[j]
        rows.append({
            "direction": direction,
            "row_a": int(i), "row_b": int(j),
            "issue_a": df["issue"].values[i], "issue_b": df["issue"].values[j],
            "text_a": titles.get(tid_a, "") or tid_a,
            "text_b": titles.get(tid_b, "") or tid_b,
            "contrib_a": df["contributor"].values[i],
            "contrib_b": df["contributor"].values[j],
            "e_cos": round(float(S_e[i, j]), 4),
            "c_jaccard": round(float(S_c[i, j]), 4),
            "shared_lemmas": int(inter_c[i, j]),
            "h_shared": int(inter_h[i, j]),
        })
    return rows


def hotspots(df, S_e, S_c, inter_c, inter_h, ii, jj, titles) -> pd.DataFrame:
    """Two pre-registered directions; deterministic order via lexsort tie-breaks."""
    e = S_e[ii, jj]; c = S_c[ii, jj]; ic = inter_c[ii, jj]; ih = inter_h[ii, jj]

    m1 = (c == 0) & (ih == 0)                       # direction 1: E-high / surface-zero
    o1 = np.lexsort((jj[m1], ii[m1], -e[m1]))[:TOP_K_PAIRS]
    d1 = list(zip(ii[m1][o1], jj[m1][o1]))

    m2 = ic >= MIN_SHARED_LEMMAS                    # direction 2: C-high / E-low
    o2 = np.lexsort((jj[m2], ii[m2], e[m2]))[:TOP_K_PAIRS]
    d2 = list(zip(ii[m2][o2], jj[m2][o2]))

    rows = _pair_rows(df, d1, S_e, S_c, inter_c, inter_h, "E_high_surface_zero", titles) \
         + _pair_rows(df, d2, S_e, S_c, inter_c, inter_h, "C_high_E_low", titles)
    print(f"direction 1 pool: {int(m1.sum()):,} pairs; direction 2 pool: {int(m2.sum()):,} pairs")
    return pd.DataFrame(rows)


def scale_stats(S_e, S_c, inter_c, inter_h, ii, jj) -> dict:
    """Distribution-level view of the two directions (the 'graded line')."""
    e = S_e[ii, jj]; c = S_c[ii, jj]; ic = inter_c[ii, jj]; ih = inter_h[ii, jj]
    m1 = (c == 0) & (ih == 0)
    m2 = ic >= MIN_SHARED_LEMMAS
    e1, e2 = e[m1].astype(np.float64), e[m2].astype(np.float64)
    return {
        "n_eligible": int(len(e)), "mean_all": round(float(e.mean()), 3),
        "d1_pool": int(m1.sum()), "d1_share": round(float(m1.mean()), 3),
        "d1_mean": round(float(e1.mean()), 3), "d1_sd": round(float(e1.std()), 3),
        "d1_p999": round(float(np.percentile(e1, 99.9)), 3),
        "d1_p9999": round(float(np.percentile(e1, 99.99)), 3),
        "d1_gt05": int((e1 > 0.5).sum()), "d1_gt06": int((e1 > 0.6).sum()),
        "d1_gt07": int((e1 > 0.7).sum()),
        "d2_pool": int(m2.sum()), "d2_share": round(float(m2.mean()), 4),
        "d2_mean": round(float(e2.mean()), 3), "d2_sd": round(float(e2.std()), 3),
        "d2_lt025": int((e2 < 0.25).sum()), "d2_lt03": int((e2 < 0.3).sum()),
    }


def neighborhood_disagreement(df, S_e, S_c, sz_c) -> pd.DataFrame:
    """Per paragraph: Jaccard between its top-K E-neighbors and top-K C-neighbors,
    restricted to eligible partners (different div, partner lemma set non-empty)."""
    n = len(df)
    div = _text_codes(df)
    rows = []
    order = np.arange(n)
    for i in range(n):
        if sz_c[i] == 0:
            continue
        mask = (div != div[i]) & (sz_c > 0)
        mask[i] = False
        cand = order[mask]
        if len(cand) < K_NEIGH:
            continue
        eN = cand[np.lexsort((cand, -S_e[i, cand]))[:K_NEIGH]]
        cN = cand[np.lexsort((cand, -S_c[i, cand]))[:K_NEIGH]]
        a, b = set(eN.tolist()), set(cN.tolist())
        rows.append({"row": i, "issue": df["issue"].values[i],
                     "div_id": df["div_id"].values[i],
                     "neigh_jaccard": round(len(a & b) / len(a | b), 4)})
    return pd.DataFrame(rows).sort_values(["neigh_jaccard", "row"]).reset_index(drop=True)


def summary_stats(S_e, S_c, inter_h, ii, jj) -> dict:
    e = S_e[ii, jj].astype(np.float64)
    rho_c, _ = spearmanr(e, S_c[ii, jj].astype(np.float64))
    rho_h, _ = spearmanr(e, (inter_h[ii, jj] > 0).astype(np.float64))
    return {"n_eligible_pairs": int(len(ii)),
            "spearman_E_C": round(float(rho_c), 4),
            "spearman_E_Hbinary": round(float(rho_h), 4)}


def _excerpt(text: str) -> str:
    t = " ".join((text or "").split())
    return t[:EXCERPT_CHARS] + ("…" if len(t) > EXCERPT_CHARS else "")


def append_report(df, hot: pd.DataFrame, neigh: pd.DataFrame, stats: dict,
                  scale: dict) -> None:
    lines = ["", "---", "", "## Results (run appended)", "",
             f"- Eligible pairs: **{stats['n_eligible_pairs']:,}**",
             f"- Spearman ρ(E, C) = **{stats['spearman_E_C']}**; "
             f"ρ(E, H-binary) = **{stats['spearman_E_Hbinary']}**", "",
             "### Scale interpretation (the graded line)", "",
             f"- **Direction 1 pool: {scale['d1_pool']:,} pairs** — "
             f"{scale['d1_share']:.0%} of all eligible pairs. Zero surface overlap is the "
             f"corpus's NORMAL condition. E-similarity over this pool: mean "
             f"{scale['d1_mean']}, sd {scale['d1_sd']}; 99.9th pct {scale['d1_p999']}, "
             f"99.99th {scale['d1_p9999']}; pairs above 0.5 / 0.6 / 0.7: "
             f"**{scale['d1_gt05']:,} / {scale['d1_gt06']} / {scale['d1_gt07']}**. The "
             f"top-10 exhibits are roughly one-in-a-million events on a smooth continuum — "
             f"the top-10 cutoff is rhetorical, not natural.",
             f"- **Direction 2 pool: {scale['d2_pool']:,} pairs** ({scale['d2_share']:.2%} "
             f"of eligible) — sharing ≥{MIN_SHARED_LEMMAS} lemmas is rare, and it nearly "
             f"guarantees model kinship (mean e {scale['d2_mean']} vs corpus "
             f"{scale['mean_all']}). The refused-overlap tail barely exists: "
             f"**{scale['d2_lt025']}** pairs below e=0.25, {scale['d2_lt03']} below 0.30 — "
             f"the top-200 CSV covers the entire phenomenon.",
             f"- **The asymmetry (a methods finding):** concept overlap is SUFFICIENT for "
             f"the model's similarity (refusals ≈ {scale['d2_lt03']/scale['d2_pool']:.1%}) "
             f"but NOT NECESSARY — surface-zero pairs average {scale['d1_mean']}, virtually "
             f"the corpus mean {scale['mean_all']}. The embedding space is built "
             f"overwhelmingly out of something other than shared concept vocabulary.",
             "",
             "Distribution figures: `figures/disagree_dist.png` (per-direction histograms "
             "with the top-10 exhibits marked).", ""]
    for direction, title in [("E_high_surface_zero", "Direction 1 — E-high / surface-zero"),
                             ("C_high_E_low", "Direction 2 — C-high / E-low")]:
        sub = hot[hot["direction"] == direction].head(N_EXHIBIT)
        lines += [f"### {title} — top {len(sub)} exhibit", ""]
        for k, r in enumerate(sub.itertuples(), 1):
            lines += [f"**{k}. “{r.text_a}” ({r.contrib_a}) ↔ “{r.text_b}” ({r.contrib_b})** "
                      f"— rows {r.row_a}↔{r.row_b}, {r.issue_a}↔{r.issue_b}, "
                      f"e_cos={r.e_cos}, c_jac={r.c_jaccard}, shared_lemmas={r.shared_lemmas}, "
                      f"h_shared={r.h_shared}",
                      f"> A: {_excerpt(df['text'].values[r.row_a])}",
                      ">", f"> B: {_excerpt(df['text'].values[r.row_b])}", ""]
    top15 = neigh.head(15)
    md_table = ["| row | issue | div_id | neigh_jaccard |", "|---|---|---|---|"] + [
        f"| {r.row} | {r.issue} | {r.div_id} | {r.neigh_jaccard} |"
        for r in top15.itertuples()]
    lines += ["### Most C/E-misplaced paragraphs (lowest neighborhood agreement)", "",
              "\n".join(md_table), "",
              "_Researcher annotations of these exhibits go below; pursued hypotheses",
              "become their own pre-registered tests._", ""]
    with (RES / "disagreement_report.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description="T1 disagreement map")
    ap.add_argument("--smoke", action="store_true", help="first 400 rows, no writes")
    args = ap.parse_args(argv)
    df, lay, emb = load_all()
    if args.smoke:
        df, lay, emb = df.head(400).copy(), lay.head(400).copy(), emb[:400]
    S_e, S_c, inter_c, inter_h, sz_c = similarity_matrices(df, lay, emb)
    assert abs(float(np.diag(S_e).mean()) - 1.0) < 1e-3
    ii, jj = eligible_pairs(df, sz_c)
    titles = {} if args.smoke else _text_titles()
    hot = hotspots(df, S_e, S_c, inter_c, inter_h, ii, jj, titles)
    neigh = neighborhood_disagreement(df, S_e, S_c, sz_c)
    stats = summary_stats(S_e, S_c, inter_h, ii, jj)
    scale = scale_stats(S_e, S_c, inter_c, inter_h, ii, jj)
    print(stats)
    print(scale)
    if args.smoke:
        print("SMOKE OK —", len(hot), "hotspot rows,", len(neigh), "paragraphs")
        return
    hot.to_csv(RES / "disagreement_pairs.csv", index=False, encoding="utf-8-sig")
    neigh.to_csv(RES / "disagreement_paragraphs.csv", index=False, encoding="utf-8-sig")
    append_report(df, hot, neigh, stats, scale)
    print("wrote disagreement_pairs.csv, disagreement_paragraphs.csv, report appended")


if __name__ == "__main__":
    main()
