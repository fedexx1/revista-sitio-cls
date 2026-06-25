"""SP3 — triangulated methods study: ana_* recovery (E vs C-open vs C-closed) + concept-labeled clusters.

Pure local compute on cached embeddings + committed §5.1 outputs ($0).
See docs/superpowers/specs/2026-06-03-sp3-triangulated-recovery-design.md
"""
from __future__ import annotations
import sys, math, argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import audit

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"

K = 16
N_PERM = 1000
KM_SEED = 0
PERM_SEED = 0

_NLP = None


def load_all():
    """Return (df, lay, emb) positionally aligned (3,476 rows; emb cached → 0 Modal)."""
    df, _ = audit.load_paragraphs()
    lay = pd.read_parquet(DATA / "concept_layer.parquet")
    assert (df["para_id"].fillna("X").values == lay["para_id"].fillna("X").values).all(), \
        "positional alignment broken — do NOT join bulk corpus on para_id"
    emb = audit.get_embeddings(df)
    assert emb.shape[0] == len(df)
    return df, lay, emb


def _iter(arr):
    return [] if arr is None else [x for x in arr]


def gold_rows(df):
    """Row indices of the 39 ana_* gold paragraphs (non-empty ana_gold)."""
    return [i for i in range(len(df)) if len(_iter(df["ana_gold"].values[i])) > 0]


def gold_subset(df):
    """List of 39 dicts joining each gold paragraph (df row) to its §5.1 outputs via para_id.

    keys: row (df index), para_id, gold_ids (set), open (str), closed (str)
    """
    cc = pd.read_csv(RES / "concept_comparison.csv", encoding="utf-8-sig").set_index("para_id")
    assert cc.index.is_unique, "concept_comparison.csv has duplicate para_ids — check for appended re-runs"
    out = []
    for i in gold_rows(df):
        pid = df["para_id"].values[i]
        r = cc.loc[pid]
        out.append({
            "row": i, "para_id": pid,
            "gold_ids": set(str(g).strip() for g in _iter(df["ana_gold"].values[i])),
            "open": str(r["sonnet_open"]),
            "closed": str(r["sonnet_closed"]),
        })
    return out


def _jac(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _open_lemmas(open_str: str) -> set:
    """Lemmatized-token set of an open-extraction string (spaCy es); for C-open Jaccard."""
    global _NLP
    if _NLP is None:
        import concept_layer as cl
        _NLP = cl._nlp()
    toks = set()
    for t in _NLP(open_str.replace(";", " ")):
        if t.is_alpha and not t.is_stop and len(t.lemma_) > 2:
            toks.add(t.lemma_.lower())
    return toks


def check_f1_anchor() -> dict:
    """Recompute C-closed micro P/R/F1 from concept_comparison.csv — must reproduce §5.1."""
    cc = pd.read_csv(RES / "concept_comparison.csv", encoding="utf-8-sig")
    tp = fp = fn = 0
    for _, r in cc.iterrows():
        gold = set(g.strip() for g in str(r["gold_ids"]).split(";") if g.strip())
        pred = set(x.strip() for x in str(r["sonnet_closed"]).split(";") if x.strip())
        tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
    p = tp / (tp + fp); rec = tp / (tp + fn); f1 = 2 * p * rec / (p + rec)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


def build_pairs(df, emb, gold) -> pd.DataFrame:
    """All C(39,2)=741 gold pairs with shared_gold flag + per-layer similarity."""
    n = len(gold)
    E = emb[[g["row"] for g in gold]].astype(np.float64)   # fancy-index returns a copy; does not mutate emb
    E /= np.linalg.norm(E, axis=1, keepdims=True)              # L2-normalize → dot = cosine
    open_sets = [_open_lemmas(g["open"]) for g in gold]
    closed_sets = [set(x.strip() for x in g["closed"].split(";") if x.strip()) for g in gold]
    gold_sets = [g["gold_ids"] for g in gold]
    rows = []
    for a, b in combinations(range(n), 2):
        rows.append({
            "pid_a": gold[a]["para_id"], "pid_b": gold[b]["para_id"],
            "shared_gold": 1 if (gold_sets[a] & gold_sets[b]) else 0,
            "e_cos": round(float(E[a] @ E[b]), 4),
            "copen_jac": round(_jac(open_sets[a], open_sets[b]), 4),
            "cclosed_jac": round(_jac(closed_sets[a], closed_sets[b]), 4),
        })
    return pd.DataFrame(rows)


def recovery_delta(pairs, gold, n_perm=N_PERM, seed=PERM_SEED) -> pd.DataFrame:
    """Per-layer Cliff's δ (sharing-pair sim vs non-sharing-pair sim) + label-permutation null.

    Positive δ ⇒ that layer places same-gold-label paragraphs closer ⇒ recovers the human grouping.
    """
    layers = {"E": "e_cos", "C-open": "copen_jac", "C-closed": "cclosed_jac"}
    shared = pairs["shared_gold"].values.astype(bool)
    sims = {col: pairs[col].values for col in layers.values()}
    obs = {name: audit.cliffs_delta(sims[col][shared], sims[col][~shared])
           for name, col in layers.items()}

    n = len(gold)
    gold_sets = [g["gold_ids"] for g in gold]
    idx_pairs = list(combinations(range(n), 2))
    rng = np.random.default_rng(seed)
    null = {name: [] for name in layers}
    for _ in range(n_perm):
        perm = rng.permutation(n)
        ps = [gold_sets[perm[k]] for k in range(n)]
        sh = np.array([1 if (ps[a] & ps[b]) else 0 for a, b in idx_pairs], dtype=bool)
        if sh.all() or (~sh).all():            # degenerate split → δ undefined, treat as 0
            for name in layers:
                null[name].append(0.0)
            continue
        for name, col in layers.items():
            null[name].append(audit.cliffs_delta(sims[col][sh], sims[col][~sh]))

    out = []
    for name in layers:
        arr = np.asarray(null[name])
        mu, sd = float(arr.mean()), float(arr.std(ddof=1))
        z = (obs[name] - mu) / sd if sd > 0 else float("nan")
        p = float((np.abs(arr) >= abs(obs[name])).mean())               # two-sided empirical (vs null measured from 0)
        out.append({"layer": name, "delta": round(obs[name], 4),
                    "n_pos": int(shared.sum()), "n_neg": int((~shared).sum()),
                    "null_mean": round(mu, 4), "null_sd": round(sd, 4),
                    "z": round(z, 3), "p": round(p, 4)})
    return pd.DataFrame(out)[["layer", "delta", "n_pos", "n_neg", "null_mean", "null_sd", "z", "p"]]


def cluster_and_label(df, lay, emb, k=K, seed=KM_SEED):
    """k-means over all embeddings; label clusters by dominant + distinctive C lemmas; overlay gold.

    Returns (clusters_df, labels) where labels[i] is the cluster of df row i.
    """
    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(emb)
    N = len(df)
    lemma_rows = [set(_iter(lay["lemmas"].values[i])) for i in range(N)]
    global_df = Counter()
    for s in lemma_rows:
        for l in s:
            global_df[l] += 1
    gset = set(gold_rows(df))

    rows = []
    for c in range(k):
        members = [i for i in range(N) if labels[i] == c]
        size = len(members)
        cnt = Counter()
        for i in members:
            for l in lemma_rows[i]:
                cnt[l] += 1
        top = [l for l, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:10]]
        dist = []
        for l, a in cnt.items():
            if a < 3:
                continue
            b = size - a; c2 = global_df[l] - a; d = N - size - c2  # c2 = (not-in-cluster ∧ has-lemma); c2 avoids shadowing loop var c
            lo = math.log(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c2 + 0.5)))
            dist.append((lo, l))
        dist = [l for _, l in sorted(dist, key=lambda kv: (-kv[0], kv[1]))[:10]]
        glab = Counter()
        for i in members:
            if i in gset:
                for tlab in _iter(df["ana_gold"].values[i]):
                    glab[tlab] += 1
        gold_labels = "; ".join(f"{tlab}({m})" for tlab, m
                                in sorted(glab.items(), key=lambda kv: (-kv[1], kv[0])))
        rows.append({"cluster_id": c, "size": size,
                     "top_lemmas": "; ".join(top),
                     "distinctive_lemmas": "; ".join(dist),
                     "n_gold": sum(1 for i in members if i in gset),
                     "gold_labels": gold_labels})
    clusters = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
    return clusters, labels


def main(argv=None):
    ap = argparse.ArgumentParser(description="SP3 triangulated recovery")
    ap.add_argument("--smoke", action="store_true", help="quick run (small n_perm), no CSV writes")
    args = ap.parse_args(argv)

    df, lay, emb = load_all()
    gold = gold_subset(df)
    anchor = check_f1_anchor()
    print("f1 anchor", anchor)
    assert anchor["precision"] == 0.242 and anchor["recall"] == 0.873 and anchor["f1"] == 0.379, \
        "§5.1 anchor FAILED — gold↔§5.1 join is wrong"

    pairs = build_pairs(df, emb, gold)
    rec = recovery_delta(pairs, gold, n_perm=100 if args.smoke else N_PERM)
    print(rec.to_string(index=False))
    clusters, _ = cluster_and_label(df, lay, emb)

    if args.smoke:
        print("SMOKE OK — pairs", len(pairs), "clusters", len(clusters))
        return
    RES.mkdir(exist_ok=True)
    rec.to_csv(RES / "sp3_recovery.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(RES / "sp3_pairs.csv", index=False, encoding="utf-8-sig")
    clusters.to_csv(RES / "sp3_clusters.csv", index=False, encoding="utf-8-sig")
    print("wrote 3 CSVs to", RES)


if __name__ == "__main__":
    main()
