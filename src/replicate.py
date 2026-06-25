"""T2 — cross-model replication: persName sanity + Blanchot probe (4 seed sets) +
SP3 E-recovery, re-run on bge-m3 embeddings. Qwen3 columns double as join anchors.

(The Stage-3 "ceiling" seed set is intentionally excluded: it is an over-fit upper
bound, not a claim, so it is not a replication target.)
"""
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
import audit, triangulate
import lexicon_audit
from lexicons import BLANCHOT_SEEDS

DATA = audit.DATA
RES = audit.RES
ANCHOR_TOL = 0.02


def probe_suite(df, emb) -> dict:
    """persName δ + four-seed-set probe table + permutation null, on one matrix."""
    rng = np.random.default_rng(audit.SEED)
    pn, _, _ = audit.persname_test(df, emb, rng)
    cur = json.loads((DATA / "curated_seeds.json").read_text(encoding="utf-8"))
    sets = {
        "baseline (BLANCHOT_SEEDS)": list(BLANCHOT_SEEDS),
        "core — discovered only": cur["core_discovered"],
        "core — discovered + manual": cur["core"],
        "core + shared": sorted(set(cur["core"]) | set(cur["shared"])),
    }
    probes = {name: lexicon_audit._probe(df, emb, seeds) for name, seeds in sets.items()}
    _, groups, _ = audit.blanchot_probe(df, emb, np.random.default_rng(audit.SEED),
                                        seeds=BLANCHOT_SEEDS)
    rngn = np.random.default_rng(audit.SEED + 1)
    null = audit.null_test(emb, groups["n_A"], groups["n_B"], rngn)
    return {"persname_delta": pn["cliffs_delta"], "probes": probes,
            "null_deltas": [r["cliffs_delta"] for r in null]}


def main():
    audit.verify_lexicon_ids()
    df, _ = audit.load_paragraphs()

    emb_q = audit.get_embeddings(df)                  # cached qwen3
    q = probe_suite(df, emb_q)
    base = q["probes"]["baseline (BLANCHOT_SEEDS)"]
    assert abs(base["delta"] - 0.427) < ANCHOR_TOL, f"qwen3 anchor broke: {base['delta']}"
    assert abs(q["persname_delta"] - 0.852) < ANCHOR_TOL, \
        f"persName anchor broke: {q['persname_delta']}"

    emb_b = audit.get_embeddings(df, model="bge-m3")  # cached after Task 6
    b = probe_suite(df, emb_b)

    gold = triangulate.gold_subset(df)
    rec_q = triangulate.recovery_delta(triangulate.build_pairs(df, emb_q, gold), gold)
    rec_b = triangulate.recovery_delta(triangulate.build_pairs(df, emb_b, gold), gold)
    # C rows do not depend on the embedding — they must be identical (join anchor)
    for layer, want in [("C-open", 0.290), ("C-closed", 0.385)]:
        got_q = float(rec_q[rec_q["layer"] == layer]["delta"].iloc[0])
        got_b = float(rec_b[rec_b["layer"] == layer]["delta"].iloc[0])
        assert abs(got_q - want) < 1e-3 and abs(got_b - want) < 1e-3, \
            f"SP3 C anchor broke: {layer} {got_q} {got_b}"

    rows = []
    for model, r in [("qwen3", q), ("bge-m3", b)]:
        rows.append({"model": model, "test": "persName sanity",
                     "delta": round(r["persname_delta"], 3), "delta_strat": None,
                     "n_A": None, "null_band": None})
        for name, p in r["probes"].items():
            rows.append({"model": model, "test": f"probe: {name}",
                         "delta": round(p["delta"], 3),
                         "delta_strat": round(p["delta_strat"], 3),
                         "n_A": p["n_A"], "null_band": None})
        nd = np.asarray(r["null_deltas"])
        rows.append({"model": model, "test": "permutation null (5 runs)",
                     "delta": round(float(nd.mean()), 3), "delta_strat": None,
                     "n_A": None, "null_band": f"[{nd.min():+.3f}, {nd.max():+.3f}]"})
    for model, rec in [("qwen3", rec_q), ("bge-m3", rec_b)]:
        e = rec[rec["layer"] == "E"].iloc[0]
        rows.append({"model": model, "test": "SP3 E-recovery (741 pairs)",
                     "delta": float(e["delta"]), "delta_strat": None,
                     "n_A": None, "null_band": f"z={e['z']}, p={e['p']}"})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "replication.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))

    md = ["", "---", "", "## Results (run appended)", "",
          "| model | test | δ | stratified δ | n_A | null band |", "|---|---|---|---|---|---|"]
    def _cell(v):
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else v
    for r in out.itertuples():
        md.append(f"| {r.model} | {r.test} | {r.delta} | {_cell(r.delta_strat)} | "
                  f"{_cell(r.n_A)} | {_cell(r.null_band)} |")
    with (RES / "replication_report.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("wrote replication.csv; report appended")


if __name__ == "__main__":
    main()
