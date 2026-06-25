"""Run the concept extractor on the 39 ana-tagged Entredichos paragraphs and
build the §5.1 comparison.

For each paragraph × {gemini, sonnet}:
  - open extraction   -> surface/lexical concept terms
  - closed-set        -> which of the 31 editor ana_* labels the model picks,
                         scored (precision/recall/F1) against the gold ana_* set

Outputs (committed to results/):
  - concept_comparison.csv   per-paragraph gold vs. both models, both modes
  - concept_metrics.csv      micro-averaged closed-set P/R/F1 per model
Provenance -> data/concept_provenance.json.

All LLM calls go through concept_extract's content-hash cache, so re-running is
free and deterministic. First run makes 39 × 2 × 2 = 156 cached API calls.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import concept_extract as ce

DATA = ce.DATA
RES = ce.ROOT / "results"


def _prf(pred: list[str], gold: list[str]) -> tuple[int, int, int, float, float, float]:
    p, g = set(pred), set(gold)
    tp = len(p & g)
    fp = len(p - g)
    fn = len(g - p)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return tp, fp, fn, prec, rec, f1


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    inv = ce.load_inventory()

    df = pd.read_parquet(DATA / "paragraphs.parquet")
    gold_df = df[df["ana_gold"].str.len() > 0].reset_index(drop=True)
    print(f"ana-tagged paragraphs: {len(gold_df)}")

    rows = []
    micro = {m: {"tp": 0, "fp": 0, "fn": 0} for m in ce.MODELS}
    for _, r in gold_df.iterrows():
        text = r["text"]
        gold = list(r["ana_gold"])
        row = {
            "para_id": r["para_id"],
            "gold_ids": "; ".join(gold),
            "gold_labels": "; ".join(inv.get(g, g) for g in gold),
        }
        for mk in ce.MODELS:
            open_c = ce.extract_open(text, mk)
            closed = ce.classify_closed(text, mk, inv)
            tp, fp, fn, prec, rec, f1 = _prf(closed, gold)
            micro[mk]["tp"] += tp
            micro[mk]["fp"] += fp
            micro[mk]["fn"] += fn
            row[f"{mk}_open"] = "; ".join(c["term"] for c in open_c)
            row[f"{mk}_closed"] = "; ".join(closed)
            row[f"{mk}_n_closed"] = len(closed)
            row[f"{mk}_precision"] = round(prec, 3)
            row[f"{mk}_recall"] = round(rec, 3)
            row[f"{mk}_f1"] = round(f1, 3)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "concept_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"wrote {RES / 'concept_comparison.csv'} ({len(out)} rows)")

    # Micro-averaged closed-set metrics + a couple of descriptive open-set figures.
    metric_rows = []
    for mk in ce.MODELS:
        tp, fp, fn = micro[mk]["tp"], micro[mk]["fp"], micro[mk]["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        mean_pred = out[f"{mk}_n_closed"].mean()
        mean_open = out[f"{mk}_open"].apply(lambda s: len(s.split("; ")) if s else 0).mean()
        metric_rows.append({
            "model": mk,
            "model_id": ce.MODELS[mk],
            "closed_tp": tp, "closed_fp": fp, "closed_fn": fn,
            "closed_micro_precision": round(prec, 3),
            "closed_micro_recall": round(rec, 3),
            "closed_micro_f1": round(f1, 3),
            "mean_labels_predicted": round(mean_pred, 2),
            "mean_gold_labels": round(out["gold_ids"].apply(lambda s: len(s.split("; "))).mean(), 2),
            "mean_open_concepts": round(mean_open, 2),
        })
    met = pd.DataFrame(metric_rows)
    met.to_csv(RES / "concept_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"wrote {RES / 'concept_metrics.csv'}")
    print(met.to_string(index=False))

    (DATA / "concept_provenance.json").write_text(json.dumps({
        "models": ce.MODELS,
        "temperature": ce.TEMPERATURE,
        "n_paragraphs": int(len(gold_df)),
        "n_inventory_concepts": len(inv),
        "dspy_version": version("dspy"),
        "litellm_version": version("litellm"),
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Closed-set predictions scored vs gold ana_* per paragraph; "
                "outputs frozen by data/concept_cache content-hash cache.",
    }, indent=2), encoding="utf-8")
    print(f"wrote {DATA / 'concept_provenance.json'}")


if __name__ == "__main__":
    main()
