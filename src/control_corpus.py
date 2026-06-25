"""T3 — Orbis Tertius control-corpus probe. Same model (qwen3), same probe code path
(audit.blanchot_probe, unchanged), different corpus. See the spec (T3) and
results/control_corpus_report.md (bands fixed pre-data)."""
from __future__ import annotations
import sys, re, json, argparse, hashlib
from datetime import datetime, timezone
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import audit
from lexicons import BLANCHOT_SEEDS, BLANCHOT_CIRCLE, BLANCHOT_ID, BLANCHOT_NAME

ORBIS = Path(__file__).resolve().parents[3] / "orbis-kg" / "data" / "raw" / "articles_orbis"
DATA = audit.DATA
RES = audit.RES
CAP = 8000              # max paragraphs to embed (A∪B∪C kept whole; D sampled)
LONG_BLOCK = 1200       # chars; blocks longer than this get sentence re-split
CHUNK_TARGET = 700      # chars; greedy accumulation target for re-split
EMBED_BATCH = 2000      # texts per Modal call

SURNAME_FOR_ID = {      # whole-word, case-insensitive; same ids as listPerson
    "kafka": r"kafka", "james_joyce": r"joyce", "samuel_beckett": r"beckett",
    "louis_ferdinand_celine": r"c[eé]line", "william_faulkner": r"faulkner",
    "borges": r"borges", "witold_gombrowicz": r"gombrowicz",
    BLANCHOT_ID: BLANCHOT_NAME,
}
_RES_FOR_ID = {pid: re.compile(rf"\b{pat}\b", re.IGNORECASE)
               for pid, pat in SURNAME_FOR_ID.items()}


def _segment_text(raw: str) -> list[str]:
    """Blank-line blocks; long blocks re-split by sentence accumulation (~CHUNK_TARGET).
    The Orbis txts hard-wrap lines and rarely blank-line between body paragraphs."""
    blocks, cur = [], []
    for line in raw.splitlines():
        if line.strip():
            cur.append(line.strip())
        elif cur:
            blocks.append(" ".join(cur)); cur = []
    if cur:
        blocks.append(" ".join(cur))
    out = []
    for b in blocks:
        if len(b) <= LONG_BLOCK:
            out.append(b); continue
        acc = ""
        for sent in re.split(r"(?<=[.!?…])\s+", b):
            acc = (acc + " " + sent).strip()
            if len(acc) >= CHUNK_TARGET:
                out.append(acc); acc = ""
        if acc:
            out.append(acc)
    return [p for p in out if len(p) >= audit.MIN_TEXT_LEN]


def build_corpus() -> pd.DataFrame:
    """Audit-compatible df: text, persons (regex ids), div_id, issue (year dir), has_note."""
    rows = []
    files = sorted(ORBIS.glob("*/*.txt"))
    assert len(files) >= 400, f"expected ~473 Orbis txts, found {len(files)}"
    for f in files:
        year_dir = f.parent.name
        for k, para in enumerate(_segment_text(f.read_text(encoding="utf-8", errors="replace"))):
            persons = [pid for pid, rx in _RES_FOR_ID.items() if rx.search(para)]
            rows.append({"para_id": None, "issue": year_dir,
                         "div_id": f.stem, "div_type": "article", "div_subtype": "",
                         "text": para, "persons": persons, "places": [],
                         "ana_gold": [], "has_note": False, "src_block": k})
    df = pd.DataFrame(rows)
    print(f"orbis paragraphs: {len(df)} from {len(files)} files; "
          f"len median={int(df['text'].str.len().median())}")
    return df


def define_groups(df) -> dict:
    """Same group logic as audit.blanchot_probe (duplicated regexes so the sampled df's
    groups can be asserted against the probe's own recomputation)."""
    seed_re = re.compile(r"\b(" + "|".join(BLANCHOT_SEEDS) + r")\b", re.IGNORECASE)
    has_seed = df["text"].apply(lambda t: bool(seed_re.search(t or "")))
    has_circle = df["persons"].apply(lambda ps: any(p in BLANCHOT_CIRCLE for p in ps))
    has_blanchot = df["persons"].apply(lambda ps: BLANCHOT_ID in ps)
    return {"A": set(np.where(has_seed)[0].tolist()),
            "B": set(np.where(has_circle)[0].tolist()),
            "C": set(np.where(has_blanchot)[0].tolist())}


def sample_corpus(df, groups, cap=CAP, seed=audit.SEED) -> pd.DataFrame:
    keep = sorted(groups["A"] | groups["B"] | groups["C"])
    rest = sorted(set(range(len(df))) - set(keep))
    n_fill = max(0, cap - len(keep))
    rng = np.random.default_rng(seed)
    fill = sorted(rng.choice(rest, size=min(n_fill, len(rest)), replace=False).tolist())
    sel = sorted(set(keep) | set(fill))
    out = df.iloc[sel].reset_index(drop=True)
    print(f"sampled corpus: {len(out)} (groups kept whole: A={len(groups['A'])}, "
          f"B={len(groups['B'])}, C={len(groups['C'])}; D fill={len(fill)})")
    return out


def get_orbis_embeddings(odf) -> np.ndarray:
    """Per-corpus cache, content-hashed — mirrors audit.get_embeddings (qwen3 only)."""
    cache = DATA / "orbis_embeddings.npy"
    meta_path = DATA / "orbis_embeddings_meta.json"
    text_hash = hashlib.sha256("\n".join(odf["text"]).encode("utf-8")).hexdigest()
    if cache.exists() and meta_path.exists():
        emb = np.load(cache)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if emb.shape[0] == len(odf) and meta.get("text_sha256") == text_hash:
            print(f"loaded cached orbis embeddings: {emb.shape}")
            return emb
        print("orbis cache invalid; re-embedding")
    from embed import embed_paragraphs, MODELS
    chunks = [odf["text"].iloc[i:i + EMBED_BATCH].tolist()
              for i in range(0, len(odf), EMBED_BATCH)]
    print(f"embedding {len(odf)} orbis paragraphs via Modal in {len(chunks)} calls …")
    emb = np.vstack([embed_paragraphs(c, model="qwen3") for c in chunks])
    np.save(cache, emb)
    meta_path.write_text(json.dumps({
        "model_id": MODELS["qwen3"], "dim": int(emb.shape[1]),
        "n_paragraphs": int(emb.shape[0]), "text_sha256": text_hash,
        "date_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
    return emb


def main(argv=None):
    ap = argparse.ArgumentParser(description="T3 Orbis control probe")
    ap.add_argument("--groups-only", action="store_true",
                    help="segment + groups + sample table only; NO embedding (pre-registration)")
    args = ap.parse_args(argv)

    df = build_corpus()
    groups = define_groups(df)
    odf = sample_corpus(df, groups)
    g2 = define_groups(odf)   # groups within the sampled corpus (A/B/C preserved whole)
    assert len(g2["A"]) == len(groups["A"]) and len(g2["B"]) == len(groups["B"]) \
        and len(g2["C"]) == len(groups["C"]), "sampling must keep A/B/C whole"

    gtab = pd.DataFrame([{
        "corpus_paragraphs_total": len(df), "sampled": len(odf),
        "n_A_lexicon": len(g2["A"]), "n_B_circle": len(g2["B"]),
        "n_C_blanchot": len(g2["C"]),
        "n_A_and_B": len(g2["A"] & g2["B"]),
        "len_median": int(odf["text"].str.len().median()),
    }])
    gtab.to_csv(RES / "control_corpus_groups.csv", index=False, encoding="utf-8-sig")
    print(gtab.to_string(index=False))
    if args.groups_only:
        print("GROUPS ONLY — no embedding (commit this table before the paid run)")
        return

    emb = get_orbis_embeddings(odf)
    assert emb.shape == (len(odf), 4096)

    cur = json.loads((DATA / "curated_seeds.json").read_text(encoding="utf-8"))
    results = {}
    for name, seeds in [("baseline (BLANCHOT_SEEDS)", list(BLANCHOT_SEEDS)),
                        ("core — discovered only", cur["core_discovered"])]:
        rng = np.random.default_rng(audit.SEED)
        rows, gprobe, _ = audit.blanchot_probe(odf, emb, rng, seeds=seeds)
        if name.startswith("baseline"):
            assert gprobe["n_A"] == len(g2["A"]) and gprobe["n_B"] == len(g2["B"]), \
                "probe recomputed different groups than the committed table"
        results[name] = (rows, gprobe)
    rngn = np.random.default_rng(audit.SEED + 1)
    null = audit.null_test(emb, results["baseline (BLANCHOT_SEEDS)"][1]["n_A"],
                           results["baseline (BLANCHOT_SEEDS)"][1]["n_B"], rngn)

    out_rows = []
    for name, (rows, gprobe) in results.items():
        for r in rows:
            out_rows.append({"seed_set": name, **{k: r[k] for k in
                             ("pair_class", "n_pairs", "median", "cliffs_delta_vs_AD")}})
    out = pd.DataFrame(out_rows)
    out.to_csv(RES / "control_corpus.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    nd = np.asarray([r["cliffs_delta"] for r in null])
    print(f"permutation null: mean={nd.mean():+.3f} band=[{nd.min():+.3f},{nd.max():+.3f}]")

    strat = next(r for r in results["baseline (BLANCHOT_SEEDS)"][0]
                 if r["pair_class"] == "A↔B (no Blanchot)")
    print(f"\nδ_O (stratified A↔B) = {strat['cliffs_delta_vs_AD']:+.3f}  "
          f"(bands: ≥+0.30 prior-dominant | ≤+0.15 corpus-intrinsic | between inconclusive)")


if __name__ == "__main__":
    main()
