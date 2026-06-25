"""LLM concept extractor for the SITIO §5.1 failure exhibit.

Two modes, both run locally (no Modal) via DSPy + litellm:
  - open extraction:     paragraph text -> [{term, canonical, evidence}]
                         (the LLM returns term + evidence; canonical is derived
                         in Python via canonicalize(term), never asked of the model)
  - closed-set classify: paragraph text + the 31 editor ana_* definitions
                         -> the subset of ana_ids the model thinks apply

Determinism + reproducibility (Step 1's contract): temperature=0, pinned model
snapshots, and a content-hash cache at data/concept_cache/. A cached result is
never re-requested, so regenerating the exhibit from a clone costs nothing. The
cache — not temp=0 — is the real reproducibility guarantee; temp=0 only narrows
first-call variance.

Smoke test (verifies determinism + that litellm can reach both providers):
    python src/concept_extract.py            # 3 sample paragraphs, twice each
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import dspy
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE_DIR = DATA / "concept_cache"
# parents[4] is the 2026/ workspace root in the monorepo layout (same convention
# as parse.py); that is where the shared .env lives, not the repo root.
ENV_PATH = Path(__file__).resolve().parents[4] / ".env"

# Sonnet-only. Gemini 2.0 Flash was dropped after Google retired gemini-2.0-flash
# mid-project (404 "no longer available"); see the model-availability note in
# results/failure_exhibit.md. claude-sonnet-4-5 is Anthropic's stable alias
# pointer; the content-hash cache (not the alias) is the reproducibility contract.
MODELS = {
    "sonnet": "anthropic/claude-sonnet-4-5",
    "haiku": "anthropic/claude-haiku-4-5",
}
TEMPERATURE = 0.0
MAX_TOKENS = 2000

# Load the workspace .env; fall back to an upward search so a flattened clone
# (where parents[4] is not the workspace root) still finds a .env. Keys already
# exported in the shell take precedence over both and make this a no-op.
if not load_dotenv(ENV_PATH):
    load_dotenv(find_dotenv(usecwd=True))


class Concept(BaseModel):
    """One concept the model says a paragraph discusses."""
    term: str = Field(description="the concept as expressed in the paragraph, in Spanish")
    evidence: str = Field(description="a short verbatim Spanish quote from the paragraph that grounds the concept")


class _Open(dspy.Signature):
    """Extrae los conceptos intelectuales clave que el párrafo discute.

    Devuelve cada concepto tal como se expresa en el texto (en español), con una
    cita textual breve que lo justifique. No inventes conceptos que no estén en el texto."""
    paragraph: str = dspy.InputField(desc="un párrafo de la revista SITIO (español)")
    concepts: list[Concept] = dspy.OutputField(desc="los conceptos presentes en el párrafo")


class _OpenTerms(dspy.Signature):
    """Extrae los conceptos intelectuales clave que el párrafo discute. Devuelve solo
    los términos de los conceptos, tal como se expresan en el texto (en español). No
    inventes conceptos que no estén en el texto."""
    paragraph: str = dspy.InputField(desc="un párrafo de la revista SITIO (español)")
    concepts: list[str] = dspy.OutputField(desc="los términos de los conceptos presentes en el párrafo")


class _Closed(dspy.Signature):
    """Dado un párrafo y un inventario fijo de conceptos temáticos (las etiquetas
    interpretativas de los editores de la revista), devuelve los ids de aquellos
    conceptos del inventario que se aplican al párrafo. Devuelve solo ids del inventario."""
    paragraph: str = dspy.InputField(desc="un párrafo de la revista SITIO (español)")
    concept_inventory: str = dspy.InputField(desc="inventario 'id: definición', uno por línea")
    applicable_ids: list[str] = dspy.OutputField(desc="subconjunto de ids del inventario que aplican")


def canonicalize(term: str) -> str:
    """Lowercase, strip accents, collapse whitespace — deterministic, done in
    Python (not by the LLM) so the canonical key is reproducible and matchable."""
    t = unicodedata.normalize("NFKD", term.lower().strip())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def load_inventory() -> dict[str, str]:
    return json.loads((DATA / "concepts.json").read_text(encoding="utf-8"))


def _inventory_text(inv: dict[str, str]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in inv.items())


def _lm(model_id: str) -> dspy.LM:
    # cache=False: our content-hash cache is the single source of truth, and the
    # determinism smoke test needs real (uncached-by-dspy) calls to be meaningful.
    return dspy.LM(model_id, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, cache=False)


def _cache_path(model_id: str, mode: str, text: str, extra: str = "", cache_dir: Path | None = None) -> Path:
    # The key deliberately omits TEMPERATURE / MAX_TOKENS: both are locked
    # constants here, and folding them in would change all 156 committed hashes,
    # orphaning the cache that IS the reproducibility contract. If either is ever
    # promoted to a runtime knob, add it here and regenerate the cache.
    h = hashlib.sha256("\x00".join([model_id, mode, text, extra]).encode("utf-8")).hexdigest()
    return (cache_dir if cache_dir is not None else CACHE_DIR) / f"{h}.json"


def _cached_call(model_id: str, mode: str, text: str, extra: str, run, cache_dir: Path | None = None):
    """Return cached result if present; else call `run()`, cache it with
    provenance, and return it. cache_dir defaults to the module CACHE_DIR; the
    bulk discovery pass passes a separate (gitignored) dir."""
    cdir = cache_dir if cache_dir is not None else CACHE_DIR
    cp = _cache_path(model_id, mode, text, extra, cache_dir=cdir)
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))["result"]
    with dspy.context(lm=_lm(model_id)):
        result = run()
    cdir.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({
        "model_id": model_id,
        "mode": mode,
        "temperature": TEMPERATURE,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def extract_open(text: str, model_key: str, cache_dir: Path | None = None) -> list[dict]:
    """[{term, canonical, evidence}] for one paragraph."""
    model_id = MODELS[model_key]

    def run():
        pred = dspy.Predict(_Open)(paragraph=text)
        return [
            {"term": c.term, "canonical": canonicalize(c.term), "evidence": c.evidence}
            for c in pred.concepts
        ]

    return _cached_call(model_id, "open", text, "", run, cache_dir=cache_dir)


def extract_open_terms(text: str, model_key: str, cache_dir: Path | None = None) -> list[str]:
    """Lean variant of extract_open: returns concept TERMS only (no evidence), to cut
    output tokens for the full-corpus pass. Cached under mode 'open_terms' (separate
    cache key from 'open', so it does not collide with the evidence-bearing results)."""
    model_id = MODELS[model_key]

    def run():
        pred = dspy.Predict(_OpenTerms)(paragraph=text)
        return [t.strip() for t in pred.concepts if t and t.strip()]

    return _cached_call(model_id, "open_terms", text, "", run, cache_dir=cache_dir)


def classify_closed(text: str, model_key: str, inv: dict[str, str] | None = None,
                    cache_dir: Path | None = None) -> list[str]:
    """ana_ids the model thinks apply, filtered to real inventory ids."""
    model_id = MODELS[model_key]
    inv = inv if inv is not None else load_inventory()
    inv_text = _inventory_text(inv)
    valid = set(inv)

    def run():
        pred = dspy.Predict(_Closed)(paragraph=text, concept_inventory=inv_text)
        # Keep only ids that exist in the inventory; drop hallucinated ids.
        return [i for i in pred.applicable_ids if i in valid]

    # extra=inv hash so a changed inventory invalidates the cache for this mode.
    extra = hashlib.sha256(inv_text.encode("utf-8")).hexdigest()
    return _cached_call(model_id, "closed", text, extra, run, cache_dir=cache_dir)


def _smoke() -> None:
    """On 3 sample paragraphs, per model: two fresh passes (cache cleared between)
    to check temp=0 determinism, then a third pass with the cache left in place to
    confirm a 0-call hit. Runs entirely inside an isolated temporary cache dir, so
    the committed data/concept_cache/ artifacts are never touched."""
    import tempfile

    import pandas as pd

    global CACHE_DIR
    df = pd.read_parquet(DATA / "paragraphs.parquet")
    sample = df[df["ana_gold"].str.len() > 0].head(3)
    inv = load_inventory()

    committed = CACHE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        CACHE_DIR = Path(tmp)
        try:
            for model_key in MODELS:
                print(f"\n=== {model_key} ({MODELS[model_key]}) ===")
                try:
                    for _, row in sample.iterrows():
                        text = row["text"]

                        def both():
                            return (extract_open(text, model_key),
                                    classify_closed(text, model_key, inv))

                        runs = []
                        for _pass in range(2):
                            for f in CACHE_DIR.glob("*.json"):  # force real calls
                                f.unlink()
                            runs.append(both())
                        det = ("IDENTICAL" if runs[0] == runs[1]
                               else "DIFFERS (temp=0 is best-effort; cache is the contract)")
                        hit = both()  # no clear: must be a pure cache hit
                        cache_ok = "HIT" if hit == runs[1] else "MISS (unexpected)"
                        print(f"  {row['para_id']}: gold={list(row['ana_gold'])}")
                        print(f"    open  -> {[c['term'] for c in runs[0][0]]}")
                        print(f"    closed-> {runs[0][1]}")
                        print(f"    determinism (2 fresh passes): {det}")
                        print(f"    cache reproducibility (3rd pass): {cache_ok}")
                except Exception as e:
                    # A model can be retired by the provider (gemini-2.0-flash was,
                    # mid-project). The committed cache still reproduces its results;
                    # only fresh calls fail. Report and move on instead of aborting.
                    print(f"  UNAVAILABLE for fresh calls: {type(e).__name__}: {str(e)[:140]}")
        finally:
            CACHE_DIR = committed


if __name__ == "__main__":
    _smoke()
