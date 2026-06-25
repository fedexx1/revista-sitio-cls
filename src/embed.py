"""Modal-hosted Qwen3-Embedding-8B embedding service for the SITIO corpus.

Setup (one-time, runs locally — needs a browser for token auth):
    pip install modal
    modal token new
    modal deploy src/embed.py

Validation (also runs locally, uses ephemeral app context):
    modal run src/embed.py::validate

Use from audit.py after `modal deploy`:
    from embed import embed_paragraphs
    embs = embed_paragraphs(["text 1", "text 2"])   # -> np.ndarray (n, 4096)
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

MODELS = {
    "qwen3": "Qwen/Qwen3-Embedding-8B",
    "bge-m3": "BAAI/bge-m3",
}
MODEL_ID = MODELS["qwen3"]   # legacy name; existing callers/meta keep working
APP_NAME = "sitio-embed"
GPU = "L40S"  # 48GB; 8B in fp16 fits with room. T4/L4 too tight for batch.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _download_models():
    """Bake BOTH models' weights into the image at build time."""
    from sentence_transformers import SentenceTransformer
    for mid in MODELS.values():
        SentenceTransformer(mid)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch>=2.5",
        "transformers>=4.51",
        "sentence-transformers>=3.0",
        "sentencepiece",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(_download_models)
)

app = modal.App(APP_NAME, image=image)


@app.cls(gpu=GPU, scaledown_window=600, timeout=1200)
class Embedder:
    @modal.enter()
    def load(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MODEL_ID, device="cuda")
        self.dim = self.model.get_sentence_embedding_dimension()

    @modal.method()
    def embed(self, texts: list[str]) -> list[list[float]]:
        embs = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=8,
            show_progress_bar=False,
        )
        return embs.tolist()


@app.cls(gpu=GPU, scaledown_window=600, timeout=1200)
class EmbedderBGE:
    @modal.enter()
    def load(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MODELS["bge-m3"], device="cuda")

    @modal.method()
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False,
        ).tolist()


_CLASS_FOR_MODEL = {"qwen3": "Embedder", "bge-m3": "EmbedderBGE"}


def embed_paragraphs(texts: list[str], model: str = "qwen3"):
    """Used by audit.py. Requires `modal deploy` after any class addition."""
    import numpy as np
    cls = modal.Cls.from_name(APP_NAME, _CLASS_FOR_MODEL[model])
    out = cls().embed.remote(texts)
    return np.array(out, dtype=np.float32)


@app.local_entrypoint()
def validate():
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(DATA_DIR / "paragraphs.parquet")
    sample = (
        df[df["text"].str.len() >= 50]["text"]
        .sample(5, random_state=0)
        .tolist()
    )

    t0 = time.monotonic()
    out = Embedder().embed.remote(sample)
    elapsed = time.monotonic() - t0

    arr = np.array(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1)
    print(f"shape: {arr.shape}")
    print(f"dtype: {arr.dtype}")
    print(f"norms (should be ~1.0): {norms.tolist()}")
    print(f"elapsed: {elapsed:.1f}s")

    meta = {
        "model_id": MODEL_ID,
        "gpu_type": GPU,
        "dim": int(arr.shape[1]),
        "dtype": str(arr.dtype),
        "sample_n": int(arr.shape[0]),
        "sample_elapsed_s": round(elapsed, 2),
        "date_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Separate file from the production provenance: this run only embeds a 5-row
    # sample, so it must not masquerade as the meta for the full embeddings.npy
    # (which audit.py::get_embeddings writes after the real embed).
    meta_path = DATA_DIR / "validation_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {meta_path}")
