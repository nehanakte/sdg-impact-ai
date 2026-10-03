"""Embeds text using a local sentence-transformer model."""

from sentence_transformers import SentenceTransformer
import numpy as np
from src.config import EMBEDDING_MODEL

_model = None


def get_model():
    """Lazy-loads the sentence-transformer model (cached)."""
    global _model
    if _model is None:
        print(f"[embedder] Loading model: {EMBEDDING_MODEL} ...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts):
    """Return a numpy array of shape (n, dim) with float32 embeddings."""
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype("float32")


def embed_query(text):
    """Embed a single query. Returns shape (1, dim)."""
    return embed_texts([text])