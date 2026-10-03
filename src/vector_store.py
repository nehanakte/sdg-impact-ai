"""A thin FAISS wrapper — build, save, load, search."""

import json
import faiss
import numpy as np

from src.config import INDEX_PATH, META_PATH


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner-product; works with normalized vectors
        self.metadata = []  # parallel list to index rows

    # ---------- Build ----------
    def add(self, embeddings: np.ndarray, metadata: list):
        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    # ---------- Search ----------
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "score": float(score),
                "metadata": self.metadata[idx],
            })
        return results

    # ---------- Persist ----------
    def save(self):
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls):
        if not INDEX_PATH.exists() or not META_PATH.exists():
            return None
        index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        store = cls(dim=index.d)
        store.index = index
        store.metadata = metadata
        return store