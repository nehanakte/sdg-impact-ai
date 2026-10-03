"""Builds/loads the vector store and exposes a retrieve() function."""

from src.knowledge_base import load_sdg_documents
from src.embedder import embed_texts, embed_query
from src.vector_store import VectorStore
from src.config import TOP_K


def build_index(force_rebuild: bool = False) -> VectorStore:
    """Build the FAISS index from the SDG knowledge base (or load from disk)."""
    if not force_rebuild:
        existing = VectorStore.load()
        if existing is not None:
            print("[retriever] Loaded existing index from disk.")
            return existing

    print("[retriever] Building SDG index ...")
    docs = load_sdg_documents()
    texts = [d["text"] for d in docs]
    metas = [d["metadata"] for d in docs]

    embeddings = embed_texts(texts)
    store = VectorStore(dim=embeddings.shape[1])
    store.add(embeddings, metas)
    store.save()
    print(f"[retriever] Indexed {len(docs)} SDGs.")
    return store


def retrieve(query: str, store: VectorStore, top_k: int = TOP_K):
    """Retrieve top_k most relevant SDGs for a given initiative description."""
    q_emb = embed_query(query)
    return store.search(q_emb, top_k=top_k)


if __name__ == "__main__":
    store = build_index()
    results = retrieve("We install solar panels in rural villages", store)
    for r in results:
        print(r["score"], "->", r["metadata"]["title"])