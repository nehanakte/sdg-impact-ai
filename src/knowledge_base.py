"""Loads and prepares the SDG knowledge base (documents for RAG)."""

import json
from src.config import KB_PATH


def load_sdg_documents():
    """
    Load SDG knowledge base and turn each SDG into a text 'document'
    suitable for embedding.
    Returns a list of dicts: {id, text, metadata}
    """
    with open(KB_PATH, "r", encoding="utf-8") as f:
        sdgs = json.load(f)

    documents = []
    for sdg in sdgs:
        text = (
            f"SDG {sdg['sdg_id']}: {sdg['title']}. "
            f"{sdg['description']} "
            f"Keywords: {', '.join(sdg['keywords'])}. "
            f"Indicators: {', '.join(sdg['indicators'])}."
        )
        documents.append({
            "id": sdg["sdg_id"],
            "text": text,
            "metadata": {
                "sdg_id": sdg["sdg_id"],
                "title": sdg["title"],
                "keywords": sdg["keywords"],
                "indicators": sdg["indicators"],
            },
        })
    return documents


def get_all_sdgs():
    """Returns the raw SDG list."""
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    docs = load_sdg_documents()
    print(f"Loaded {len(docs)} SDG documents.")
    print("Sample:", docs[0]["text"][:200])