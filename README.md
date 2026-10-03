# SDG Impact AI 🌍

An AI-powered tool that maps sustainability initiatives to the UN Sustainable Development Goals (SDGs)
and produces an impact analysis — using a **RAG (Retrieval-Augmented Generation)** pipeline.

## What it demonstrates (GenAI + RAG concepts)

1. **Embeddings** — local `sentence-transformers` (`all-MiniLM-L6-v2`).
2. **Vector search** — FAISS similarity search over the 17 UN SDGs.
3. **Retrieval-Augmented Generation** — retrieved SDGs are injected into the LLM prompt.
4. **Structured LLM output** — the model returns strict JSON that we parse.
5. **Report generation** — a Markdown impact report is saved to `reports/`.

No LangChain. No training from scratch. Clean Python + free LLM API.

## Setup

```bash
# 1. Clone / create the project folder, then:
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get a free Gemini API key
#    https://aistudio.google.com/app/apikey
#    Paste it into .env as GEMINI_API_KEY=...

# 4. First run (builds the FAISS index — takes ~10 seconds)
python cli.py
```

## Usage

### Terminal
```bash
python cli.py
```

### Streamlit UI
```bash
streamlit run app.py
```

## Architecture

```
User initiative
      │
      ▼
[Embedder]  ──►  [FAISS vector store]  ──►  Top-K SDGs (RAG evidence)
      │
      ▼
[Prompt template + retrieved SDGs]  ──►  [LLM (Gemini)]  ──►  JSON analysis
      │
      ▼
[Report generator]  ──►  Markdown report
```

## Project structure

- `src/knowledge_base.py` — loads the 17 SDGs (the RAG corpus).
- `src/embedder.py` — sentence-transformer embeddings.
- `src/vector_store.py` — FAISS index wrapper.
- `src/retriever.py` — builds & queries the index.
- `src/llm_client.py` — unified LLM caller (Gemini / Groq).
- `src/sdg_analyzer.py` — the RAG pipeline (retrieve → prompt → parse).
- `src/report_generator.py` — Markdown report.
- `cli.py`, `app.py` — entry points.