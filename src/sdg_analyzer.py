"""The RAG pipeline: retrieve relevant SDGs and ask the LLM to analyze impact."""

import json
import re

from src.retriever import build_index, retrieve
from src.llm_client import generate
from src.config import TOP_K


_store = None


def get_store():
    global _store
    if _store is None:
        _store = build_index()
    return _store


def _format_context(retrieved):
    """Turn retrieved SDG metadata into a context block for the prompt."""
    lines = []
    for r in retrieved:
        m = r["metadata"]
        lines.append(
            f"- SDG {m['sdg_id']}: {m['title']} (similarity={r['score']:.3f})\n"
            f"  Keywords: {', '.join(m['keywords'])}\n"
            f"  Indicators: {', '.join(m['indicators'])}"
        )
    return "\n".join(lines)


PROMPT_TEMPLATE = """You are an expert sustainability analyst specialized in the UN Sustainable Development Goals (SDGs).

A user has described an initiative. Your job is to map this initiative to the most relevant SDGs
and produce a structured impact analysis.

Use ONLY the SDG information provided in the CONTEXT below. Do not invent SDGs.

INITIATIVE:
\"\"\"{initiative}\"\"\"

CONTEXT (retrieved SDGs, ranked by relevance):
{context}

TASK:
1. Identify the primary SDG (the single best match).
2. Identify up to 4 secondary SDGs that are also relevant.
3. For each identified SDG, give a short explanation (1-2 sentences) of WHY the initiative contributes.
4. Provide an overall impact analysis: expected positive outcomes, risks/trade-offs, and a rough
   qualitative impact level (Low / Medium / High) with justification.
5. Suggest 2-3 concrete KPIs to measure success, tied to the SDG indicators.

Return ONLY valid JSON in exactly this schema (no markdown, no code fences):
{{
  "primary_sdg": {{"sdg_id": <int>, "title": "<str>", "reason": "<str>"}},
  "secondary_sdgs": [
    {{"sdg_id": <int>, "title": "<str>", "reason": "<str>"}}
  ],
  "impact_analysis": {{
    "positive_outcomes": ["<str>", ...],
    "risks_and_tradeoffs": ["<str>", ...],
    "impact_level": "Low | Medium | High",
    "justification": "<str>"
  }},
  "suggested_kpis": ["<str>", "<str>", "<str>"]
}}
"""


def analyze_initiative(initiative: str, top_k: int = TOP_K) -> dict:
    """
    Full RAG pipeline:
      1. Retrieve relevant SDGs from vector store.
      2. Build augmented prompt.
      3. Call LLM.
      4. Parse JSON result.
    Returns a dict with keys: retrieved, prompt, result (parsed), raw (raw text).
    """
    if not initiative or len(initiative.strip()) < 10:
        raise ValueError("Please describe the initiative in at least 10 characters.")

    store = get_store()
    retrieved = retrieve(initiative, store, top_k=top_k)
    context = _format_context(retrieved)
    prompt = PROMPT_TEMPLATE.format(initiative=initiative.strip(), context=context)

    raw = generate(prompt)
    parsed = _parse_json(raw)

    return {
        "retrieved": retrieved,
        "prompt": prompt,
        "result": parsed,
        "raw": raw,
    }


def _parse_json(text: str) -> dict:
    """Extract JSON from LLM response, tolerating stray text / code fences."""
    text = text.strip()

    # Strip code fences if present
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: grab first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"LLM did not return valid JSON. Raw output:\n{text[:500]}")


if __name__ == "__main__":
    demo = "A solar microgrid project that powers rural schools and health clinics in Bihar."
    out = analyze_initiative(demo)
    print(json.dumps(out["result"], indent=2))