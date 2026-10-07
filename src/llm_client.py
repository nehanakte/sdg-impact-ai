"""
Unified LLM client — uses the modern google-genai SDK,
which supports both AIza... and AQ.... API key formats.
"""

import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from src.config import GEMINI_API_KEY, GROQ_API_KEY, GEMINI_MODEL, GROQ_MODEL


# Ordered list of Gemini models to try. The first available one wins.
GEMINI_CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]


def _gemini_generate(prompt: str) -> str:
    """Try each candidate Gemini model until one works."""
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)

    last_error = None
    for name in GEMINI_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=name,
                contents=prompt,
            )
            if response and getattr(response, "text", None):
                return response.text
        except Exception as e:
            last_error = e
            time.sleep(0.5)
            continue

    raise RuntimeError(
        f"All Gemini model attempts failed. Last error: {last_error}"
    )


def _groq_generate(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    chat = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return chat.choices[0].message.content


def generate(prompt: str) -> str:
    """Send a prompt to whichever LLM is configured."""
    if GEMINI_API_KEY:
        return _gemini_generate(prompt)
    if GROQ_API_KEY:
        return _groq_generate(prompt)
    raise RuntimeError("No LLM API key found.")
