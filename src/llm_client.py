"""
Unified LLM client — uses Google Gemini if GEMINI_API_KEY is set,
otherwise falls back to Groq. Keeps all LLM code in one place.
"""

import time
import warnings

# Silence the deprecation warning from google.generativeai
warnings.filterwarnings("ignore", category=FutureWarning)

from src.config import GEMINI_API_KEY, GROQ_API_KEY, GEMINI_MODEL, GROQ_MODEL


# Ordered list of Gemini models to try. The first available one wins.
GEMINI_CANDIDATES = [
    GEMINI_MODEL,               # from config.py (gemini-2.5-flash)
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

def _gemini_generate(prompt: str) -> str:
    """Try each candidate Gemini model until one works."""
    import google.generativeai as genai
    print(f"[DEBUG] GEMINI_API_KEY length = {len(GEMINI_API_KEY)}", flush=True)
    print(f"[DEBUG] GEMINI_API_KEY prefix = {GEMINI_API_KEY[:6]}...", flush=True)
    genai.configure(api_key=GEMINI_API_KEY)
    # ... rest of the function unchanged

def _gemini_generate(prompt: str) -> str:
    """Try each candidate Gemini model until one works."""
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

    last_error = None
    for name in GEMINI_CANDIDATES:
        try:
            model = genai.GenerativeModel(name)
            response = model.generate_content(prompt)
            if response and getattr(response, "text", None):
                return response.text

        except Exception as e:
            print(f"[DEBUG] {name} failed: {type(e).__name__}: {e}", flush=True)
            last_error = e
            time.sleep(0.5)
            continue    

    raise RuntimeError(
        f"All Gemini model attempts failed. Last error: {last_error}\n"
        f"Tip: run this to list available models:\n"
        f"  python -c \"import google.generativeai as genai, os; "
        f"from dotenv import load_dotenv; load_dotenv(); "
        f"genai.configure(api_key=os.getenv('GEMINI_API_KEY')); "
        f"[print(m.name) for m in genai.list_models()]\""
    )


def _groq_generate(prompt: str) -> str:
    from groq import Groq  # only imported if used
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
    raise RuntimeError(
        "No LLM API key found. Set GEMINI_API_KEY (recommended) or GROQ_API_KEY in your .env file."
    )
