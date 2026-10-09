"""Talks to the Groq API (OpenAI-compatible, free tier)."""
import os
import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are a patient math tutor. Solve the problem step by step, "
    "then state the final answer clearly."
)


async def ask_llm(question: str, hint: str = "") -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Check your .env file.")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": (SYSTEM_PROMPT + " " + hint).strip()},
            {"role": "user", "content": question},
        ],
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            GROQ_URL, json=payload, headers={"Authorization": f"Bearer {key}"}
        )
    if r.status_code != 200:
        raise RuntimeError(f"Groq error {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]
