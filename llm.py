"""Talks to the Groq API (OpenAI-compatible, free tier)."""
import os
import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are a patient math tutor. Solve the problem step by step, "
    "then state the final answer clearly. Use the earlier messages of the "
    "conversation as context when the user refers to them."
)


async def ask_llm(question: str, hint: str = "", history: list | None = None) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Check your .env file.")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    messages = [{"role": "system", "content": (SYSTEM_PROMPT + " " + hint).strip()}]
    for m in (history or [])[-10:]:  # last 10 messages of context
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            messages.append({"role": m["role"], "content": m["content"][:4000]})
    messages.append({"role": "user", "content": question})

    payload = {"model": model, "messages": messages, "temperature": 0.3}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            GROQ_URL, json=payload, headers={"Authorization": f"Bearer {key}"}
        )
    if r.status_code != 200:
        raise RuntimeError(f"Groq error {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]
