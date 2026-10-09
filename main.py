from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()  # must run before importing llm

from llm import ask_llm  # noqa: E402
from ml import analyze  # noqa: E402

app = FastAPI(title="Math Assistant")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


class Question(BaseModel):
    text: str


@app.post("/api/ask")
async def ask(q: Question):
    text = q.text.strip()
    if not text:
        raise HTTPException(400, "Empty question")

    info = analyze(text)  # our own trained ML models
    hint = ""
    if info:
        # The ML prediction adapts how the LLM explains the problem.
        hint = f"The problem topic is {info['topic']}, difficulty {info['level']} out of 5."
        if info["level"] <= 2:
            hint += " Explain very simply and slowly."
        elif info["level"] >= 4:
            hint += " This is hard: be rigorous and check the result."
    try:
        answer = await ask_llm(text, hint)
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    return {"answer": answer, "analysis": info}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")