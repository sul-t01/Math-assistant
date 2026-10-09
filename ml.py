"""Loads the trained models (made by train.py) and predicts topic + difficulty."""
from pathlib import Path

import joblib

MODELS_DIR = Path(__file__).resolve().parent / "models"
_cache = {}


def _load(name):
    if name not in _cache:
        path = MODELS_DIR / f"{name}.joblib"
        _cache[name] = joblib.load(path) if path.exists() else None
    return _cache[name]


def analyze(text: str):
    """Return {'topic': str, 'level': int} or None if models are not trained yet."""
    topic_model, level_model = _load("type"), _load("level")
    if topic_model is None or level_model is None:
        return None
    return {
        "topic": str(topic_model.predict([text])[0]),
        "level": int(level_model.predict([text])[0]),
    }
