"""Subject router.

Version 0: simple keyword scoring (a placeholder).
Later this file is replaced by a trained ML classifier
(TF-IDF + LogReg, sentence embeddings, DistilBERT).
The function signature must stay the same so nothing else breaks.
"""

KEYWORDS = {
    "math": ["solve", "equation", "derivative", "integral", "sin", "cos", "prove",
             "factor", "sum", "=", "limit", "matrix", "x^2", "x²"],
    "phys": ["force", "velocity", "speed", "fall", "mass", "kg", "energy",
             "acceleration", "gravity", "current", "voltage", "wave"],
    "code": ["python", "function", "code", "list", "javascript", "bug", "array",
             "loop", "class", "sql", "api", "write a"],
    "chem": ["balance", "molecule", "acid", "reaction", "mol", "element",
             "bond", "ph", "->", "→"],
}


def route(text: str) -> dict[str, float]:
    """Return {subject: probability} for a question."""
    text = text.lower()
    scores = {}
    for subject, words in KEYWORDS.items():
        scores[subject] = 0.4 + sum(2 for w in words if w in text)
    total = sum(scores.values())
    return {s: v / total for s, v in scores.items()}
