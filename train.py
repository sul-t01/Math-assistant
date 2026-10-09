"""
Train and compare classical ML models on the MATH dataset.

Two tasks (both predicted from the problem text only):
  1. topic : 7 classes (Algebra, Geometry, Number Theory, ...)
  2. level : difficulty 1..5 (ordinal)

Models compared: Naive Bayes, Logistic Regression, Linear SVM
(all on TF-IDF features). The best model per task (by macro-F1 on the
test set) is saved to models/ and used by the website.

Run:  python train.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Several mirrors of the MATH dataset; the first one that works is used.
DATASET_SOURCES = [
    "DigitalLearningGmbH/MATH-lighteval",
    "WhiteGiverPlus/MATH",
]


def load_data():
    """Return (train_rows, test_rows); each row is a dict with problem/level/type."""
    from datasets import load_dataset

    last_error = None
    for name in DATASET_SOURCES:
        try:
            ds = load_dataset(name)
            print(f"Loaded dataset: {name}")
            return list(ds["train"]), list(ds["test"])
        except Exception as e:  # try next mirror
            last_error = e
            print(f"Could not load {name}: {e}")
    raise RuntimeError(f"No dataset source worked. Last error: {last_error}")


def build_xy(rows, target):
    X, y = [], []
    for r in rows:
        text = r["problem"]
        if target == "level":
            lvl = str(r["level"]).replace("Level", "").strip()
            if not lvl.isdigit():  # skip "Level ?" rows
                continue
            label = int(lvl)
        else:
            label = r["type"]
        X.append(text)
        y.append(label)
    return X, np.array(y)


def make_models():
    def tfidf():
        return TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b|[^\w\s]",  # keep math symbols
        )
    return {
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.1))]),
        "Logistic Regression": Pipeline([("tfidf", tfidf()),
                                         ("clf", LogisticRegression(max_iter=2000, C=5))]),
        "Linear SVM": Pipeline([("tfidf", tfidf()), ("clf", LinearSVC(C=0.5))]),
    }


def run_task(target, train_rows, test_rows):
    print(f"\n===== Task: {target} =====")
    Xtr, ytr = build_xy(train_rows, target)
    Xte, yte = build_xy(test_rows, target)
    print(f"train={len(Xtr)}  test={len(Xte)}")

    results, fitted = {}, {}
    for name, model in make_models().items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        res = {
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "macro_f1": round(float(f1_score(yte, pred, average="macro")), 4),
        }
        if target == "level":  # ordinal: how often are we off by at most 1 level?
            res["within_1_level"] = round(float(np.mean(np.abs(pred - yte) <= 1)), 4)
        results[name] = res
        fitted[name] = (model, pred)
        print(f"{name:22s} {res}")

    # majority-class baseline, so the numbers have context
    values, counts = np.unique(ytr, return_counts=True)
    majority = values[np.argmax(counts)]
    base_pred = np.full(len(yte), majority)
    results["Majority baseline"] = {
        "accuracy": round(float(accuracy_score(yte, base_pred)), 4),
        "macro_f1": round(float(f1_score(yte, base_pred, average="macro")), 4),
    }
    print("Majority baseline      ", results["Majority baseline"])

    best = max(fitted, key=lambda n: results[n]["macro_f1"])
    model, pred = fitted[best]
    print(f"\nBest model for '{target}': {best}")
    print(classification_report(yte, pred, zero_division=0))

    labels = sorted(set(yte.tolist()))
    cm = confusion_matrix(yte, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=[str(l) for l in labels]).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"{target}: {best}")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"confusion_{target}.png", dpi=150)
    plt.close(fig)

    joblib.dump(model, MODELS_DIR / f"{target}.joblib")
    return {"best_model": best, "scores": results}


def main():
    train_rows, test_rows = load_data()
    out = {t: run_task(t, train_rows, test_rows) for t in ("type", "level")}
    (RESULTS_DIR / "results.json").write_text(json.dumps(out, indent=2))
    print("\nSaved models to models/ and results to results/")


if __name__ == "__main__":
    main()
