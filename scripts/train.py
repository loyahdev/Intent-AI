#!/usr/bin/env python3
"""
train.py
--------
Intended to be fast (< 1 min on M‑series Mac) while good and
reasonable precision/recall for four binary intent labels.

# custom paths
python3 scripts/train.py --csv data/train.csv --model src/local/LocalIntentClassifier.pkl
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


TARGET_COLS = ["manipulative", "polarizing", "emotionally_loaded", "informative"]


def build_pipeline(max_features: int = 50_000) -> Pipeline:
    """
    Returns a scikit‑learn pipeline:
      • TF‑IDF (1‑2 grams)
      • One‑vs‑Rest logistic regression
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    lowercase=True,
                    max_features=max_features,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=1000,
                        n_jobs=-1,
                        solver="liblinear",
                        class_weight="balanced",
                    )
                ),
            ),
        ]
    )


def train(csv_path: Path, model_path: Path) -> None:
    df = pd.read_csv(csv_path).fillna("")
    X = df["text"].astype(str)
    Y = df[TARGET_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # Evaluation
    preds = pipe.predict(X_test)
    print("\nClassification report (micro avg):")
    print(
        classification_report(
            y_test, preds, target_names=TARGET_COLS, zero_division=0
        )
    )

    # Persist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    print(f"\n[✓] Saved model → {model_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Intent‑AI local model.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/train.csv"),
        help="Path to training CSV.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("src/local/LocalIntentClassifier.pkl"),
        help="Output path for pickled model.",
    )
    args = parser.parse_args()

    train(args.csv, args.model)
