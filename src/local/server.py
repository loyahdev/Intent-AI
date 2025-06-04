#!/usr/bin/env python3
"""
FastAPI wrapper around the pickled local intent classifier.

Run:
    python3 src/local/server.py
"""
from pathlib import Path
from typing import Dict

import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

MODEL_PATH = Path(__file__).parent / "LocalIntentClassifier.pkl"
pipeline = joblib.load(MODEL_PATH)

app = FastAPI(title="Intent‑AI Local Model", version="0.1.0")

class InText(BaseModel):
    text: str

@app.post("/predict")
def predict(body: InText) -> Dict[str, float]:
    labels = ["manipulative", "polarizing", "emotionally_loaded", "informative"]
    probs = pipeline.predict_proba([body.text])[0]
    return dict(zip(labels, probs.tolist()))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)