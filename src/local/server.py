#!/usr/bin/env python3
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "LocalIntentClassifier.pkl"
pipe = joblib.load(MODEL_PATH)

app = FastAPI(title="Intent‑AI Local Model")

class In(BaseModel):
    text: str

@app.post("/predict")
def predict(body: In):
    probs = pipe.predict_proba([body.text])[0]  # list of len 4
    labels = ["manipulative", "polarizing", "emotionally_loaded", "informative"]
    return dict(zip(labels, probs.tolist()))

if __name__ == "__main__":
    # runs at http://127.0.0.1:8001/predict
    uvicorn.run(app, host="0.0.0.0", port=8001)