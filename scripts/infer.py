import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import sys

# Load model and tokenizer
model_path = "models/intentai"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Define labels in the same order used during training
labels = ["manipulative", "informative", "polarizing", "emotionally_loaded"]

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        print("Raw logits:", outputs.logits.tolist())  # Debug info
        probs = torch.sigmoid(outputs.logits).squeeze().tolist()
    return {label: round(prob, 3) for label, prob in zip(labels, probs)}, {label: int(prob > 0.3) for label, prob in zip(labels, probs)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer.py \"Your input text here\"")
    else:
        input_text = sys.argv[1]
        result, predictions = predict_intent(input_text)
        print("\n=== Predicted intent scores ===")
        for label, score in result.items():
            print(f"{label}: {score}")
        print("\n=== Threshold-passed labels (threshold > 0.3) ===")
        for label, passed in predictions.items():
            if passed:
                print(f"{label}: Likely")
