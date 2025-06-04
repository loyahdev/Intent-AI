import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Load dataset
df = pd.read_csv("data/train.csv", encoding="utf-8", on_bad_lines='skip')

# Basic checks
assert 'text' in df.columns and 'label' in df.columns, "CSV must contain 'text' and 'label' columns"

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Create model pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Validation Accuracy: {score:.4f}")

# Save model
joblib.dump(model, "model/keyword_classifier.joblib")
print("Model saved to model/keyword_classifier.joblib")
