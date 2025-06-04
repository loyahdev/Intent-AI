

import pandas as pd

def load_keywords_dataset(filepath="datasets/train.csv"):
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} keywords from {filepath}")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return pd.DataFrame()

def save_keywords_dataset(data, filepath="data/keywords_train.csv"):
    try:
        df = pd.DataFrame(data, columns=["keyword", "category"])
        df.to_csv(filepath, index=False)
        print(f"Saved dataset to {filepath}")
    except Exception as e:
        print(f"Failed to save dataset: {e}")

if __name__ == "__main__":
    df = load_keywords_dataset()
    # Example processing step (if needed)
    filtered = df[df["keyword"].str.len() > 2]
    save_keywords_dataset(filtered.values.tolist())