#!/usr/bin/env python3
"""
dataset-gen.py
--------------
Datasets being used
  • Fake‑News dataset (mrm8488)       – manipulative + polarizing
  • Anthropic Persuasion Corpus       – manipulative
  • GoEmotions (anger/fear/disgust)   – emotionally_loaded
  • Wikipedia sentences               – informative (neutral)
"""

from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
from pathlib import Path


def load_propaganda(max_len: int = 300) -> Dataset:
    try:
        ds = load_dataset("QCRI/sem_eval_2020_task11", split="train")
    except Exception:
        print(
            "[!] SemEval‑2020 Task 11 dataset not found on the Hub "
            "— skipping propaganda examples."
        )
        return Dataset.from_list([])

    rows = [
        {
            "text": ex["text"].strip().replace("\n", " "),
            "manipulative": 1,
            "polarizing": 1,
            "emotionally_loaded": 0,
            "informative": 0,
        }
        for ex in ds
        if len(ex["text"]) <= max_len
    ]
    return Dataset.from_list(rows)


def load_persuasion() -> Dataset:
    try:
        ds = load_dataset("Anthropic/persuasion", split="train")
    except Exception:
        print("[!] Anthropic persuasion dataset not found — skipping.")
        return Dataset.from_list([])

    # Some rows have a `label` field (1 = persuasive). In other versions
    # the boolean may be stored under `is_persuasive`. We check both.
    rows = []
    for ex in ds:
        label = ex.get("label", ex.get("is_persuasive", None))
        if label is None:
            continue
        if label in (1, True, "persuasive"):
            rows.append(
                {
                    "text": str(ex["text"]).strip().replace("\n", " "),
                    "manipulative": 1,
                    "polarizing": 0,
                    "emotionally_loaded": 0,
                    "informative": 0,
                }
            )
    return Dataset.from_list(rows)


def load_fake_news(sample: int = 5000) -> Dataset:
    try:
        ds = load_dataset("mrm8488/fake-news", split="train")
    except Exception:
        print("[!] Fake‑news dataset not found — skipping.")
        return Dataset.from_list([])

    # Down‑sample for speed if desired
    if sample and len(ds) > sample:
        ds = ds.shuffle(seed=42).select(range(sample))

    rows = [
        {
            "text": str(ex["text"]).strip().replace("\n", " "),
            "manipulative": 1,
            "polarizing": 1,
            "emotionally_loaded": 0,
            "informative": 0,
        }
        for ex in ds
        if ex.get("label", 0) == 1
    ]
    return Dataset.from_list(rows)


def load_emotions(max_len: int = 200) -> Dataset:
    ds = None
    try:
        ds = load_dataset("go_emotions", split="train")
        mode = "multi"
    except Exception:
        try:
            ds = load_dataset("SetFit/go_emotions", split="train")
            mode = "single"
        except Exception:
            print("[!] GoEmotions dataset not found — skipping.")
            return Dataset.from_list([])

    target_labels = {"anger", "fear", "disgust", "sadness"}

    # Resolve label names list depending on schema
    if mode == "multi":
        label_names = ds.features["labels"].feature.names  # Sequence(ClassLabel)
    else:  # single label
        label_names = ds.features["label"].names

    rows = []
    for ex in ds:
        if mode == "multi":
            label_ids = ex["labels"]
            names = {label_names[i] for i in label_ids}
        else:
            names = {label_names[ex["label"]]}

        if names & target_labels and len(ex["text"]) <= max_len:
            rows.append(
                {
                    "text": ex["text"].strip().replace("\n", " "),
                    "manipulative": 0,
                    "polarizing": 0,
                    "emotionally_loaded": 1,
                    "informative": 0,
                }
            )
    return Dataset.from_list(rows)


def load_wikipedia(limit: int = 5000) -> Dataset:
    split = f"train[:{limit}]"
    ds = load_dataset("wikipedia", "20220301.en", split=split)
    rows = []
    for ex in ds:
        first_sentence = ex["text"].split("\n")[0]
        if first_sentence:
            rows.append(
                {
                    "text": first_sentence.strip(),
                    "manipulative": 0,
                    "polarizing": 0,
                    "emotionally_loaded": 0,
                    "informative": 1,
                }
            )
    return Dataset.from_list(rows)


def main() -> None:
    propaganda = load_propaganda()
    persuasion = load_persuasion()
    fake_news = load_fake_news()
    emotions = load_emotions()
    wiki = load_wikipedia()

    datasets_to_merge = [
        d for d in (propaganda, persuasion, fake_news, emotions, wiki) if len(d) > 0
    ]
    full_ds = concatenate_datasets(datasets_to_merge).shuffle(seed=42)
    df = pd.DataFrame(full_ds)

    out_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[✓] Wrote {len(df):,} rows → {out_path}")


if __name__ == "__main__":
    main()