"""Command-line entry point.

Three subcommands, all driven by one YAML config:

    python main.py prepare  --config config.yaml
    python main.py train     --config config.yaml
    python main.py evaluate  --config config.yaml

The ``use_pointer`` flag in the config selects pointer vs no-pointer behaviour
and the checkpoint name (best_model_pg.pt vs best_model_nopg.pt).
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from summarizer import ensure_nltk, get_device, load_config, set_seed
from summarizer.data import (
    SummarizationDataset,
    Vocabulary,
    build_ngram_scores,
    build_vocabulary,
    collate_fn,
    extractive_summary,
)
from summarizer.infer import evaluate_rouge
from summarizer.model import build_model
from summarizer.train import Trainer


# --------------------------------------------------------------------------- #
def prepare(cfg) -> None:
    """Load the dataset, split it, run the extractive stage, build the vocab."""
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    print(f"Loading {cfg.data.dataset_name} ({cfg.data.dataset_config}) ...")
    dataset = load_dataset(cfg.data.dataset_name, cfg.data.dataset_config)
    df = pd.DataFrame(dataset["train"]).rename(
        columns={"article": "Text", "highlights": "Summary"}
    )
    df = df.dropna(subset=["Text", "Summary"])
    df = df[(df["Text"].str.strip() != "") & (df["Summary"].str.strip() != "")]
    if cfg.data.subsample_size:
        df = df.sample(n=cfg.data.subsample_size, random_state=cfg.seed)
    df = df.reset_index(drop=True)

    train_df, temp_df = train_test_split(df, test_size=cfg.data.temp_size, random_state=cfg.seed)
    val_df, test_df = train_test_split(temp_df, test_size=cfg.data.val_test_ratio, random_state=cfg.seed)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print("Building n-gram rarity dictionary ...")
    ngram_dict, max_score = build_ngram_scores(
        tqdm(train_df["Text"], desc="3-gram dictionary"),
        ngram_n=cfg.extractive.ngram_n,
        min_count=cfg.extractive.min_ngram_count,
    )
    print(f"Unique n-grams kept: {len(ngram_dict):,} | max score: {max_score:.4f}")

    budget = cfg.extractive.token_budget
    os.makedirs(cfg.data.processed_dir, exist_ok=True)
    for name, split_df in {"train": train_df, "val": val_df, "test": test_df}.items():
        extracts = [
            extractive_summary(t, ngram_dict, max_score, budget, cfg.extractive.ngram_n)
            for t in tqdm(split_df["Text"], desc=f"{name} extractive ({budget} tok)")
        ]
        out = pd.DataFrame({"Text": extracts, "Summary": split_df["Summary"].tolist()})
        out.to_csv(os.path.join(cfg.data.processed_dir, f"{name}.csv"), index=False)

    train_csv = pd.read_csv(os.path.join(cfg.data.processed_dir, "train.csv"))
    vocab = build_vocabulary(
        list(train_csv["Text"].astype(str)) + list(train_csv["Summary"].astype(str)),
        max_vocab=cfg.vocab.max_vocab,
    )
    vocab.save(os.path.join(cfg.data.processed_dir, "vocab.json"))
    print(f"Vocabulary size: {len(vocab):,}. Data preparation complete.")


def _load_split(processed_dir: str, name: str):
    df = pd.read_csv(os.path.join(processed_dir, f"{name}.csv"))
    return df["Text"].astype(str).tolist(), df["Summary"].astype(str).tolist()


def train(cfg) -> None:
    """Train the model and save the best checkpoint."""
    device = get_device()
    print(f"Device: {device}")
    vocab = Vocabulary.load(os.path.join(cfg.data.processed_dir, "vocab.json"))

    train_texts, train_sums = _load_split(cfg.data.processed_dir, "train")
    val_texts, val_sums = _load_split(cfg.data.processed_dir, "val")

    train_ds = SummarizationDataset(train_texts, train_sums, vocab, cfg.model.max_src, cfg.model.max_tgt)
    val_ds = SummarizationDataset(val_texts, val_sums, vocab, cfg.model.max_src, cfg.model.max_tgt)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=cfg.training.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=cfg.training.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = build_model(len(vocab), cfg.model, vocab.pad_idx)
    Trainer(model, cfg, vocab.pad_idx, vocab.unk_idx, device).fit(train_loader, val_loader)


def evaluate(cfg) -> None:
    """Load the best checkpoint and report ROUGE on the test set."""
    device = get_device()
    print(f"Device: {device}")
    vocab = Vocabulary.load(os.path.join(cfg.data.processed_dir, "vocab.json"))

    test_df = pd.read_csv(os.path.join(cfg.data.processed_dir, "test.csv"))
    texts = test_df["Text"].astype(str).tolist()
    summaries = test_df["Summary"].astype(str).tolist()

    model = build_model(len(vocab), cfg.model, vocab.pad_idx).to(device)
    ckpt = os.path.join(cfg.training.checkpoint_dir, cfg.checkpoint_name)
    print(f"Loading checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    print(f"Evaluating (use_pointer={cfg.model.use_pointer}) on {len(texts)} examples ...")
    scores = evaluate_rouge(model, texts, summaries, vocab, cfg, device)
    print("\n=== Test ROUGE (F-measure) ===")
    for rt, val in scores.items():
        print(f"  {rt}: {val:.4f}")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Rare-words summarization pipeline.")
    parser.add_argument("command", choices=["prepare", "train", "evaluate"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    ensure_nltk()

    {"prepare": prepare, "train": train, "evaluate": evaluate}[args.command](cfg)


if __name__ == "__main__":
    main()
