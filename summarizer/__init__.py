"""rare-words summarization package.

Holds the configuration objects and a few small helpers. The heavier modules
(``data``, ``model``, ``train``, ``infer``) are imported directly where needed.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import yaml


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "cnn_dailymail"
    dataset_config: str = "3.0.0"
    text_column: str = "Text"
    summary_column: str = "Summary"
    subsample_size: Optional[int] = None
    temp_size: float = 0.2
    val_test_ratio: float = 0.5
    processed_dir: str = "data/processed"


@dataclass(frozen=True)
class ExtractiveConfig:
    token_budget: int = 400
    ngram_n: int = 3
    min_ngram_count: int = 1


@dataclass(frozen=True)
class VocabConfig:
    max_vocab: int = 15000


@dataclass(frozen=True)
class ModelConfig:
    use_pointer: bool = True
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    max_src: int = 512
    max_tgt: int = 150


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 30
    lr: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0
    patience: int = 3
    num_workers: int = 4
    checkpoint_dir: str = "checkpoints"


@dataclass(frozen=True)
class InferenceConfig:
    beam_size: int = 4
    length_alpha: float = 0.7


@dataclass(frozen=True)
class EvalConfig:
    rouge_types: List[str] = field(
        default_factory=lambda: ["rouge1", "rouge2", "rougeL"]
    )
    use_stemmer: bool = True


@dataclass(frozen=True)
class Config:
    """Top-level configuration aggregating every sub-section."""

    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    extractive: ExtractiveConfig = field(default_factory=ExtractiveConfig)
    vocab: VocabConfig = field(default_factory=VocabConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def checkpoint_name(self) -> str:
        """Checkpoint file name, e.g. ``best_model_pg.pt``."""
        tag = "pg" if self.model.use_pointer else "nopg"
        return f"best_model_{tag}.pt"


def _section(cls, raw):
    if raw is None:
        return cls()
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: str) -> Config:
    """Load a :class:`Config` from a YAML file."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config(
        seed=raw.get("seed", 42),
        data=_section(DataConfig, raw.get("data")),
        extractive=_section(ExtractiveConfig, raw.get("extractive")),
        vocab=_section(VocabConfig, raw.get("vocab")),
        model=_section(ModelConfig, raw.get("model")),
        training=_section(TrainingConfig, raw.get("training")),
        inference=_section(InferenceConfig, raw.get("inference")),
        eval=_section(EvalConfig, raw.get("eval")),
    )


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_nltk() -> None:
    """Download the NLTK resources the pipeline relies on (idempotent)."""
    import nltk

    for resource in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:  # network/offline tolerant
            pass


__all__ = ["Config", "load_config", "set_seed", "get_device", "ensure_nltk"]
