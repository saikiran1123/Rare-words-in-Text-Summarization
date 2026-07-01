"""Data: extractive stage, vocabulary, dataset and collate.

Three pieces that all live on the data path:

* Extractive stage (paper Algorithm 1): n-gram rarity scoring + greedy sentence
  selection to a token budget, emitted in original document order.
* Vocabulary: word->index map capped at ``max_vocab`` plus a per-word
  frequency-rarity score used by the M2 encoder attention.
* Dataset / collate: turns (extract, summary) pairs into model tensors, with
  the extended-vocabulary OOV handling the pointer-generator needs.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from nltk import ngrams as nltk_ngrams
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

NGram = Tuple[str, ...]

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
SPECIAL = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


# --------------------------------------------------------------------------- #
# Extractive stage
# --------------------------------------------------------------------------- #
def _stopword_set() -> set:
    return set(stopwords.words("english"))


def remove_stopwords_tokens(tokens: Iterable[str], stop: set) -> List[str]:
    """Drop stop words and non-alphabetic tokens (lower-cased comparison)."""
    return [t for t in tokens if t.lower() not in stop and t.isalpha()]


def build_ngram_scores(
    texts: Iterable[str], ngram_n: int = 3, min_count: int = 1
) -> Tuple[Dict[NGram, float], float]:
    """Build the corpus-level n-gram rarity dictionary.

    Returns ``(score_dict, max_score)`` where each n-gram maps to
    ``1 / log(occ) + shift`` and ``max_score`` is assigned to unseen n-grams.
    """
    stop = _stopword_set()
    counts: Counter = Counter()
    for text in texts:
        tokens = remove_stopwords_tokens(word_tokenize(str(text).lower()), stop)
        counts.update(list(nltk_ngrams(tokens, ngram_n)))
    counts = {k: v for k, v in counts.items() if v > min_count}

    raw = {g: 1.0 / math.log(occ) for g, occ in counts.items()}
    if raw:
        shift = 1.0 - float(np.mean(list(raw.values())))
        score_dict = {g: s + shift for g, s in raw.items()}
        return score_dict, max(score_dict.values())
    return {}, 1.0


def score_sentence(
    sentence: str, ngram_dict: Dict[NGram, float], max_score: float, ngram_n: int = 3
) -> float:
    """Mean n-gram rarity score of a sentence (unseen n-grams -> max_score)."""
    stop = _stopword_set()
    tokens = remove_stopwords_tokens(word_tokenize(sentence.lower()), stop)
    if len(tokens) < ngram_n:
        return 0.0
    scores = [ngram_dict.get(g, max_score) for g in nltk_ngrams(tokens, ngram_n)]
    return float(np.mean(scores))


def extractive_summary(
    text: str,
    ngram_dict: Dict[NGram, float],
    max_score: float,
    max_tokens: int,
    ngram_n: int = 3,
) -> str:
    """Greedily select the highest-scoring sentences up to a token budget,
    then re-order them to their original document positions."""
    sentences = sent_tokenize(text)
    if not sentences:
        return text

    scored = [
        (i, s, score_sentence(s, ngram_dict, max_score, ngram_n))
        for i, s in enumerate(sentences)
    ]
    selected_idx: set = set()
    token_count = 0
    for idx, sent, _ in sorted(scored, key=lambda x: x[2], reverse=True):
        n = len(word_tokenize(sent.lower()))
        if token_count + n <= max_tokens:
            selected_idx.add(idx)
            token_count += n
        if token_count >= max_tokens:
            break

    selected = [s for i, s, _ in scored if i in selected_idx]
    return " ".join(selected) if selected else sentences[0]


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class Vocabulary:
    """Word-level vocabulary with rarity-based frequency scores."""

    def __init__(self, max_vocab: int = 15000) -> None:
        self.max_vocab = max_vocab
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()
        self.freq_scores: Dict[str, float] = {}
        self.max_freq_sc: float = 3.0

    def build(self, texts: Iterable[str]) -> None:
        for t in texts:
            self.word_freq.update(word_tokenize(str(t).lower()))
        common = self.word_freq.most_common(self.max_vocab - len(SPECIAL))

        self.word2idx, self.idx2word = {}, {}
        for i, tok in enumerate(SPECIAL):
            self.word2idx[tok] = i
            self.idx2word[i] = tok
        for i, (w, _) in enumerate(common, len(SPECIAL)):
            self.word2idx[w] = i
            self.idx2word[i] = w

        raw = {w: 1.0 / math.log(o) for w, o in self.word_freq.items() if o > 1}
        if raw:
            shift = 1.0 - float(np.mean(list(raw.values())))
            self.freq_scores = {w: s + shift for w, s in raw.items()}
            self.max_freq_sc = max(self.freq_scores.values())

    def get_freq_score(self, word: str) -> float:
        return self.freq_scores.get(word, self.max_freq_sc)

    @property
    def pad_idx(self) -> int:
        return self.word2idx[PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.word2idx[UNK_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.word2idx[SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.word2idx[EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.word2idx)

    def save(self, path: str) -> None:
        payload = {
            "max_vocab": self.max_vocab,
            "word2idx": self.word2idx,
            "word_freq": dict(self.word_freq),
            "freq_scores": self.freq_scores,
            "max_freq_sc": self.max_freq_sc,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        v = cls(max_vocab=payload["max_vocab"])
        v.word2idx = {w: int(i) for w, i in payload["word2idx"].items()}
        v.idx2word = {int(i): w for w, i in v.word2idx.items()}
        v.word_freq = Counter(payload["word_freq"])
        v.freq_scores = payload["freq_scores"]
        v.max_freq_sc = payload["max_freq_sc"]
        return v


def build_vocabulary(texts: List[str], max_vocab: int) -> Vocabulary:
    v = Vocabulary(max_vocab=max_vocab)
    v.build(texts)
    return v


# --------------------------------------------------------------------------- #
# Dataset + collate
# --------------------------------------------------------------------------- #
class SummarizationDataset(Dataset):
    """Maps (source extract, reference summary) pairs to model tensors.

    Tokenisation is done once at construction time and cached. Each item yields
    ``src_ids`` (OOV -> <unk>), ``src_ext`` (OOV -> extended ids for copying),
    ``tgt_in``, ``tgt_out`` (extended vocab), ``src_freq`` and the OOV count.
    """

    def __init__(
        self,
        texts: List[str],
        summaries: List[str],
        vocab: Vocabulary,
        max_src: int = 512,
        max_tgt: int = 150,
    ) -> None:
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.src_toks = [word_tokenize(str(t).lower())[:max_src] for t in texts]
        self.tgt_toks = [word_tokenize(str(t).lower())[: max_tgt - 2] for t in summaries]

    def __len__(self) -> int:
        return len(self.src_toks)

    def __getitem__(self, idx: int):
        v = self.vocab
        src_toks = self.src_toks[idx]
        tgt_toks = self.tgt_toks[idx]

        oov_list: List[str] = []
        oov_map: dict = {}
        for t in src_toks:
            if t not in v.word2idx and t not in oov_map:
                oov_map[t] = self.vocab_size + len(oov_list)
                oov_list.append(t)

        src_ids = [v.word2idx.get(t, v.unk_idx) for t in src_toks]
        src_ext = [oov_map.get(t, v.word2idx.get(t, v.unk_idx)) for t in src_toks]
        src_freq = [v.get_freq_score(t) for t in src_toks]
        tgt_in = [v.sos_idx] + [v.word2idx.get(t, v.unk_idx) for t in tgt_toks]
        tgt_out = [oov_map.get(t, v.word2idx.get(t, v.unk_idx)) for t in tgt_toks] + [
            v.eos_idx
        ]

        pad_src = self.max_src - len(src_ids)
        src_ids += [v.pad_idx] * pad_src
        src_ext += [v.pad_idx] * pad_src
        src_freq += [0.0] * pad_src

        pad_tgt = self.max_tgt - len(tgt_in)
        tgt_in += [v.pad_idx] * pad_tgt
        tgt_out += [v.pad_idx] * (self.max_tgt - len(tgt_out))

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(src_ext, dtype=torch.long),
            torch.tensor(tgt_in, dtype=torch.long),
            torch.tensor(tgt_out, dtype=torch.long),
            torch.tensor(src_freq, dtype=torch.float),
            len(oov_list),
        )


def collate_fn(batch):
    """Stack a batch and report the batch-wide maximum OOV count."""
    src_ids, src_ext, tgt_in, tgt_out, src_freq, oov_lens = zip(*batch)
    return (
        torch.stack(src_ids),
        torch.stack(src_ext),
        torch.stack(tgt_in),
        torch.stack(tgt_out),
        torch.stack(src_freq),
        max(oov_lens),
    )
