"""Model: attention, layers, pointer-generator and the full Transformer.

A from-scratch encoder-decoder with the M2 frequency-aware encoder attention and
an optional pointer-generator output layer, selected by ``use_pointer``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
class MultiHeadAttention(nn.Module):
    """Standard multi-head scaled dot-product attention (returns weights too)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None):
        B = q.size(0)
        Q, K, V = self._split(self.W_q(q)), self._split(self.W_k(k)), self._split(self.W_v(v))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out), attn


class FrequencyAwareAttention(nn.Module):
    """Encoder self-attention with M2 frequency boosting (paper Eq. 4).

    A learned scalar maps each token's frequency-rarity score to a positive
    multiplier (via ``softplus``) on its attention logits, initialised to ~1 so
    the layer starts close to vanilla attention.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_freq = nn.Linear(1, 1, bias=False)
        nn.init.ones_(self.W_freq.weight)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None, freq=None):
        B = q.size(0)
        Q, K, V = self._split(self.W_q(q)), self._split(self.W_k(k)), self._split(self.W_v(v))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if freq is not None:
            fs = F.softplus(
                self.W_freq(freq.unsqueeze(1).unsqueeze(2).unsqueeze(-1))
            ).squeeze(-1)
            scores = scores * fs
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with dropout."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """Frequency-aware self-attention + FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = FrequencyAwareAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, freq=None):
        x = self.norm1(x + self.dropout(self.attn(x, x, x, mask, freq)))
        return self.norm2(x + self.dropout(self.ff(x)))


class DecoderLayer(nn.Module):
    """Masked self-attention, cross-attention, FFN (returns cross weights)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc, src_mask, tgt_mask):
        self_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_out))
        cross_out, cross_weights = self.cross_attn(x, enc, enc, src_mask)
        x = self.norm2(x + self.dropout(cross_out))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x, cross_weights


class Encoder(nn.Module):
    """Embedding + positional encoding + stack of encoder layers."""

    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, pad_idx, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pe = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask, freq=None):
        x = self.pe(self.emb(src))
        for layer in self.layers:
            x = layer(x, mask, freq)
        return self.norm(x)


class Decoder(nn.Module):
    """Embedding + positional encoding + stack of decoder layers.

    Returns final hidden states and the last layer's cross-attention weights.
    """

    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, pad_idx, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pe = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc, src_mask, tgt_mask):
        x = self.pe(self.emb(tgt))
        cross = None
        for layer in self.layers:
            x, cross = layer(x, enc, src_mask, tgt_mask)
        return self.norm(x), cross


# --------------------------------------------------------------------------- #
# Pointer-generator
# --------------------------------------------------------------------------- #
class PointerGeneratorLayer(nn.Module):
    """Mixes generation and copying into one extended-vocab distribution.

    The copy distribution starts at zeros and is filled purely by scatter-adding
    attention onto source positions; it is NOT seeded with the vocabulary
    distribution (seeding would make the final distribution sum to ``2 - p_gen``
    instead of 1 and corrupt the loss).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.W_h = nn.Linear(d_model, 1, bias=False)
        self.W_s = nn.Linear(d_model, 1, bias=False)
        self.W_x = nn.Linear(d_model, 1, bias=False)
        self.b_ptr = nn.Parameter(torch.zeros(1))

    def forward(self, context_vec, dec_state, dec_input, vocab_dist, attn_dist, src_ext_ids, max_oov):
        p_gen = torch.sigmoid(
            self.W_h(context_vec) + self.W_s(dec_state) + self.W_x(dec_input) + self.b_ptr
        )  # [B, T, 1]

        B, T, V = vocab_dist.shape
        ext_vocab_size = V + max_oov

        ext_dist = torch.zeros(B, T, ext_vocab_size, device=vocab_dist.device)
        src_ext = src_ext_ids.unsqueeze(1).expand(B, T, -1)
        ext_dist.scatter_add_(2, src_ext, attn_dist)

        vocab_ext = torch.zeros(B, T, ext_vocab_size, device=vocab_dist.device)
        vocab_ext[:, :, :V] = vocab_dist

        final_dist = p_gen * vocab_ext + (1.0 - p_gen) * ext_dist
        return final_dist, p_gen


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class SummarizationTransformer(nn.Module):
    """Transformer with frequency-aware encoder and optional pointer-generator."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 1024,
        max_src_len: int = 512,
        max_tgt_len: int = 150,
        pad_idx: int = 0,
        dropout: float = 0.1,
        use_pointer: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, n_layers, n_heads, d_ff, max_src_len, pad_idx, dropout)
        self.decoder = Decoder(vocab_size, d_model, n_layers, n_heads, d_ff, max_tgt_len, pad_idx, dropout)
        self.vocab_proj = nn.Linear(d_model, vocab_size)
        self.pointer_gen = PointerGeneratorLayer(d_model)
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.use_pointer = use_pointer

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        B, L = tgt.shape
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        sub_mask = torch.tril(torch.ones(L, L, device=tgt.device)).bool()
        return pad_mask & sub_mask.unsqueeze(0).unsqueeze(0)

    def forward(self, src, src_ext_ids, tgt, freq_scores, max_oov):
        """Returns ``[B, T, V]`` when the pointer is off, or
        ``[B, T, V + max_oov]`` when it is on."""
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_out = self.encoder(src, src_mask, freq_scores)
        dec_out, cross = self.decoder(tgt, enc_out, src_mask, tgt_mask)

        vocab_logits = self.vocab_proj(dec_out)
        vocab_logits[:, :, self.pad_idx] = -1e9  # never predict <pad>
        vocab_dist = torch.softmax(vocab_logits, dim=-1)

        if not self.use_pointer:
            return vocab_dist  # [B, T, V]

        attn_dist = cross.mean(dim=1)  # [B, T, src_len]
        # Renormalise: dropout on cross-attention makes row sums drift from 1.
        attn_dist = attn_dist / attn_dist.sum(-1, keepdim=True).clamp(min=1e-9)

        context_vec = torch.bmm(attn_dist, enc_out)
        dec_emb = self.decoder.emb(tgt)
        final_dist, _ = self.pointer_gen(
            context_vec, dec_out, dec_emb, vocab_dist, attn_dist, src_ext_ids, max_oov
        )
        return final_dist  # [B, T, V + max_oov]


def build_model(vocab_size: int, model_cfg, pad_idx: int) -> SummarizationTransformer:
    """Construct the model from a ``ModelConfig``-like object."""
    return SummarizationTransformer(
        vocab_size=vocab_size,
        d_model=model_cfg.d_model,
        n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads,
        d_ff=model_cfg.d_ff,
        max_src_len=model_cfg.max_src,
        max_tgt_len=model_cfg.max_tgt,
        pad_idx=pad_idx,
        dropout=model_cfg.dropout,
        use_pointer=model_cfg.use_pointer,
    )
