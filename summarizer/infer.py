"""Inference: beam-search decoding and ROUGE evaluation.

Beam search tracks two id streams: true ids (including extended-OOV ids, used to
build the output string) and embed-safe ids (OOV mapped to ``<unk>``, fed back
to the decoder). ROUGE evaluation decodes every test example and averages
ROUGE-1/2/L F-measures. No precomputed numbers are stored anywhere.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from tqdm import tqdm

from . import Config
from .data import UNK_TOKEN, Vocabulary


@torch.no_grad()
def beam_search_decode(
    model,
    src: torch.Tensor,
    src_ext: torch.Tensor,
    src_freq: torch.Tensor,
    vocab: Vocabulary,
    max_len: int,
    device: torch.device,
    vocab_size: int,
    max_oov: int = 0,
    beam_size: int = 4,
    oov_words: Optional[List[str]] = None,
    length_alpha: float = 0.7,
) -> str:
    """Decode a single example into a summary string (pointer ON or OFF)."""
    model.eval()
    oov_words = oov_words or []

    src = src.unsqueeze(0).to(device)
    src_ext = src_ext.unsqueeze(0).to(device)
    src_freq = src_freq.unsqueeze(0).to(device)
    src_mask = (src != vocab.pad_idx).unsqueeze(1).unsqueeze(2)
    enc_out = model.encoder(src, src_mask, src_freq)

    use_pointer = getattr(model, "use_pointer", True)

    beams = [(0.0, [vocab.sos_idx], [vocab.sos_idx])]
    completed = []

    def norm_score(item) -> float:
        return item[0] / (len(item[1]) ** length_alpha)

    for _ in range(max_len):
        candidates = []
        for log_prob, true_ids, feed_ids in beams:
            if true_ids[-1] == vocab.eos_idx:
                completed.append((log_prob, true_ids, feed_ids))
                continue

            tgt = torch.tensor([feed_ids], device=device)
            tgt_mask = model.make_tgt_mask(tgt)
            dec_out, cross = model.decoder(tgt, enc_out, src_mask, tgt_mask)

            attn_dist = cross.mean(1)
            attn_dist = attn_dist / attn_dist.sum(-1, keepdim=True).clamp(min=1e-9)
            vocab_logits = model.vocab_proj(dec_out[:, -1:])
            vocab_logits[:, :, vocab.pad_idx] = -1e9
            vocab_dist = torch.softmax(vocab_logits, dim=-1)

            if use_pointer:
                context_vec = torch.bmm(attn_dist[:, -1:], enc_out)
                dec_emb = model.decoder.emb(tgt[:, -1:])
                final_dist, _ = model.pointer_gen(
                    context_vec, dec_out[:, -1:], dec_emb, vocab_dist,
                    attn_dist[:, -1:], src_ext, max_oov,
                )
                log_probs = torch.log(final_dist.squeeze(1).clamp(min=1e-10))
            else:
                log_probs = torch.log(vocab_dist.squeeze(1).clamp(min=1e-10))

            topk_lp, topk_ids = log_probs[0].topk(beam_size)
            for lp, tid in zip(topk_lp.tolist(), topk_ids.tolist()):
                feed_id = tid if tid < vocab_size else vocab.unk_idx
                candidates.append((log_prob + lp, true_ids + [tid], feed_ids + [feed_id]))

        if not candidates:
            break
        beams = sorted(candidates, key=norm_score, reverse=True)[:beam_size]
        if len(completed) >= beam_size:
            break

    best = sorted(completed + beams, key=norm_score, reverse=True)[0]

    words: List[str] = []
    for tid in best[1][1:]:  # skip <sos>
        if tid == vocab.eos_idx:
            break
        if tid >= vocab_size:  # copied OOV -> restore source surface form
            oov_i = tid - vocab_size
            words.append(oov_words[oov_i] if oov_i < len(oov_words) else UNK_TOKEN)
        else:
            words.append(vocab.idx2word.get(tid, UNK_TOKEN))
    return " ".join(words)


def _encode_source(text: str, vocab: Vocabulary, max_src: int, vocab_size: int):
    """Build (src_ids, src_ext, src_freq, oov_words) for one document."""
    tokens = word_tokenize(str(text).lower())[:max_src]

    oov_words: List[str] = []
    oov_map: dict = {}
    for t in tokens:
        if t not in vocab.word2idx and t not in oov_map:
            oov_map[t] = vocab_size + len(oov_words)
            oov_words.append(t)

    src_ids = [vocab.word2idx.get(t, vocab.unk_idx) for t in tokens]
    src_ext = [oov_map.get(t, vocab.word2idx.get(t, vocab.unk_idx)) for t in tokens]
    src_freq = [vocab.get_freq_score(t) for t in tokens]

    pad = max_src - len(src_ids)
    src_ids += [vocab.pad_idx] * pad
    src_ext += [vocab.pad_idx] * pad
    src_freq += [0.0] * pad
    return src_ids, src_ext, src_freq, oov_words


def evaluate_rouge(model, texts, summaries, vocab: Vocabulary, config: Config, device) -> Dict[str, float]:
    """Beam-search decode the test set and average ROUGE-1/2/L F-measures."""
    rouge_types = list(config.eval.rouge_types)
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=config.eval.use_stemmer)
    totals = {rt: 0.0 for rt in rouge_types}

    vocab_size = len(vocab)
    n = len(texts)
    for i in tqdm(range(n), desc="Test ROUGE"):
        src_ids, src_ext, src_freq, oov_words = _encode_source(
            texts[i], vocab, config.model.max_src, vocab_size
        )
        pred = beam_search_decode(
            model,
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(src_ext, dtype=torch.long),
            torch.tensor(src_freq, dtype=torch.float),
            vocab,
            config.model.max_tgt,
            device,
            vocab_size=vocab_size,
            max_oov=len(oov_words),
            beam_size=config.inference.beam_size,
            oov_words=oov_words,
            length_alpha=config.inference.length_alpha,
        )
        scores = scorer.score(str(summaries[i]), pred)
        for rt in rouge_types:
            totals[rt] += scores[rt].fmeasure

    return {rt: totals[rt] / max(1, n) for rt in rouge_types}
