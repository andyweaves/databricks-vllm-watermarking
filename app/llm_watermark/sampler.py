"""SynthID-style tournament sampling (torch).

For each step the model's distribution is restricted to its top-k candidates,
then reweighted through `num_layers` tournament layers. With two candidates per
match, the winner's distribution after layer l has the closed form

    q_{l+1}(x) = q_l(x) * (1 + g_l(x) - E_{q_l}[g_l])

(Dathathri et al., Nature 2024). Tokens whose keyed g-values are 1 become more
likely; the expected distribution over keys is unchanged, so quality is
preserved in the "non-distortionary" sense. The final token is sampled from q.
"""

from __future__ import annotations

from .hashing import context_hash_torch, g_values_torch


def tournament(probs, g):
    """probs: [B, K] float, g: [B, L, K] bits. Returns the reweighted [B, K] distribution."""
    q = probs
    gf = g.to(probs.dtype)
    for layer in range(gf.shape[1]):
        gl = gf[:, layer, :]
        mean = (q * gl).sum(dim=-1, keepdim=True)
        q = q * (1.0 + gl - mean)
    return q.clamp_min(0.0)


def watermark_sample(logits, contexts, temperatures, top_ks, seeds, generator=None):
    """Choose one watermarked token per row.

    logits: [B, V] raw (unscaled) logits.
    contexts: [B, H] int64 preceding token ids, oldest first (PAD_TOKEN before the start).
    temperatures: [B] float, all > 0.
    top_ks: [B] int64, each between 2 and the configured top_k.
    seeds: [L] int64 tensor of signed per-layer seeds.
    Returns [B] int64 token ids.
    """
    import torch

    k_max = int(top_ks.max().item())
    scaled = logits.float() / temperatures.float()[:, None]
    values, indices = torch.topk(scaled, k_max, dim=-1)
    ranks = torch.arange(k_max, device=logits.device)[None, :]
    values = values.masked_fill(ranks >= top_ks[:, None], float("-inf"))
    probs = torch.softmax(values, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)

    g = g_values_torch(context_hash_torch(contexts.long()), indices.long(), seeds)
    q = tournament(probs, g)
    # Rows where every candidate was masked (should not happen) fall back to the top token.
    empty = q.sum(dim=-1, keepdim=True) <= 0
    top_only = torch.zeros_like(q)
    top_only[:, 0] = 1.0
    q = torch.where(empty, top_only, q)
    choice = torch.multinomial(q, 1, generator=generator)
    return indices.gather(1, choice).squeeze(1)


def force_tokens(logits, rows, tokens):
    """Make `tokens` the only possible choice for `rows`, whatever sampler runs next."""
    logits[rows] = float("-inf")
    logits[rows, tokens] = 0.0
    return logits
