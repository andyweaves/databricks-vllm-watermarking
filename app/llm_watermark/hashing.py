"""Keyed g-value hashing, implemented identically in numpy and torch.

A g-value is a pseudo-random bit g_l(context, token) for each tournament layer l.
Generation (torch, on the GPU) and detection (numpy, in the app) must produce
exactly the same bits, so both implementations use the same SplitMix64 mixing
function on 64-bit integers. numpy works on uint64; torch has no uint64
arithmetic, so it works on int64 with the same bit patterns (wrapping
two's-complement arithmetic and logical right shifts emulated with masks).
`tests/test_hashing.py` checks the two agree.
"""

from __future__ import annotations

import numpy as np

GOLDEN = 0x9E3779B97F4A7C15
MIX1 = 0xBF58476D1CE4E5B9
MIX2 = 0x94D049BB133111EB
CONTEXT_SEED = 0x6C6C6D776D2D7631  # "llmwm-v1"
PAD_TOKEN = -1  # used for context positions before the start of the sequence


# ---------------------------------------------------------------- numpy (detector)

def _mix_np(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = x + np.uint64(GOLDEN)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(MIX1)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(MIX2)
        return z ^ (z >> np.uint64(31))


def _to_u64(a) -> np.ndarray:
    return np.asarray(a, dtype=np.int64).astype(np.uint64)


def context_hash_np(contexts) -> np.ndarray:
    """contexts: [N, H] token ids, oldest first. Returns [N] uint64."""
    ctx = _to_u64(contexts)
    h = _mix_np(np.full(ctx.shape[0], CONTEXT_SEED, dtype=np.uint64))
    for j in range(ctx.shape[1]):
        h = _mix_np(h ^ ctx[:, j])
    return h


def g_values_np(context_hashes: np.ndarray, tokens, layer_seeds) -> np.ndarray:
    """context_hashes: [N] uint64, tokens: [N, K] ids. Returns [N, L, K] uint8 bits."""
    seeds = np.asarray(layer_seeds, dtype=np.uint64)
    tok = _to_u64(tokens)
    a = _mix_np(context_hashes[:, None] ^ seeds[None, :])          # [N, L]
    z = _mix_np(a[:, :, None] ^ tok[:, None, :])                   # [N, L, K]
    return (z >> np.uint64(63)).astype(np.uint8)


# ---------------------------------------------------------------- torch (generator)

def _signed(c: int) -> int:
    return c - (1 << 64) if c >= (1 << 63) else c


_GOLDEN_S, _MIX1_S, _MIX2_S, _CTX_S = (_signed(c) for c in (GOLDEN, MIX1, MIX2, CONTEXT_SEED))


def _lshr(z, s: int):
    """Logical right shift for int64 tensors (torch's >> is arithmetic)."""
    return (z >> s) & ((1 << (64 - s)) - 1)


def _mix_torch(z):
    z = z + _GOLDEN_S
    z = (z ^ _lshr(z, 30)) * _MIX1_S
    z = (z ^ _lshr(z, 27)) * _MIX2_S
    return z ^ _lshr(z, 31)


def signed_seeds(layer_seeds) -> list[int]:
    return [_signed(int(s)) for s in layer_seeds]


def context_hash_torch(contexts):
    """contexts: [B, H] int64 tensor, oldest first. Returns [B] int64."""
    import torch

    h = _mix_torch(torch.full((contexts.shape[0],), _CTX_S, dtype=torch.int64, device=contexts.device))
    for j in range(contexts.shape[1]):
        h = _mix_torch(h ^ contexts[:, j])
    return h


def g_values_torch(context_hashes, tokens, seeds):
    """context_hashes: [B] int64, tokens: [B, K] int64, seeds: [L] int64 tensor (signed).

    Returns [B, L, K] int64 tensor of bits.
    """
    a = _mix_torch(context_hashes[:, None] ^ seeds[None, :])       # [B, L]
    z = _mix_torch(a[:, :, None] ^ tokens[:, None, :])             # [B, L, K]
    return _lshr(z, 63)
