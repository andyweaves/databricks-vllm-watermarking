"""Detection for vLLM's built-in watermark (vLLM >= 0.30, `--watermark-config`).

A numpy port of vLLM's Philox4x32-10 PRF and Gumbel-max detectors
(vllm/v1/watermarking/prfs/philox.py and gumbel.py, Apache-2.0), so the app
can detect the built-in watermark without installing vLLM or torch.
`tests/test_vllm_native.py` checks the port bit-for-bit against vLLM's own
torch implementation.

The vLLM key is derived from the same Databricks secret as llm_watermark's
key, so one secret drives either mode.
"""

from __future__ import annotations

import hashlib
import hmac
import math
from statistics import NormalDist

import numpy as np

from .config import SCHEME, WatermarkKey
from .detector import DEFAULT_ALPHA, DEFAULT_MIN_TOKENS, DetectionResult

ALGORITHMS = ("gumbel", "dual_key_gumbel")
_MASK = np.uint64(0xFFFFFFFF)
_M0, _M1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
_W0, _W1 = np.uint64(0x9E3779B9), np.uint64(0xBB67AE85)
_CONTEXT_DOMAIN, _TOKEN_DOMAIN = 0x574D4358, 0x574D544B


def native_key(key: WatermarkKey) -> int:
    """64-bit vLLM watermark key derived from the llm_watermark secret."""
    digest = hmac.new(key.secret.encode(), f"{SCHEME}/vllm-native".encode(), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big")


def derive_watermark_key(key: int, domain: bytes) -> int:
    """Same as vllm.config.watermarking.derive_watermark_key."""
    digest = hashlib.sha256(domain + key.to_bytes(8, "big")).digest()
    return int.from_bytes(digest[:8], "big")


def _mulhilo32(multiplier: np.uint64, values: np.ndarray):
    product = values * multiplier                     # < 2**64, no overflow
    return (product >> np.uint64(32)) & _MASK, product & _MASK


def _philox4x32_10(c0, c1, c2, c3, k0, k1):
    for round_index in range(10):
        h0, l0 = _mulhilo32(_M0, c0)
        h1, l1 = _mulhilo32(_M1, c2)
        c0, c1, c2, c3 = (h1 ^ c1 ^ k0) & _MASK, l1, (h0 ^ c3 ^ k1) & _MASK, l0
        if round_index != 9:
            k0 = (k0 + _W0) & _MASK
            k1 = (k1 + _W1) & _MASK
    return c0, c1, c2, c3


def philox_uniform(key: int, contexts: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
    """contexts: [N, W] token ids (-1 = padding), token_ids: [N]. Returns [N] float32 in (0, 1)."""
    with np.errstate(over="ignore"):
        ctx = (np.asarray(contexts, dtype=np.int64) & 0xFFFFFFFF).astype(np.uint64)
        n, width = ctx.shape
        k0 = np.full(n, key & 0xFFFFFFFF, dtype=np.uint64)
        k1 = np.full(n, key >> 32, dtype=np.uint64)
        state = (np.full(n, _CONTEXT_DOMAIN, dtype=np.uint64), np.full(n, width, dtype=np.uint64),
                 np.zeros(n, dtype=np.uint64), np.zeros(n, dtype=np.uint64))
        for offset in range(0, width, 4):
            block = [ctx[:, offset + i] if offset + i < width else np.full(n, 0xFFFFFFFF - i, dtype=np.uint64)
                     for i in range(4)]
            counter = tuple((state[i] ^ block[i]) & _MASK for i in range(4))
            state = _philox4x32_10(*counter, (k0 ^ np.uint64(offset)) & _MASK, k1)
        tokens = (np.asarray(token_ids, dtype=np.int64) & 0xFFFFFFFF).astype(np.uint64)
        outs = _philox4x32_10(tokens >> np.uint64(2), state[0], state[1], state[2],
                              (k0 ^ state[3]) & _MASK, (k1 ^ np.uint64(_TOKEN_DOMAIN)) & _MASK)
        word = tokens & np.uint64(3)
        out = outs[0].copy()
        for i in range(1, 4):
            out = np.where(word == np.uint64(i), outs[i], out)
        mantissas = (out & _MASK) >> np.uint64(8)
        return ((mantissas.astype(np.float64) + 1.0) / (2 ** 24 + 1)).astype(np.float32)


def _gamma_survival_integer_shape(score: float, shape: int) -> float:
    """Survival function of Gamma(shape, 1), as in vLLM."""
    if shape <= 0 or score <= 0:
        return 1.0
    log_score = math.log(score)
    log_term, max_log_term, log_terms = 0.0, 0.0, [0.0]
    for index in range(1, shape):
        log_term += log_score - math.log(index)
        log_terms.append(log_term)
        max_log_term = max(max_log_term, log_term)
    log_sum = max_log_term + math.log(sum(math.exp(t - max_log_term) for t in log_terms))
    return min(1.0, math.exp(min(0.0, -score + log_sum)))


def _gammaincc(shape: float, x: float) -> float:
    from scipy.special import gammaincc

    return float(gammaincc(shape, x))


def _prepare(token_ids, context_width: int, deduplicate: bool):
    """Mirror of vllm WatermarkDetector._prepare_inputs, keeping each target's position."""
    contexts, targets, positions, seen = [], [], [], set()
    prefix = [-1] * context_width
    for pos, token in enumerate(token_ids):
        context = tuple(prefix[-context_width:])
        if not deduplicate or context not in seen:
            seen.add(context)
            contexts.append(context)
            targets.append(token)
            positions.append(pos)
        prefix.append(token)
    return np.asarray(contexts, dtype=np.int64).reshape(-1, context_width), np.asarray(targets, dtype=np.int64), positions


def detect_native(token_ids, key: WatermarkKey, algorithm: str = "gumbel", context_width: int = 4,
                  dual_alpha: float = 0.1, alpha: float = DEFAULT_ALPHA,
                  min_tokens: int = DEFAULT_MIN_TOKENS) -> DetectionResult:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")
    ids = [int(t) for t in token_ids]
    contexts, targets, positions = _prepare(ids, context_width, deduplicate=True)
    n = len(targets)
    scheme = f"vllm-{algorithm.replace('_', '-')}"
    token_scores: list = [None] * len(ids)
    if n == 0:
        return DetectionResult("too_short", 1.0, 0.0, 0.5, len(ids), 0, key.key_id, scheme=scheme,
                               alpha=alpha, min_tokens=min_tokens, token_scores=token_scores)
    base = native_key(key)
    if algorithm == "gumbel":
        u = philox_uniform(base, contexts, targets)
        scores = -np.log1p(-u.astype(np.float64))
        p = _gamma_survival_integer_shape(float(scores.sum()), n)
        shown = u.astype(np.float64)
    else:
        ua = philox_uniform(derive_watermark_key(base, b"key_a"), contexts, targets)
        ub = philox_uniform(derive_watermark_key(base, b"key_b"), contexts, targets)
        scores = (1 - dual_alpha) * -np.log1p(-ua.astype(np.float64)) + dual_alpha * -np.log1p(-ub.astype(np.float64))
        variance = (1 - dual_alpha) ** 2 + dual_alpha ** 2
        total = float(scores.sum())
        if dual_alpha in (0, 0.5, 1):
            p = _gamma_survival_integer_shape(total / variance, round(n / variance))
        else:
            p = _gammaincc(n / variance, total / variance)
        shown = np.maximum(ua, ub).astype(np.float64)
    for pos, value in zip(positions, shown):
        token_scores[pos] = float(value)
    z = -NormalDist().inv_cdf(min(max(p, 1e-300), 1 - 1e-16))   # z-score with the same one-sided p-value
    if p < alpha:
        verdict = "watermarked"
    elif n < min_tokens:
        verdict = "too_short"
    else:
        verdict = "not_detected"
    return DetectionResult(verdict, p, z, float(shown.mean()), len(ids), n, key.key_id, scheme=scheme,
                           alpha=alpha, min_tokens=min_tokens, token_scores=token_scores)
