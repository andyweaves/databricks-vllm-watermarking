"""Watermark detection (numpy only, no GPU or model needed).

Detection recomputes the keyed g-values of each token from its preceding
context. Without the watermark each g-value is a fair coin, so the weighted
mean has a known null distribution and the test needs no trained detector:

    z = (S - N * 0.5 * sum(w)) / sqrt(N * 0.25 * sum(w^2))

where S is the weighted sum of g-values over N scored tokens. The p-value is
the probability of a score at least this high from unwatermarked text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import SCHEME, WatermarkConfig, WatermarkKey
from .hashing import context_hash_np, g_values_np

DEFAULT_ALPHA = 1e-3       # false-positive rate for a "watermarked" verdict
DEFAULT_MIN_TOKENS = 200   # below this, a negative result is "too short to assess" (Code of Practice)


def layer_weights(num_layers: int) -> np.ndarray:
    """Earlier layers carry more signal; weight them linearly from 10 down to 1."""
    return np.linspace(10.0, 1.0, num_layers) if num_layers > 1 else np.ones(1)


@dataclass
class DetectionResult:
    verdict: str                  # "watermarked" | "not_detected" | "too_short"
    p_value: float
    z_score: float
    mean_g: float                 # weighted mean g-value; 0.5 expected without a watermark
    num_tokens: int
    num_scored: int
    key_id: str
    scheme: str = SCHEME
    alpha: float = DEFAULT_ALPHA
    min_tokens: int = DEFAULT_MIN_TOKENS
    token_scores: list = field(default_factory=list)   # per-token weighted mean g, None if not scored

    def to_dict(self, include_tokens: bool = False) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "token_scores"}
        if include_tokens:
            d["token_scores"] = self.token_scores
        return d


def score_tokens(token_ids: Sequence[int], key: WatermarkKey, config: WatermarkConfig):
    """Return (per-token weighted g-score or None, weights) for a token sequence."""
    ids = np.asarray(list(token_ids), dtype=np.int64)
    h = config.context_width
    n = len(ids)
    scores: list = [None] * n
    if n <= h:
        return scores, layer_weights(config.num_layers)

    # Skip tokens whose (context, token) n-gram already appeared: repeats are not independent.
    seen: set = set()
    keep = []
    for i in range(h, n):
        ngram = tuple(ids[i - h : i + 1].tolist())
        if ngram not in seen:
            seen.add(ngram)
            keep.append(i)
    if not keep:
        return scores, layer_weights(config.num_layers)

    positions = np.asarray(keep)
    contexts = np.stack([ids[i - h : i] for i in keep])
    g = g_values_np(context_hash_np(contexts), ids[positions][:, None], key.layer_seeds(config.num_layers))[:, :, 0]
    w = layer_weights(config.num_layers)
    per_token = (g * w[None, :]).sum(axis=1) / w.sum()
    for pos, value in zip(keep, per_token):
        scores[pos] = float(value)
    return scores, w


def detect_tokens(
    token_ids: Sequence[int],
    key: WatermarkKey,
    config: WatermarkConfig | None = None,
    alpha: float = DEFAULT_ALPHA,
    min_tokens: int = DEFAULT_MIN_TOKENS,
) -> DetectionResult:
    config = config or WatermarkConfig()
    scores, w = score_tokens(token_ids, key, config)
    scored = [s for s in scores if s is not None]
    n = len(scored)
    if n == 0:
        return DetectionResult("too_short", 1.0, 0.0, 0.5, len(scores), 0, key.key_id,
                               alpha=alpha, min_tokens=min_tokens, token_scores=scores)
    w1, w2 = float(w.sum()), float((w ** 2).sum())
    total = sum(scored) * w1                      # weighted sum of g-values
    z = (total - n * 0.5 * w1) / math.sqrt(n * 0.25 * w2)
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    if p < alpha:
        verdict = "watermarked"
    elif n < min_tokens:
        verdict = "too_short"
    else:
        verdict = "not_detected"
    return DetectionResult(verdict, p, z, sum(scored) / n, len(scores), n, key.key_id,
                           alpha=alpha, min_tokens=min_tokens, token_scores=scores)


def detect_with_keys(token_ids: Sequence[int], keys: Sequence[WatermarkKey], config: WatermarkConfig | None = None,
                     alpha: float = DEFAULT_ALPHA, min_tokens: int = DEFAULT_MIN_TOKENS) -> DetectionResult:
    """Test several keys (for example current and retired ones), Bonferroni-corrected."""
    if not keys:
        raise ValueError("at least one key is required")
    results = [detect_tokens(token_ids, k, config, alpha / len(keys), min_tokens) for k in keys]
    best = min(results, key=lambda r: r.p_value)
    best.p_value = min(1.0, best.p_value * len(keys))
    best.alpha = alpha
    return best
