"""Watermark configuration and key handling.

The same configuration must be used for generation (the vLLM logits processor)
and detection (the app). Every value can be set through an environment variable
so that the serving endpoint and the app can be configured identically.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass, field

SCHEME = "llmwm-v1"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class WatermarkConfig:
    """Parameters shared by the generator and the detector.

    context_width: number of preceding tokens hashed with the key to seed each step.
    num_layers: tournament layers. More layers give a stronger signal per token.
    top_k: size of the candidate set the tournament chooses from.
    greedy_temperature: temperature used when a request asks for greedy decoding
        (temperature 0), which would otherwise leave no room for a watermark.
    allow_opt_out: honour a per-request {"watermark": false} flag. Demo use only.
    """

    context_width: int = 4
    num_layers: int = 30
    top_k: int = 40
    greedy_temperature: float = 0.7
    allow_opt_out: bool = False

    @classmethod
    def from_env(cls) -> "WatermarkConfig":
        return cls(
            context_width=_env_int("WATERMARK_CONTEXT_WIDTH", cls.context_width),
            num_layers=_env_int("WATERMARK_NUM_LAYERS", cls.num_layers),
            top_k=_env_int("WATERMARK_TOP_K", cls.top_k),
            greedy_temperature=_env_float("WATERMARK_GREEDY_TEMPERATURE", cls.greedy_temperature),
            allow_opt_out=_env_bool("WATERMARK_ALLOW_OPT_OUT", cls.allow_opt_out),
        )

    def __post_init__(self) -> None:
        if self.context_width < 1:
            raise ValueError("context_width must be at least 1")
        if not 1 <= self.num_layers <= 64:
            raise ValueError("num_layers must be between 1 and 64")
        if self.top_k < 2:
            raise ValueError("top_k must be at least 2, or the tournament has nothing to choose from")
        if self.greedy_temperature <= 0:
            raise ValueError("greedy_temperature must be positive")


@dataclass(frozen=True)
class WatermarkKey:
    """A secret key and the per-layer seeds derived from it.

    The seeds are derived with HMAC-SHA256, so the key itself never enters the
    hashing used at generation time.
    """

    secret: str = field(repr=False)
    key_id: str = ""

    def __post_init__(self) -> None:
        if not self.secret or len(self.secret) < 16:
            raise ValueError("watermark key must be at least 16 characters; use e.g. `openssl rand -hex 32`")
        if not self.key_id:
            fingerprint = hmac.new(self.secret.encode(), f"{SCHEME}/key-id".encode(), hashlib.sha256)
            object.__setattr__(self, "key_id", fingerprint.hexdigest()[:12])

    @classmethod
    def from_env(cls) -> "WatermarkKey":
        secret = os.environ.get("WATERMARK_KEY", "")
        if not secret:
            raise RuntimeError("WATERMARK_KEY is not set; refusing to run without a watermark key")
        return cls(secret=secret, key_id=os.environ.get("WATERMARK_KEY_ID", ""))

    def layer_seeds(self, num_layers: int) -> list[int]:
        """Unsigned 64-bit seed for each tournament layer."""
        seeds = []
        for layer in range(num_layers):
            digest = hmac.new(self.secret.encode(), f"{SCHEME}/layer/{layer}".encode(), hashlib.sha256).digest()
            seeds.append(int.from_bytes(digest[:8], "big"))
        return seeds
