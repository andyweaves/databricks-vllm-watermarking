"""Watermark modes shared by the notebooks and the app.

llmwm                 this package's SynthID-style processor (vllm_processor.py)
vllm_gumbel           vLLM's built-in Gumbel-max watermark (vLLM >= 0.30)
vllm_dual_key_gumbel  vLLM's built-in dual-key Gumbel-max watermark, which also
                      supports speculative decoding
"""

from __future__ import annotations

from .config import WatermarkConfig, WatermarkKey
from .detector import DetectionResult, detect_tokens

MODES = ("llmwm", "vllm_gumbel", "vllm_dual_key_gumbel")


def check_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"watermark mode must be one of {MODES}, not {mode!r}")
    return mode


def detect(token_ids, key: WatermarkKey, mode: str, config: WatermarkConfig | None = None) -> DetectionResult:
    check_mode(mode)
    config = config or WatermarkConfig.from_env()
    if mode == "llmwm":
        return detect_tokens(token_ids, key, config)
    from .vllm_native import detect_native

    return detect_native(token_ids, key, mode.removeprefix("vllm_"), context_width=config.context_width)


def opt_out_fields(mode: str) -> dict:
    """Request fields that ask the endpoint for an unwatermarked (control) generation."""
    check_mode(mode)
    return {"vllm_xargs": {"watermark": False}} if mode == "llmwm" else {"watermarking": False}


def launch_prefix(mode: str) -> list[str]:
    """Start of the endpoint entrypoint for each mode (vLLM flags follow)."""
    check_mode(mode)
    if mode == "llmwm":
        return ["python -u -m vllm.entrypoints.openai.api_server",
                "--logits-processors llm_watermark.vllm_processor:WatermarkLogitsProcessor"]
    return [f"WATERMARK_VLLM_ALGORITHM={mode.removeprefix('vllm_')}", "python -u -m llm_watermark.launch_native"]
