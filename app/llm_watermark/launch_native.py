"""Start vLLM's OpenAI server with its built-in watermark, keyed from WATERMARK_KEY.

Used as the endpoint entrypoint in the vLLM-native modes:

    WATERMARK_VLLM_ALGORITHM=gumbel python -u -m llm_watermark.launch_native --model ... (other vLLM flags)

The vLLM key is derived from the Databricks secret at start-up, so it never
appears in the logged model, its entrypoint or the endpoint configuration.
(vLLM itself redacts `watermark_config` from its startup log.)

On vLLM >= 0.34 the launcher sets `enforce`, which overrides per-request
`watermarking: false`, unless WATERMARK_ALLOW_OPT_OUT=true. vLLM 0.30-0.33
cannot block opt-outs; the launcher logs a warning.
"""

from __future__ import annotations

import json
import os
import sys
from importlib.metadata import version

from .config import WatermarkConfig, WatermarkKey
from .vllm_native import ALGORITHMS, native_key


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split(".")[:2] if p.isdigit())


def build_config(vllm_version: str) -> dict:
    key = WatermarkKey.from_env()
    algorithm = os.environ.get("WATERMARK_VLLM_ALGORITHM", "gumbel")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"WATERMARK_VLLM_ALGORITHM must be one of {ALGORITHMS}")
    cfg: dict = {
        "algorithm": algorithm,
        "key": native_key(key),
        "context_width": int(os.environ.get("WATERMARK_CONTEXT_WIDTH", "4")),
        "deduplicate_contexts": "single_turn",
    }
    if algorithm == "dual_key_gumbel":
        cfg["alpha"] = float(os.environ.get("WATERMARK_VLLM_ALPHA", "0.1"))
    allow_opt_out = WatermarkConfig.from_env().allow_opt_out
    if _version_tuple(vllm_version) >= (0, 34):
        cfg["enforce"] = not allow_opt_out
    elif not allow_opt_out:
        print(f"llm_watermark: WARNING vLLM {vllm_version} cannot block per-request watermarking=false; "
              "upgrade to vLLM 0.34 or later, or block the field at a gateway", file=sys.stderr, flush=True)
    return cfg


def main() -> None:
    cfg = build_config(version("vllm"))
    print(f"llm_watermark: starting vLLM with built-in {cfg['algorithm']} watermark, key_id="
          f"{WatermarkKey.from_env().key_id}, enforce={cfg.get('enforce', False)}", file=sys.stderr, flush=True)
    argv = [sys.executable, "-u", "-m", "vllm.entrypoints.openai.api_server", *sys.argv[1:],
            "--watermark-config", json.dumps(cfg)]
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
