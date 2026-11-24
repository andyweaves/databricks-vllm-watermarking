"""Watermark detection service: REST API plus a small static UI.

Configuration comes from environment variables set by the bundle
(databricks.yml, resources.apps.watermark_detector.config.env):

    WATERMARK_KEY        secret key (from the app's secret resource)
    WATERMARK_MODE       vllm_dual_key_gumbel (default) | vllm_gumbel | llmwm (must match the endpoint)
    WATERMARK_*          scheme parameters, identical to the serving endpoint
    TOKENIZER_PATH       /Volumes/.../tokenizer.json written by notebook 01
    SERVING_ENDPOINT     the watermarked vLLM endpoint, for the "Generate" tab
    MODEL_KEY            model key (qwen3_8_27b or gpt_oss_120b), shown in the UI
    ALLOW_CONTROL        "true" to offer an unwatermarked control generation
                         (the endpoint must also set WATERMARK_ALLOW_OPT_OUT=true)

An optional second model uses the same names with a _2 suffix (MODEL_KEY_2, WATERMARK_MODE_2,
TOKENIZER_PATH_2, SERVING_ENDPOINT_2); MODEL_KEY_2=none (the bundle default) turns it off. Each request picks a
model with "target" (default: the first).

Submitted text is never logged or stored.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from llm_watermark import SCHEME, WatermarkConfig, WatermarkKey
from llm_watermark.models import chat_params
from llm_watermark.modes import check_mode, detect as detect_mode, opt_out_fields

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("watermark-app")

MAX_CHARS = 60_000
STATIC = Path(__file__).parent / "static"
CONFIG = WatermarkConfig.from_env()
ALLOW_CONTROL = os.environ.get("ALLOW_CONTROL", "false").lower() == "true"


def _targets() -> dict[str, dict]:
    """Models the app can check, keyed by model key, in configuration order."""
    out: dict[str, dict] = {}
    for suffix in ("", "_2"):
        model = os.environ.get(f"MODEL_KEY{suffix}", "")
        tokenizer_path = os.environ.get(f"TOKENIZER_PATH{suffix}", "")
        if suffix and model in ("", "none"):
            continue
        out[model] = {"model": model, "mode": check_mode(os.environ.get(f"WATERMARK_MODE{suffix}", "vllm_dual_key_gumbel")),
                      "endpoint": os.environ.get(f"SERVING_ENDPOINT{suffix}", ""), "tokenizer_path": tokenizer_path}
    return out


TARGETS = _targets()
DEFAULT_TARGET = next(iter(TARGETS))


def target(name: str | None) -> dict:
    if not name:
        return TARGETS[DEFAULT_TARGET]
    if name not in TARGETS:
        raise HTTPException(400, f"unknown target {name!r}; choose one of {list(TARGETS)}")
    return TARGETS[name]

app = FastAPI(title="LLM watermark detector", version="0.1.0",
              description="Detects the llm_watermark text watermark. Responses are signed and timestamped.")


# ------------------------------------------------------------------ dependencies

@lru_cache(maxsize=1)
def watermark_key() -> WatermarkKey:
    return WatermarkKey.from_env()


@lru_cache(maxsize=1)
def workspace():
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


@lru_cache(maxsize=4)
def tokenizer(path: str):
    from tokenizers import Tokenizer

    if not path:
        raise RuntimeError("TOKENIZER_PATH is not set")
    if Path(path).exists():                       # local development
        return Tokenizer.from_file(path)
    data = workspace().files.download(path).contents.read()
    log.info("loaded tokenizer from %s (%d bytes)", path, len(data))
    return Tokenizer.from_str(data.decode("utf-8"))


def _signing_key() -> bytes:
    return hmac.new(watermark_key().secret.encode(), f"{SCHEME}/response-signing".encode(), hashlib.sha256).digest()


def _sign(payload: dict) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hmac.new(_signing_key(), body, hashlib.sha256).hexdigest()


# ------------------------------------------------------------------ detection

def analyse(text: str, include_tokens: bool, tgt: dict) -> dict:
    if len(text) > MAX_CHARS:
        raise HTTPException(413, f"text is longer than {MAX_CHARS} characters")
    enc = tokenizer(tgt["tokenizer_path"]).encode(text, add_special_tokens=False)
    result = detect_mode(enc.ids, watermark_key(), tgt["mode"], CONFIG)
    payload = result.to_dict()
    payload.update({
        "model": tgt["model"],
        "mode": tgt["mode"],
        "config": {"context_width": CONFIG.context_width, "num_layers": CONFIG.num_layers, "top_k": CONFIG.top_k},
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    })
    payload["signature"] = {"alg": "HMAC-SHA256", "value": _sign(payload)}
    if include_tokens:
        payload["segments"] = _segments(text, enc.offsets, result.token_scores)
    return payload


def _segments(text: str, offsets, scores) -> list[dict]:
    """Split the text into spans, one per token, with that token's g-score (or null)."""
    segs, cursor = [], 0
    for (start, end), score in zip(offsets, scores):
        if start > cursor:
            segs.append({"t": text[cursor:start], "s": None})
        if end > start:
            segs.append({"t": text[start:end], "s": None if score is None else round(score, 3)})
        cursor = max(cursor, end)
    if cursor < len(text):
        segs.append({"t": text[cursor:], "s": None})
    return segs


# ------------------------------------------------------------------ API models

class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1)
    include_tokens: bool = False
    target: str | None = None


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    max_tokens: int = Field(400, ge=16, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    control: bool = False
    target: str | None = None


class RobustnessRequest(BaseModel):
    text: str = Field(..., min_length=1)
    truncate_tokens: int = Field(200, ge=10, le=4000)
    delete_fraction: float = Field(0.2, ge=0.0, le=0.9)
    seed: int = 0
    target: str | None = None


# ------------------------------------------------------------------ routes

@app.get("/api/v1/info")
def info():
    try:
        key_id = watermark_key().key_id
    except Exception as e:  # noqa: BLE001
        key_id = f"unavailable: {e}"
    first = TARGETS[DEFAULT_TARGET]
    return {"scheme": SCHEME, "mode": first["mode"], "model": first["model"], "key_id": key_id,
            "serving_endpoint": first["endpoint"],
            "targets": [{k: t[k] for k in ("model", "mode", "endpoint")} for t in TARGETS.values()],
            "allow_control": ALLOW_CONTROL, "config": CONFIG.__dict__, "min_tokens": 200, "alpha": 1e-3}


@app.post("/api/v1/detect")
def detect(req: DetectRequest):
    return analyse(req.text, req.include_tokens, target(req.target))


def _chat(tgt: dict, prompt: str, max_tokens: int, temperature: float, watermark: bool) -> tuple[str, float]:
    client = workspace().serving_endpoints.get_open_ai_client()
    max_tokens, extra = chat_params(tgt["model"], max_tokens)
    if not watermark:
        extra.update(opt_out_fields(tgt["mode"]))
    t0 = time.time()
    resp = client.chat.completions.create(
        model=tgt["endpoint"], messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=temperature, extra_body=extra or None,
    )
    return resp.choices[0].message.content or "", time.time() - t0


@app.post("/api/v1/generate")
async def generate(req: GenerateRequest):
    tgt = target(req.target)
    if not tgt["endpoint"]:
        raise HTTPException(400, "no serving endpoint is configured for this model")
    if req.control and not ALLOW_CONTROL:
        raise HTTPException(400, "control generations are disabled (ALLOW_CONTROL=false)")
    jobs = [asyncio.to_thread(_chat, tgt, req.prompt, req.max_tokens, req.temperature, True)]
    if req.control:
        jobs.append(asyncio.to_thread(_chat, tgt, req.prompt, req.max_tokens, req.temperature, False))
    try:
        outputs = await asyncio.gather(*jobs)
    except Exception as e:  # noqa: BLE001
        log.warning("serving endpoint call failed: %s", type(e).__name__)
        raise HTTPException(502, f"serving endpoint call failed: {e}") from e
    labels = ["watermarked", "control"]
    return {name: {"text": text, "latency_s": round(latency, 2), "detection": analyse(text, True, tgt)}
            for name, (text, latency) in zip(labels, outputs)}


@app.post("/api/v1/robustness")
def robustness(req: RobustnessRequest):
    tgt = target(req.target)
    rng = random.Random(req.seed)
    tok = tokenizer(tgt["tokenizer_path"])
    enc = tok.encode(req.text, add_special_tokens=False)
    truncated = tok.decode(enc.ids[: req.truncate_tokens])
    words = req.text.split(" ")
    deleted = " ".join(w for w in words if rng.random() >= req.delete_fraction)
    sentences = re.split(r"(?<=[.!?])\s+", req.text.strip())
    shuffled = sentences[:]
    rng.shuffle(shuffled)
    variants = {
        "original": req.text,
        f"truncated to {req.truncate_tokens} tokens": truncated,
        f"{int(req.delete_fraction * 100)}% of words deleted": deleted,
        "sentences shuffled": " ".join(shuffled),
    }
    return {name: {"text": text, "detection": analyse(text, False, tgt)} for name, text in variants.items()}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.api_route("/", methods=["GET", "HEAD"])
def index():
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("DATABRICKS_APP_PORT", "8000")))
