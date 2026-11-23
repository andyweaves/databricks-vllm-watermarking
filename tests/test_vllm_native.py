"""The numpy port of vLLM's built-in watermark detector must match vLLM exactly."""

import json

import numpy as np
import pytest
import torch

from llm_watermark import WatermarkKey
from llm_watermark.launch_native import build_config
from llm_watermark.vllm_native import detect_native, native_key, philox_uniform

from . import _vllm_reference as ref
from .conftest import TEST_SECRET


def _ref_uniform(key, contexts, tokens):
    c = torch.tensor(contexts, dtype=torch.int64)
    t = torch.tensor(tokens, dtype=torch.int64).unsqueeze(-1)
    return ref._philox_uniform(key & 0xFFFFFFFF, key >> 32, c, t).squeeze(-1).numpy()


@pytest.mark.parametrize("width", [1, 4, 5, 8])
def test_philox_matches_vllm_bit_for_bit(width):
    rng = np.random.default_rng(width)
    contexts = rng.integers(0, 300_000, size=(500, width))
    contexts[:20, : min(width, 3)] = -1                         # detector padding
    tokens = rng.integers(0, 300_000, size=500)
    for key in (0, 42, 2**64 - 1, native_key(WatermarkKey(TEST_SECRET))):
        ours = philox_uniform(key, contexts, tokens)
        theirs = _ref_uniform(key, contexts, tokens)
        assert ours.dtype == np.float32
        assert np.array_equal(ours, theirs)


def test_uniforms_are_uniform():
    rng = np.random.default_rng(0)
    u = philox_uniform(123, rng.integers(0, 50_000, size=(20_000, 4)), rng.integers(0, 50_000, size=20_000))
    assert 0 < u.min() and u.max() < 1
    assert abs(u.mean() - 0.5) < 0.01


def _gumbel_generate(key_int, length, vocab=3000, seed=0, temperature=1.0):
    """Gumbel-max sampling as vLLM does it: argmax(logits / T - log(-log(u)))."""
    gen = torch.Generator().manual_seed(seed)
    tokens = []
    for _ in range(length):
        logits = torch.randn(vocab, generator=gen) * 3.0 / temperature
        ctx = ([-1] * 4 + tokens)[-4:]
        u = _ref_uniform(key_int, [ctx] * vocab, list(range(vocab)))
        u = np.maximum(u, np.finfo(np.float32).tiny)
        noise = -np.log(-np.log(u.astype(np.float64)))
        tokens.append(int(np.argmax(logits.numpy() + noise)))
    return tokens


def test_detection_matches_vllm_reference():
    key = WatermarkKey(TEST_SECRET)
    tokens = _gumbel_generate(native_key(key), 250)
    ours = detect_native(tokens, key, "gumbel")
    score, p_ref, n_ref = ref.reference_detect(native_key(key), tokens)
    assert ours.num_scored == n_ref
    assert ours.p_value == pytest.approx(p_ref, rel=1e-9, abs=1e-300)
    assert ours.verdict == "watermarked"


def test_unwatermarked_text_is_not_detected():
    key = WatermarkKey(TEST_SECRET)
    rng = np.random.default_rng(5)
    flagged = [detect_native(rng.integers(0, 50_000, 300), key, "gumbel").verdict == "watermarked" for _ in range(200)]
    assert np.mean(flagged) < 0.02                   # alpha = 1e-3


def test_wrong_key_and_other_scheme_do_not_detect():
    key = WatermarkKey(TEST_SECRET)
    tokens = _gumbel_generate(native_key(key), 250, seed=1)
    assert detect_native(tokens, WatermarkKey("x" * 32), "gumbel").verdict != "watermarked"
    from llm_watermark import detect_tokens

    assert detect_tokens(tokens, key).verdict != "watermarked"      # llmwm detector on vLLM-native text


def test_dual_key_detects_key_a_generation():
    from llm_watermark.vllm_native import derive_watermark_key

    key = WatermarkKey(TEST_SECRET)
    tokens = _gumbel_generate(derive_watermark_key(native_key(key), b"key_a"), 250, seed=2)
    assert detect_native(tokens, key, "dual_key_gumbel", dual_alpha=0.1).verdict == "watermarked"


def test_launcher_config(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEY", TEST_SECRET)
    monkeypatch.setenv("WATERMARK_VLLM_ALGORITHM", "dual_key_gumbel")
    cfg = build_config("0.34.0")
    assert cfg["enforce"] is True and cfg["algorithm"] == "dual_key_gumbel" and cfg["alpha"] == 0.1
    assert cfg["key"] == native_key(WatermarkKey(TEST_SECRET)) < 2**64
    assert "enforce" not in build_config("0.33.0")
    monkeypatch.setenv("WATERMARK_ALLOW_OPT_OUT", "true")
    assert build_config("0.34.1")["enforce"] is False
    json.dumps(cfg)


def test_modes_dispatch_and_opt_out_fields():
    from llm_watermark.modes import detect, launch_prefix, opt_out_fields

    key = WatermarkKey(TEST_SECRET)
    tokens = _gumbel_generate(native_key(key), 250, seed=3)
    assert detect(tokens, key, "vllm_gumbel").verdict == "watermarked"
    assert detect(tokens, key, "llmwm").verdict != "watermarked"
    assert opt_out_fields("llmwm") == {"vllm_xargs": {"watermark": False}}
    assert opt_out_fields("vllm_gumbel") == {"watermarking": False}
    assert "--logits-processors" in " ".join(launch_prefix("llmwm"))
    assert launch_prefix("vllm_dual_key_gumbel")[0] == "WATERMARK_VLLM_ALGORITHM=dual_key_gumbel"
    with pytest.raises(ValueError):
        detect(tokens, key, "nope")
