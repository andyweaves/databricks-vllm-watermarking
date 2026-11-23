"""Exercise WatermarkLogitsProcessor through the vLLM 0.33 Model Runner V2 call pattern."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_watermark import WatermarkConfig, detect_tokens

from .conftest import TEST_SECRET

VOCAB = 3000
MAX_REQS = 4
MAX_LEN = 1024


@dataclass
class FakeReqStates:
    device: torch.device
    max_num_reqs: int
    vocab_size: int
    all_token_ids: SimpleNamespace


def _processor(monkeypatch, **env):
    from llm_watermark.vllm_processor import WatermarkLogitsProcessor

    monkeypatch.setenv("WATERMARK_KEY", TEST_SECRET)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    states = FakeReqStates(torch.device("cpu"), MAX_REQS, VOCAB,
                           SimpleNamespace(gpu=torch.zeros(MAX_REQS, MAX_LEN, dtype=torch.int32)))
    return WatermarkLogitsProcessor(SimpleNamespace(speculative_config=None), states), states


def _params(temperature=0.8, top_k=-1, extra=None):
    return SimpleNamespace(temperature=temperature, top_k=top_k, extra_args=extra)


def _ctx(slots, positions, input_ids):
    slots = torch.tensor(slots)
    return SimpleNamespace(
        expanded_idx_mapping=slots, idx_mapping=slots, idx_mapping_np=slots.numpy(),
        expanded_local_pos=torch.zeros_like(slots), input_ids=torch.tensor(input_ids),
        pos=torch.tensor(positions), seq_lens_upper_bound_np=np.array(positions) + 1,
    )


def test_requires_key(monkeypatch):
    from llm_watermark.vllm_processor import WatermarkLogitsProcessor

    monkeypatch.delenv("WATERMARK_KEY", raising=False)
    states = FakeReqStates(torch.device("cpu"), 1, 10, SimpleNamespace(gpu=torch.zeros(1, 8, dtype=torch.int32)))
    with pytest.raises(RuntimeError, match="WATERMARK_KEY"):
        WatermarkLogitsProcessor(SimpleNamespace(speculative_config=None), states)


def test_rejects_speculative_decoding(monkeypatch):
    from llm_watermark.vllm_processor import WatermarkLogitsProcessor

    monkeypatch.setenv("WATERMARK_KEY", TEST_SECRET)
    states = FakeReqStates(torch.device("cpu"), 1, 10, SimpleNamespace(gpu=torch.zeros(1, 8, dtype=torch.int32)))
    with pytest.raises(ValueError, match="speculative"):
        WatermarkLogitsProcessor(SimpleNamespace(speculative_config=object()), states)


def test_forces_one_token_per_enabled_row(monkeypatch):
    proc, _ = _processor(monkeypatch)
    for slot in range(3):
        assert proc.add_request(slot, _params())
    proc.apply_staged_writes()
    logits = torch.randn(3, VOCAB)
    out = proc.apply(logits.clone(), _ctx([2, 0, 1], [10, 20, 30], [5, 6, 7]))
    finite = torch.isfinite(out)
    assert finite.sum(dim=1).tolist() == [1, 1, 1]
    # The forced token must come from the row's own top-k candidates.
    for row in range(3):
        chosen = int(out[row].argmax())
        assert chosen in torch.topk(logits[row], proc.config.top_k).indices.tolist()


def test_opt_out_only_when_allowed(monkeypatch):
    proc, _ = _processor(monkeypatch)
    assert proc.add_request(0, _params(extra={"watermark": False})) is True      # not allowed by default
    proc, _ = _processor(monkeypatch, WATERMARK_ALLOW_OPT_OUT="true")
    assert proc.add_request(0, _params(extra={"watermark": False})) is False
    assert proc.add_request(1, _params()) is True
    proc.apply_staged_writes()
    logits = torch.randn(2, VOCAB)
    out = proc.apply(logits.clone(), _ctx([0, 1], [3, 3], [1, 2]))
    assert torch.equal(out[0], logits[0])                 # opted-out row untouched
    assert torch.isfinite(out[1]).sum() == 1


def test_validate_params(monkeypatch):
    from llm_watermark.vllm_processor import WatermarkLogitsProcessor

    WatermarkLogitsProcessor.validate_params(_params(extra={"watermark": True}))
    with pytest.raises(ValueError):
        WatermarkLogitsProcessor.validate_params(_params(extra={"watermark": "no"}))


def test_greedy_requests_are_watermarked(monkeypatch):
    proc, _ = _processor(monkeypatch)
    proc.add_request(0, _params(temperature=0.0))
    assert proc.temperature_np[0] == pytest.approx(WatermarkConfig().greedy_temperature)


def test_generation_loop_is_detectable(monkeypatch, key):
    """Drive the processor like vLLM does, token by token, then detect the output."""
    proc, states = _processor(monkeypatch)
    proc.add_request(1, _params(temperature=1.0))
    proc.apply_staged_writes()
    gen = torch.Generator().manual_seed(3)
    prompt = torch.randint(0, VOCAB, (12,), generator=gen).tolist()
    seq = list(prompt)
    states.all_token_ids.gpu[1, : len(seq)] = torch.tensor(seq, dtype=torch.int32)
    for _ in range(300):
        pos = len(seq) - 1
        logits = torch.randn(1, VOCAB, generator=gen) * 3.0
        out = proc.apply(logits, _ctx([1], [pos], [seq[-1]]))
        nxt = int(out[0].argmax())                         # what any vLLM sampler would now pick
        seq.append(nxt)
        states.all_token_ids.gpu[1, len(seq) - 1] = nxt   # vLLM commits the sampled token
    result = detect_tokens(seq[len(prompt):], key, proc.config)
    assert result.verdict == "watermarked", result.to_dict()
