"""Test fixtures, including a minimal stand-in for vLLM's processor interface.

vLLM needs CUDA and does not install on most laptops, so the processor tests
import llm_watermark.vllm_processor against a stub of the two names it uses
from vllm 0.33 (`vllm.v1.worker.gpu.sample.logits_processor.LogitsProcessor`).
"""

import sys
import types
from abc import ABC, abstractmethod

import pytest

from llm_watermark import WatermarkConfig, WatermarkKey


def _install_vllm_stub() -> None:
    try:
        import vllm.v1.worker.gpu.sample.logits_processor  # noqa: F401  (real vLLM available)
        return
    except Exception:
        pass

    class LogitsProcessor(ABC):
        def __init__(self, vllm_config, req_states):  # noqa: B027
            pass

        @classmethod
        def validate_params(cls, sampling_params):  # noqa: B027
            pass

        def add_request(self, req_idx, sampling_params):
            return True

        def apply_staged_writes(self):  # noqa: B027
            pass

        @abstractmethod
        def apply(self, logits, ctx):
            raise NotImplementedError

    names = ["vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu", "vllm.v1.worker.gpu.sample",
             "vllm.v1.worker.gpu.sample.logits_processor"]
    for name in names:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["vllm.v1.worker.gpu.sample.logits_processor"].LogitsProcessor = LogitsProcessor


_install_vllm_stub()

TEST_SECRET = "test-key-0123456789abcdef0123456789abcdef"


@pytest.fixture
def key():
    return WatermarkKey(TEST_SECRET)


@pytest.fixture
def config():
    return WatermarkConfig()
