"""Per-model request settings shared by the notebooks and the app.

gpt-oss writes a reasoning section (returned separately as `reasoning_content`) before its answer, and both count
towards `max_tokens`. The extra budget keeps the visible answer about as long as a non-reasoning model's, which
matters because detection needs a couple of hundred tokens of answer text.
"""

from __future__ import annotations

MODEL_KEYS = ("qwen3_8_27b", "gpt_oss_120b")

_REQUEST_EXTRAS = {"gpt_oss_120b": {"reasoning_effort": "low"}}
_REASONING_BUDGET = {"gpt_oss_120b": 800}


def chat_params(model_key: str, max_tokens: int) -> tuple[int, dict]:
    """(max_tokens to send, extra request fields) for a requested answer length."""
    return max_tokens + _REASONING_BUDGET.get(model_key, 0), dict(_REQUEST_EXTRAS.get(model_key, {}))
