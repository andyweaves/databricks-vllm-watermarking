"""vLLM custom logits processor (Model Runner V2 interface, vLLM >= 0.33).

Load it with:

    VLLM_USE_V2_MODEL_RUNNER=1 vllm serve <model> \
        --logits-processors llm_watermark.vllm_processor:WatermarkLogitsProcessor

and set WATERMARK_KEY (plus the optional WATERMARK_* settings in config.py).

How it works: vLLM calls `apply()` after its own bias, penalty and grammar
stages and before temperature and top-k/top-p. For every watermarked row the
processor applies the request's temperature and top-k itself, runs the
tournament (sampler.py), samples the winning token and then forces it by
setting every other logit to -inf. Whatever vLLM's sampler does next
(temperature, top-p, greedy argmax) can only pick that token. With vLLM's
default `raw_logprobs` mode, returned logprobs still come from the unmodified
model distribution.

vLLM imports this module in its frontend process to validate request
parameters, so nothing heavy (torch) is imported at module level.

Limitations: speculative decoding is rejected (it uses one logits row per
draft token); top_p, min_p and per-request seeds are not applied to
watermarked rows.
"""

from __future__ import annotations

import logging

import numpy as np
from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor

from .config import WatermarkConfig, WatermarkKey
from .hashing import PAD_TOKEN, context_hash_np, context_hash_torch, signed_seeds

logger = logging.getLogger(__name__)


class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, req_states):
        import torch

        self.config = WatermarkConfig.from_env()
        self.key = WatermarkKey.from_env()   # fails closed if WATERMARK_KEY is missing

        if getattr(vllm_config, "speculative_config", None) is not None:
            raise ValueError("llm_watermark does not support speculative decoding")

        self.req_states = req_states
        self.device = req_states.device
        n = req_states.max_num_reqs
        self.seeds = torch.tensor(signed_seeds(self.key.layer_seeds(self.config.num_layers)),
                                  dtype=torch.int64, device=self.device)

        # Per-slot settings, written on the host in add_request() and staged to the device.
        self.enabled_np = np.zeros(n, dtype=bool)
        self.temperature_np = np.ones(n, dtype=np.float32)
        self.top_k_np = np.full(n, self.config.top_k, dtype=np.int64)
        self.temperature = torch.ones(n, dtype=torch.float32, device=self.device)
        self.top_k = torch.full((n,), self.config.top_k, dtype=torch.int64, device=self.device)
        self._dirty = False

        self._self_check()
        logger.info("llm_watermark enabled: key_id=%s context=%d layers=%d top_k=%d opt_out=%s",
                    self.key.key_id, self.config.context_width, self.config.num_layers,
                    self.config.top_k, self.config.allow_opt_out)

    def _self_check(self) -> None:
        """Refuse to start if device integer arithmetic differs from the detector's."""
        import torch

        sample = np.array([[0, 1, 2, 3], [151643, -1, 7, 200000], [5, 5, 5, 5]], dtype=np.int64)
        expected = context_hash_np(sample).astype(np.int64)
        actual = context_hash_torch(torch.tensor(sample, device=self.device)).cpu().numpy()
        if not np.array_equal(expected, actual):
            raise RuntimeError("llm_watermark self-check failed: device hashing does not match the detector")

    @classmethod
    def validate_params(cls, sampling_params) -> None:
        extra = getattr(sampling_params, "extra_args", None) or {}
        if "watermark" in extra and not isinstance(extra["watermark"], bool):
            raise ValueError("vllm_xargs.watermark must be true or false")

    def add_request(self, req_idx: int, sampling_params) -> bool:
        extra = getattr(sampling_params, "extra_args", None) or {}
        enabled = not (self.config.allow_opt_out and extra.get("watermark") is False)

        temperature = sampling_params.temperature
        if temperature is None or temperature <= 0:
            temperature = self.config.greedy_temperature   # greedy would leave no room for a mark
        top_k = sampling_params.top_k
        if top_k is None or top_k <= 0 or top_k > self.config.top_k:
            top_k = self.config.top_k
        top_k = max(int(top_k), 2)

        self.enabled_np[req_idx] = enabled
        self.temperature_np[req_idx] = temperature
        self.top_k_np[req_idx] = top_k
        self._dirty = True
        return enabled

    def apply_staged_writes(self) -> None:
        if self._dirty:
            import torch

            self.temperature.copy_(torch.from_numpy(self.temperature_np), non_blocking=True)
            self.top_k.copy_(torch.from_numpy(self.top_k_np), non_blocking=True)
            self._dirty = False

    def apply(self, logits, ctx):
        import torch

        from .sampler import force_tokens, watermark_sample

        # Cheap host-side check before touching the device.
        if not self.enabled_np[ctx.idx_mapping_np].any():
            return logits

        slots = ctx.expanded_idx_mapping.long()
        enabled = torch.from_numpy(self.enabled_np).to(self.device, non_blocking=True)[slots]
        rows = enabled.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return logits

        s = slots[rows]
        pos = ctx.pos[rows].long()                 # position of the token fed this step
        h = self.config.context_width

        # Context = the h tokens ending at `pos`, oldest first. The token at `pos` is ctx.input_ids;
        # earlier ones are committed in req_states.all_token_ids.
        if h > 1:
            offsets = torch.arange(h - 1, 0, -1, device=self.device)        # h-1 ... 1
            cpos = pos[:, None] - offsets[None, :]
            history = self.req_states.all_token_ids.gpu[s[:, None], cpos.clamp(min=0)].long()
            history = torch.where(cpos >= 0, history, torch.full_like(history, PAD_TOKEN))
            contexts = torch.cat([history, ctx.input_ids[rows].long()[:, None]], dim=1)
        else:
            contexts = ctx.input_ids[rows].long()[:, None]

        tokens = watermark_sample(logits[rows], contexts, self.temperature[s], self.top_k[s], self.seeds)
        return force_tokens(logits, rows, tokens)
