"""Hugging Face Transformers adapter, for local tests and CPU demos.

Uses the same tournament sampler as the vLLM processor, so text generated
here is detected exactly like text from the serving endpoint (given the same
tokenizer). Use it with greedy decoding (do_sample=False): the processor does
its own temperature and top-k, then forces the chosen token.
"""

from __future__ import annotations

from .config import WatermarkConfig, WatermarkKey
from .hashing import PAD_TOKEN, signed_seeds


class HFWatermarkLogitsProcessor:
    def __init__(self, key: WatermarkKey, config: WatermarkConfig | None = None,
                 temperature: float = 0.7, top_k: int | None = None, generator=None):
        import torch

        self.config = config or WatermarkConfig()
        self.seeds = torch.tensor(signed_seeds(key.layer_seeds(self.config.num_layers)), dtype=torch.int64)
        self.temperature = temperature
        self.top_k = min(top_k or self.config.top_k, self.config.top_k)
        self.generator = generator

    def __call__(self, input_ids, scores):
        import torch

        from .sampler import force_tokens, watermark_sample

        h = self.config.context_width
        contexts = input_ids[:, -h:].long()
        if contexts.shape[1] < h:
            pad = torch.full((contexts.shape[0], h - contexts.shape[1]), PAD_TOKEN, dtype=torch.long)
            contexts = torch.cat([pad, contexts], dim=1)
        b = scores.shape[0]
        tokens = watermark_sample(
            scores, contexts,
            torch.full((b,), self.temperature), torch.full((b,), self.top_k, dtype=torch.long),
            self.seeds.to(scores.device), generator=self.generator,
        )
        scores = scores.clone()
        return force_tokens(scores, torch.arange(b), tokens)
