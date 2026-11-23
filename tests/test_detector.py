import numpy as np
import torch

from llm_watermark import WatermarkConfig, WatermarkKey, detect_tokens, detect_with_keys
from llm_watermark.hashing import PAD_TOKEN, signed_seeds
from llm_watermark.sampler import tournament, watermark_sample

VOCAB = 2000


def _generate(key, config, length, watermark=True, seed=0, temperature=1.0):
    """Autoregressive generation from a random 'language model' over a small vocabulary."""
    gen = torch.Generator().manual_seed(seed)
    seeds = torch.tensor(signed_seeds(key.layer_seeds(config.num_layers)), dtype=torch.int64)
    tokens = [int(t) for t in torch.randint(0, VOCAB, (config.context_width,), generator=gen)]
    for _ in range(length):
        logits = torch.randn(1, VOCAB, generator=gen) * 3.0
        if watermark:
            ctx = torch.tensor([tokens[-config.context_width:]], dtype=torch.long)
            nxt = watermark_sample(logits, ctx, torch.tensor([temperature]),
                                   torch.tensor([config.top_k]), seeds, generator=gen)
            tokens.append(int(nxt))
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            tokens.append(int(torch.multinomial(probs, 1, generator=gen)))
    return tokens


def test_unwatermarked_scores_follow_null(key, config):
    rng = np.random.default_rng(2)
    zs = [detect_tokens(rng.integers(0, 50_000, 400), key, config).z_score for _ in range(300)]
    assert abs(np.mean(zs)) < 0.2
    assert 0.8 < np.std(zs) < 1.2
    false_positives = np.mean([z > 3.09 for z in zs])      # alpha = 1e-3
    assert false_positives < 0.02


def test_watermarked_text_is_detected(key, config):
    tokens = _generate(key, config, 300, watermark=True)
    result = detect_tokens(tokens, key, config)
    assert result.verdict == "watermarked", result.to_dict()
    assert result.mean_g > 0.55


def test_plain_sampling_is_not_detected(key, config):
    result = detect_tokens(_generate(key, config, 300, watermark=False), key, config)
    assert result.verdict in ("not_detected", "too_short"), result.to_dict()


def test_wrong_key_does_not_detect(key, config):
    tokens = _generate(key, config, 300, watermark=True)
    other = WatermarkKey("another-key-0123456789abcdef0123456789")
    assert detect_tokens(tokens, other, config).verdict != "watermarked"


def test_short_text_verdict(key, config):
    result = detect_tokens(list(range(50)), key, config)
    assert result.verdict == "too_short"
    assert result.num_scored == 50 - config.context_width


def test_retired_keys(key, config):
    tokens = _generate(key, config, 300, watermark=True)
    retired = WatermarkKey("retired-key-0123456789abcdef01234567")
    result = detect_with_keys(tokens, [retired, key], config)
    assert result.verdict == "watermarked" and result.key_id == key.key_id


def test_tournament_preserves_distribution_on_average(config):
    """Averaged over keys, the tournament should not shift the model's distribution."""
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(1, 8, dtype=torch.float64) * 2, dim=-1)
    trials = 200_000                                   # one random key per trial
    g = torch.randint(0, 2, (trials, config.num_layers, 8))
    average = tournament(probs.expand(trials, 8), g).mean(dim=0)
    assert torch.allclose(average, probs[0], atol=0.005)   # ~5 standard errors


def test_tournament_output_is_a_distribution(config):
    torch.manual_seed(1)
    probs = torch.softmax(torch.randn(16, 40), dim=-1)
    g = torch.randint(0, 2, (16, config.num_layers, 40))
    q = tournament(probs, g)
    assert torch.all(q >= 0)
    assert torch.allclose(q.sum(-1), torch.ones(16), atol=1e-4)


def test_pad_tokens_are_valid_context(key, config):
    seeds = torch.tensor(signed_seeds(key.layer_seeds(config.num_layers)), dtype=torch.int64)
    ctx = torch.full((2, config.context_width), PAD_TOKEN, dtype=torch.long)
    out = watermark_sample(torch.randn(2, 100), ctx, torch.ones(2), torch.tensor([40, 2]), seeds)
    assert out.shape == (2,)
