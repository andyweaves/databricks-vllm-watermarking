import numpy as np
import torch

from llm_watermark.hashing import (
    context_hash_np,
    context_hash_torch,
    g_values_np,
    g_values_torch,
    signed_seeds,
)


def test_numpy_and_torch_agree(key, config):
    rng = np.random.default_rng(0)
    contexts = rng.integers(0, 250_000, size=(64, config.context_width), dtype=np.int64)
    contexts[:5, :2] = -1                      # padding before the start of a sequence
    tokens = rng.integers(0, 250_000, size=(64, config.top_k), dtype=np.int64)
    seeds = key.layer_seeds(config.num_layers)

    h_np = context_hash_np(contexts)
    h_t = context_hash_torch(torch.tensor(contexts))
    assert np.array_equal(h_np.astype(np.int64), h_t.numpy())

    g_np = g_values_np(h_np, tokens, seeds)
    g_t = g_values_torch(h_t, torch.tensor(tokens), torch.tensor(signed_seeds(seeds), dtype=torch.int64))
    assert g_np.shape == (64, config.num_layers, config.top_k)
    assert np.array_equal(g_np.astype(np.int64), g_t.numpy())


def test_g_values_look_like_fair_coins(key, config):
    rng = np.random.default_rng(1)
    contexts = rng.integers(0, 100_000, size=(2000, config.context_width))
    tokens = rng.integers(0, 100_000, size=(2000, 8))
    g = g_values_np(context_hash_np(contexts), tokens, key.layer_seeds(config.num_layers))
    assert abs(g.mean() - 0.5) < 0.01
    # Different layers should be uncorrelated.
    corr = np.corrcoef(g[:, 0, :].ravel(), g[:, 1, :].ravel())[0, 1]
    assert abs(corr) < 0.03


def test_different_keys_give_different_bits(config):
    from llm_watermark import WatermarkKey

    contexts = np.arange(40).reshape(10, 4)
    tokens = np.arange(100).reshape(10, 10)
    a = g_values_np(context_hash_np(contexts), tokens, WatermarkKey("a" * 32).layer_seeds(4))
    b = g_values_np(context_hash_np(contexts), tokens, WatermarkKey("b" * 32).layer_seeds(4))
    assert (a != b).mean() > 0.3
