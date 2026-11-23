"""Generate with a real (tiny) model on CPU, re-tokenize the decoded text, and detect.

This is the closest local check to production: detection runs on the decoded
string, as the app does, so tokenization round-trips are included.
Run with: pytest -m slow
"""

import pytest

pytestmark = pytest.mark.slow

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPTS = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain to a new employee why data governance matters, in a friendly tone.",
    "Describe a walk through a city market on a summer morning.",
]


@pytest.fixture(scope="module")
def model_and_tokenizer():
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL)
    return model, tok


def _generate(model, tok, prompt, processor=None, max_new_tokens=300):
    import torch
    from transformers import LogitsProcessorList

    msgs = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    kwargs = dict(max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    torch.manual_seed(0)
    if processor is not None:
        out = model.generate(**inputs, do_sample=False, logits_processor=LogitsProcessorList([processor]), **kwargs)
    else:
        out = model.generate(**inputs, do_sample=True, temperature=0.7, top_k=40, **kwargs)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def test_detects_decoded_text(model_and_tokenizer, key, config):
    from llm_watermark import detect_tokens
    from llm_watermark.hf_processor import HFWatermarkLogitsProcessor

    model, tok = model_and_tokenizer
    proc = HFWatermarkLogitsProcessor(key, config, temperature=0.7)
    for prompt in PROMPTS:
        marked = _generate(model, tok, prompt, proc)
        plain = _generate(model, tok, prompt)
        r_marked = detect_tokens(tok.encode(marked, add_special_tokens=False), key, config)
        r_plain = detect_tokens(tok.encode(plain, add_special_tokens=False), key, config)
        print(f"\n{prompt[:40]}... marked z={r_marked.z_score:.1f} ({r_marked.num_scored} tok) "
              f"plain z={r_plain.z_score:.1f}")
        assert r_marked.verdict == "watermarked", r_marked.to_dict()
        assert r_plain.verdict != "watermarked", r_plain.to_dict()
