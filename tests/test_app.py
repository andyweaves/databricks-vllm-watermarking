"""API tests for the detection app, using a local tokenizer file and the HF adapter."""

import importlib
import sys
from pathlib import Path

import pytest

from .conftest import TEST_SECRET

pytestmark = pytest.mark.slow
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def client_and_text(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    from huggingface_hub import hf_hub_download

    tok_path = hf_hub_download(MODEL, "tokenizer.json")
    mp = pytest.MonkeyPatch()
    mp.setenv("WATERMARK_KEY", TEST_SECRET)
    mp.setenv("TOKENIZER_PATH", tok_path)
    mp.setenv("MODEL_KEY", "smollm2-135m")
    mp.setenv("WATERMARK_MODE", "llmwm")               # the fixture generates with the HF llmwm processor
    mp.setenv("MODEL_KEY_2", "smollm2-native")          # second target: same tokenizer, vLLM built-in watermark
    mp.setenv("TOKENIZER_PATH_2", tok_path)
    mp.setenv("WATERMARK_MODE_2", "vllm_gumbel")
    sys.path.insert(0, str(Path(__file__).parents[1] / "app"))
    app_module = importlib.import_module("app")

    from fastapi.testclient import TestClient

    from llm_watermark import WatermarkConfig, WatermarkKey
    from llm_watermark.hf_processor import HFWatermarkLogitsProcessor

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL)
    msgs = [{"role": "user", "content": "Write a short essay about rivers and the cities built on them."}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    proc = HFWatermarkLogitsProcessor(WatermarkKey(TEST_SECRET), WatermarkConfig())
    out = model.generate(**inputs, do_sample=False, max_new_tokens=300, min_new_tokens=300,
                         logits_processor=transformers.LogitsProcessorList([proc]), pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    yield TestClient(app_module.app), text
    mp.undo()


def test_info(client_and_text):
    client, _ = client_and_text
    r = client.get("/api/v1/info")
    assert r.status_code == 200 and r.json()["model"] == "smollm2-135m"


def test_detect_watermarked(client_and_text):
    client, text = client_and_text
    r = client.post("/api/v1/detect", json={"text": text, "include_tokens": True})
    d = r.json()
    assert r.status_code == 200, d
    assert d["verdict"] == "watermarked"
    assert d["signature"]["alg"] == "HMAC-SHA256" and len(d["signature"]["value"]) == 64
    assert "".join(s["t"] for s in d["segments"]) == text


def test_detect_human_text(client_and_text):
    client, _ = client_and_text
    human = ("The river had been there long before the town, and the town knew it. Every spring the water rose "
             "over the lower meadows and every autumn it fell back, leaving silt that the farmers cursed and then "
             "praised. ") * 6
    d = client.post("/api/v1/detect", json={"text": human}).json()
    assert d["verdict"] != "watermarked"


def test_robustness(client_and_text):
    client, text = client_and_text
    d = client.post("/api/v1/robustness", json={"text": text, "truncate_tokens": 120, "delete_fraction": 0.2}).json()
    assert d["original"]["detection"]["verdict"] == "watermarked"
    assert len(d) == 4


def test_index(client_and_text):
    client, _ = client_and_text
    r = client.get("/")
    assert r.status_code == 200 and "LLM watermark detector" in r.text


def test_second_target(client_and_text):
    client, text = client_and_text
    info = client.get("/api/v1/info").json()
    assert [t["model"] for t in info["targets"]] == ["smollm2-135m", "smollm2-native"]
    first = client.post("/api/v1/detect", json={"text": text}).json()
    second = client.post("/api/v1/detect", json={"text": text, "target": "smollm2-native"}).json()
    assert first["verdict"] == "watermarked" and first["mode"] == "llmwm"
    assert second["mode"] == "vllm_gumbel" and second["verdict"] != "watermarked"
    assert client.post("/api/v1/detect", json={"text": text, "target": "nope"}).status_code == 400
