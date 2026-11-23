# Watermarked open-weight LLMs on Databricks, with a detection app

An end-to-end example of **generation-time text watermarking** for an open-weight LLM served with **vLLM** on
Databricks [Custom LLM Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms),
plus a **Databricks App** that detects the watermark through a REST API and a small UI.

It was built as a worked example for EU AI Act Article 50(2) discussions: open-weight models ship with no watermark,
so a provider building on them has to mark output in its own serving layer and offer a way to detect it.

> **Not production-ready and not legal advice.** This is a reference implementation of a published technique. A real
> deployment needs its own security review, detector calibration on its own outputs, robustness testing and
> documentation. See [Limitations](#limitations).

## What's in the repo

```
app/
  llm_watermark/          the watermark package (shared by the serving endpoint and the app)
    config.py             scheme parameters and key handling (HMAC-derived per-layer seeds)
    hashing.py            keyed g-values, bit-identical in numpy (detector) and torch (GPU)
    sampler.py            SynthID-style tournament sampling
    detector.py           analytic z-test, no trained detector needed
    vllm_processor.py     vLLM >= 0.33 custom logits processor (Model Runner V2 interface)
    hf_processor.py       Hugging Face Transformers adapter, for local tests
  app.py, static/         FastAPI detection service and UI (Databricks App)
notebooks/
  01_log_and_deploy.py    AI Runtime: download weights, test vLLM + watermark, log, register, deploy
  02_smoke_test.py        query the endpoint and check its replies
tests/                    CPU tests, including end-to-end generation with a 135M model
databricks.yml            bundle: schema, volume and the app
```

## Watermark modes

Notebook 01's `watermark_mode` widget and the bundle's `watermark_mode` variable must match.

| Mode | What runs on the endpoint | Opt-out by callers | Speculative decoding |
|---|---|---|---|
| `llmwm` (default) | This repo's SynthID-style logits processor | Only if the endpoint sets `WATERMARK_ALLOW_OPT_OUT=true` | Not supported |
| `vllm_gumbel` | vLLM's built-in Gumbel-max watermark (vLLM 0.30+), started by `llm_watermark.launch_native` | `watermarking: false` works on vLLM 0.33; on 0.34+ the launcher sets `enforce` | Not supported by this algorithm |
| `vllm_dual_key_gumbel` | vLLM's built-in dual-key Gumbel-max watermark | As above | Supported by vLLM (not enabled here) |

In both vLLM-native modes the launcher derives vLLM's 64-bit key from the same Databricks secret, so the key never
appears in the logged model. The app detects them with a numpy port of vLLM's Philox PRF and Gumbel detector,
tested bit-for-bit against vLLM's own code. vLLM's built-in watermark does not mark greedy (temperature 0) requests.

## How it works (llmwm mode)

**Generation.** vLLM calls the processor at every decoding step, after its own bias, penalty and grammar stages and
before temperature and top-k. For each request the processor:

1. applies the request's temperature and keeps the top-k candidates (default 40);
2. hashes the previous 4 tokens with the secret key to get a pseudo-random bit `g` for each candidate in each of 30
   tournament layers;
3. reweights the candidates layer by layer, `q ← q · (1 + g − E_q[g])`, the closed form of a two-candidate tournament
   ([Dathathri et al., Nature 2024](https://www.nature.com/articles/s41586-024-08025-4));
4. samples one token from the result and forces it by setting every other logit to −inf.

Averaged over keys, the tournament leaves the model's distribution unchanged (this is tested), so the mark costs
little quality on free-form text. Forcing the token makes the processor independent of vLLM's own sampler, including
greedy requests (temperature 0 is replaced by 0.7, since greedy decoding leaves no room for a mark). Logprobs still come
from the unmodified model.

**Detection.** The app re-tokenizes the text, recomputes the g-values for each token from its preceding 4 tokens and
the key, and runs a z-test on their weighted mean. Without the watermark each g-value is a fair coin, so the null
distribution is known exactly and no detector training is needed. Repeated n-grams are skipped. Verdicts:

| Verdict | Meaning |
|---|---|
| `watermarked` | p < 0.001 |
| `not_detected` | at least 200 tokens scored and p ≥ 0.001 |
| `too_short` | fewer than 200 tokens scored and p ≥ 0.001; a negative result means little |

The 200-token threshold follows the EU Code of Practice on Transparency of AI-Generated Content. Every response
includes the p-value, z-score, number of tokens scored, key ID, a timestamp, a SHA-256 of the text and an HMAC
signature over the result.

The detector needs only `numpy` and the model's `tokenizer.json`, not the model or a GPU.

## Supported models and regions

Custom LLM Serving runs on a single GPU. Both models below fit on one 80 GB A100 (`GPU_LARGE` on Azure Databricks).

| `model` | Hugging Face repo | Weights | Notes |
|---|---|---|---|
| `qwen3_8_27b` (default) | `Qwen/Qwen3.8-27B` | BF16, ~56 GB | Comfortable fit; thinking is off by default on the server; no MTP speculative decoding |
| `gpt_oss_120b` | `openai/gpt-oss-120b` | MXFP4, ~65 GB | Tight; context capped at 16k tokens, vLLM's single-GPU settings applied |

**EU regions.** Custom LLM Serving is in Public Preview and its documentation currently lists US regions only, although
AI Runtime (which it depends on) is GA in EU regions. This example was deployed end to end in an **Azure Sweden
Central** workspace on 23 November 2026: logged on AI Runtime 8x H100, served on `GPU_LARGE` (1x A100). Check
availability in your own region and workspace before relying on it.

## Tested results

`llmwm` mode, Qwen3.8-27B, vLLM 0.33.0, Azure Sweden Central, 23 November 2026. Each reply was checked from its
decoded text by notebook 02.

| Prompt | Arm | Verdict | z | p-value | Tokens scored |
|---|---|---|---|---|---|
| 300-word blog post on provenance | watermarked | watermarked | 6.4 | 7.4e-11 | 396 |
| | control (opt-out) | not_detected | -1.1 | 0.85 | 396 |
| Heat pump explainer | watermarked | watermarked | 5.0 | 2.8e-07 | 396 |
| | control (opt-out) | not_detected | 0.6 | 0.27 | 396 |
| Team planning email | watermarked | watermarked | 4.9 | 5.7e-07 | 297 |
| | control (opt-out) | not_detected | 1.4 | 0.08 | 292 |

The vLLM-native modes are tested only on CPU so far (detector bit-for-bit against vLLM's code), not on a GPU endpoint.

## Prerequisites

- A workspace with Unity Catalog, **AI Runtime**, and **Custom LLM Serving** enabled (Admin settings → Previews).
- An existing catalog where you can create a schema.
- A recent Databricks CLI (tested with v1.24.0), authenticated to the workspace (`databricks auth login --host … --profile …`).

## Deploy

```bash
PROFILE=<your-profile>
CATALOG=<your-catalog>

# 1. Create the watermark key (keep it secret; anyone with it can detect and forge the mark)
databricks secrets create-scope llm-watermark -p $PROFILE
databricks secrets put-secret llm-watermark watermark-key --string-value "$(openssl rand -hex 32)" -p $PROFILE

# 2. Deploy the bundle: notebooks, schema, volume and the app
databricks bundle deploy -p $PROFILE --var catalog=$CATALOG
```

3. Open `notebooks/01_log_and_deploy` in the bundle folder in your workspace, attach it to **AI Runtime** (8x H100 in
   Azure EU regions) with the **Standard environment, version 6** (the notebook installs vLLM 0.33.0) or the **AI
   environment, version 6** (it uses the bundled vLLM), set the `catalog` widget, and run all. Set `stage` to `test` to
   stop after the in-notebook vLLM check, and `wait_for_endpoint` to `false` to avoid holding the GPU while the
   endpoint builds (20–30 minutes).

   To run it as a job instead, submit a one-time run with `"compute": {"hardware_accelerator": "GPU_8xH100"}` on the
   task and `{"environment_version": "6"}` as its environment. It logs and registers the model, creates the endpoint, writes
   the tokenizer to the volume and grants the app access. Detach once it reaches step 9; the endpoint keeps deploying.

4. Start the app and open it:

```bash
databricks bundle run watermark_detector -p $PROFILE --var catalog=$CATALOG
```

5. Optional: run `notebooks/02_smoke_test` on serverless CPU.

To deploy gpt-oss-120b instead, set the notebook's `model` widget and pass `--var model=gpt_oss_120b` to the bundle.
To demo a side-by-side unwatermarked control, set `allow_opt_out=true` in the notebook and `--var allow_control=true`
for the bundle. Leave both off otherwise: an opt-out flag is a way to switch the watermark off.

## Cost

List prices, which vary by region and contract:

- AI Runtime 8x H100 notebook (the only in-geo option in Azure EU regions): about $7 per H100-hour, so roughly
  **$56 per hour**. Notebook 01 takes roughly 30–60 minutes per model with the local test, less without it.
- `GPU_LARGE` endpoint: 78.6 DBU per hour on Azure. The endpoint scales to zero when idle, but cold starts take several
  minutes. Delete it when you're done.

## Use the API

```bash
APP_URL=$(databricks apps get llm-watermark-detector -p $PROFILE -o json | jq -r .url)
TOKEN=$(databricks auth token -p $PROFILE | jq -r .access_token)
curl -s -X POST "$APP_URL/api/v1/detect" -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"text": "…"}' | jq
```

Endpoints: `POST /api/v1/detect`, `POST /api/v1/generate`, `POST /api/v1/robustness`, `GET /api/v1/info`,
and OpenAPI docs at `/docs`. Access is limited to users and service principals with permission on the app, which maps
to the Code of Practice's option of restricting text-watermark detection to verified experts. An open detector would
let anyone query it until an edit removes the mark.

## Run the tests

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest -m "not slow"   # unit tests, seconds
.venv/bin/python -m pytest -m slow -s      # generates with SmolLM2-135M on CPU, about a minute
```

The vLLM processor is tested against a stub of the vLLM 0.33 interface, since vLLM needs CUDA. The notebook's local
test (step 5) is the first run against real vLLM.

## Limitations

- **Not SynthID.** The algorithm follows the published SynthID-Text tournament, but the hashing and keys are this
  project's own. Google's detectors will not recognise this mark, and this detector will not recognise Google's.
- **Robustness.** Like all statistical text watermarks, the mark survives light edits and truncation but is weakened
  or removed by paraphrasing and translation. Short and low-entropy outputs (code, numbers, fixed formats) carry
  little signal.
- **vLLM features.** Speculative decoding is rejected. `top_p`, `min_p` and per-request seeds are not applied to
  watermarked rows. The processor targets vLLM 0.33's Model Runner V2 interface, which is still evolving.
- **Tokenization.** Detection re-tokenizes decoded text; non-canonical tokenizations slightly weaken the signal.
- **Key management.** One key from a Databricks secret. The detector supports several keys (`detect_with_keys`) so
  retired keys still detect older text, but rotation is manual.
- **Endpoint request fields.** The control option relies on the serving endpoint passing `vllm_xargs` (`{"watermark": 0}`) through to vLLM.

## License

Apache 2.0. Model weights are downloaded from Hugging Face under their own licenses (both Apache 2.0 at the time of
writing).
