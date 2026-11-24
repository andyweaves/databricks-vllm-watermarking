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
    launch_native.py      starts vLLM with its built-in watermark, keyed from the secret (default mode)
    vllm_native.py        numpy port of vLLM's watermark detector, so the app needs no GPU or vLLM
    modes.py, models.py   watermark modes and per-model request settings
    sampler.py            optional llmwm mode: SynthID-style tournament sampling
    detector.py           optional llmwm mode: analytic z-test, no trained detector needed
    vllm_processor.py     optional llmwm mode: vLLM >= 0.33 custom logits processor (Model Runner V2 interface)
    hf_processor.py       Hugging Face Transformers adapter, for local tests
  app.py, static/         FastAPI detection service and UI (Databricks App)
notebooks/
  01_log_and_deploy.py    AI Runtime: download weights, test vLLM + watermark, log, register, deploy
  02_smoke_test.py        query the endpoint and check its replies, edits and the other scheme
  03_gateway_check.py     detect the watermark in responses logged by AI Gateway's inference table
tests/                    CPU tests, including end-to-end generation with a 135M model
databricks.yml            bundle: schema, volume and the app
```

## Watermark modes

Notebook 01's `watermark_mode` widget and the bundle's `watermark_mode` variable must match.

| Mode | What runs on the endpoint | Opt-out by callers | Speculative decoding | Temperature 0 |
|---|---|---|---|---|
| `vllm_dual_key_gumbel` (default) | vLLM's built-in dual-key Gumbel-max watermark (vLLM 0.30+), started by `llm_watermark.launch_native` | `watermarking: false` works on vLLM 0.33; on 0.34+ the launcher sets `enforce` | Supported by vLLM (not enabled here) | Not marked |
| `vllm_gumbel` | vLLM's built-in single-key Gumbel-max watermark | As above | Not supported by this algorithm | Not marked |
| `llmwm` (optional) | This repo's SynthID-style logits processor | Only if the endpoint sets `WATERMARK_ALLOW_OPT_OUT=true` | Not supported | Marked (sampled at 0.7) |

In the vLLM-native modes the launcher derives vLLM's 64-bit key from the Databricks secret, so the key never appears
in the logged model. The app detects them with a numpy port of vLLM's Philox PRF and Gumbel detector, tested
bit-for-bit against vLLM's own code, so the app needs neither a GPU nor vLLM.

### Why vLLM's built-in watermark is the default

- **No watermark code of our own on the generation path.** vLLM maintains the sampler, keeps it working across its own
  releases and model runners, and handles batching, CUDA graphs and (in dual-key mode) speculative decoding. The only
  code this repo maintains is the launcher and the CPU detector port, which is pinned to vLLM's implementation by tests.
- **Dual-key rather than single-key.** Plain Gumbel-max takes all of its randomness from the key and the previous
  tokens, so the same prompt at the same temperature gives the same reply every time (vLLM's docs note this for
  `n > 1`). Dual-key picks between two keys at random for each token, which restores variety between requests at the
  cost of a slightly weaker signal per token, and it is the variant vLLM supports with speculative decoding.
- **What to know.** On vLLM 0.33, the version on the Databricks PyPI proxy at the time of writing, any caller allowed to
  query the endpoint can send `watermarking: false`; upgrade to vLLM 0.34 or later, where the launcher sets `enforce`,
  or block that field in front of the endpoint. Greedy (temperature 0) requests are not marked, so reject or override
  them in the client or app. It is not SynthID, so the open-source SynthID tools cannot detect it.

### Why `llmwm` is optional, and the alternatives

`llmwm` implements the SynthID-Text tournament from the published paper as a custom vLLM logits processor. It works
(see Tested results), and it blocks opt-out and marks greedy requests even on vLLM 0.33. It is not the default because
the repo would have to maintain it: vLLM's logits processor interface is still marked as in progress and has already
changed once (Model Runner V2, vLLM 0.31), the processor has to track per-request state and force the sampled token
itself, and it cannot be combined with speculative decoding. Its hashing and keys are also this repo's own rather than
SynthID's reference functions, so Google's open-source SynthID detectors do not recognise it.

Alternatives, if you need SynthID specifically:

| Option | What it is | Trade-off |
|---|---|---|
| Transformers SynthID ([`SynthIDTextWatermarkLogitsProcessor`](https://huggingface.co/docs/transformers/main/en/internal/generation_utils)) | Google's reference SynthID-Text, maintained in Hugging Face Transformers, with a trainable Bayesian detector | Runs inside Transformers `generate()`, which has no continuous batching or paged attention, so throughput per GPU is far lower than vLLM. Serve it as an MLflow pyfunc on classic GPU Model Serving. We chose vLLM over this for performance |
| `llmwm` with SynthID's reference hashing | Call Transformers' SynthID hashing and g-value functions from this repo's vLLM processor | The open-source SynthID tools can then read the output, but the vLLM integration is still ours to maintain, plus a pinned Transformers dependency in the server |
| [Meta TextSeal](https://github.com/facebookresearch/textseal) | Research watermark with two-key detection and speculative-decoding support | Research release, not a supported product |
| Green-list (Kirchenbauer et al.) | The classic keyed vocabulary bias, easy to write as a vLLM processor | Distortion-based: it shifts token probabilities, so it costs quality, especially on code |

Keep the scheme behind one interface (here, `modes.py`) so it can be swapped as standards for interoperable text
watermark detection settle.

## How the optional `llmwm` mode works

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

All runs: vLLM 0.33.0, Azure Sweden Central, Custom LLM Serving on `GPU_LARGE` (1x A100), temperature 0.7, up to
400 tokens per reply. Each reply was checked from its decoded text by notebook 02, which also applies the Robustness
tab's edits. Controls were generated by the same endpoint with the watermark switched off per request.

**Default mode, `vllm_dual_key_gumbel`, Qwen3.8-27B, 24 November 2026:**

| Check (three prompts) | Verdicts | z |
|---|---|---|
| Watermarked replies | 3 of 3 detected | 3.2 to 6.1 |
| Controls | 0 of 3 detected | −0.8 to 0.3 |
| Truncated to 150 tokens | 2 of 3 detected (one too short) | 2.8 to 3.9 |
| 20% of words deleted | 0 of 3 detected | 1.8 to 2.7 |
| Sentences shuffled | 2 of 3 detected | 2.1 to 4.4 |
| Checked with the `llmwm` detector | 0 of 3 detected | −0.8 to 1.4 |
| Generated at temperature 0 | not marked, as documented | 0.3 |

**`vllm_gumbel` on gpt-oss-120b, 24 November 2026**, in notebook 01's local test on one H100: watermarked reply
detected (z 4.2, 548 tokens), control not (z 1.0). Its endpoint was not tested: it waited for an A100 in Sweden Central
for two hours and was deleted.

**Optional mode, `llmwm`, Qwen3.8-27B, 23–24 November 2026**, with AI Gateway switched on for the endpoint
(inference table, usage tracking, rate limit):

| Check (three prompts) | Verdicts | z |
|---|---|---|
| Watermarked replies | 3 of 3 detected (twice) | 3.5 to 7.7 |
| Controls | 0 of 3 detected (twice) | −1.1 to 2.6 |
| Truncated to 150 tokens | 3 of 3 detected | 3.5 to 4.0 |
| 20% of words deleted | 1 of 3 detected | 1.8 to 3.3 |
| Sentences shuffled | 3 of 3 detected | 4.4 to 6.7 |
| Checked with the `vllm_gumbel` detector | 0 of 3 detected | 0.2 to 1.1 |
| Generated at temperature 0 | detected | 6.3 |

On these prompts the default mode gives a weaker signal per token than `llmwm`, and the shortest reply (a 360-token
email) only just passes the threshold. Gumbel-max detection power depends on how much choice the model has at each
token, so expect weaker results on low-entropy or short text, and calibrate on your own outputs.

**Gateway.** Notebook 03 read the `llmwm` requests back from the AI Gateway inference table: every logged watermarked
response was detected with exactly the z-score the client saw, and no control was, so the gateway logs the text
unchanged. AI Gateway guardrails are not available for this endpoint type, and Unity Gateway model services do not
accept custom serving endpoints as destinations (only pay-per-token, provisioned throughput and external models).

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
To run both side by side, deploy the second model to its own endpoint (for example `endpoint_name=watermarked-llm-gpt-oss`)
and tell the app about it, so the UI offers a model selector:

```bash
databricks bundle deploy -p $PROFILE --var catalog=$CATALOG --var model_2=gpt_oss_120b \
  --var watermark_mode_2=vllm_gumbel --var endpoint_name_2=watermarked-llm-gpt-oss
```

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

## Test it yourself

Allow 10–15 minutes. If the endpoint has scaled to zero, the first request wakes it, which takes several minutes; send
one request and wait for the endpoint's state to show **Ready** before you start.

1. **Generate and check.** Open the app (**Compute → Apps → llm-watermark-detector**). On **Generate and check**, pick a
   model, keep the default prompt, tick **Also generate an unwatermarked control** (needs `allow_control=true`) and
   click **Generate and check**. Expect **Watermark detected** for the watermarked reply, with a p-value below 0.001,
   a z-score of roughly 4 or more and mean g above 0.5, and **Not detected** for the control. The shading shows which
   tokens the key favoured. There is no visible difference between the two texts: that is the point.
2. **Check text from outside the app.** Query the endpoint directly, then paste the reply into **Check text**:

   ```bash
   databricks serving-endpoints query watermarked-llm -p $PROFILE --json '{
     "messages": [{"role": "user", "content": "Write 300 words on the history of the bicycle."}],
     "max_tokens": 400, "temperature": 0.7}' | jq -r '.choices[0].message.content'
   ```

   Expect **Watermark detected**. Also check something you wrote yourself (200 words or more): expect **Not
   detected**. A sentence or two gives **Too short to assess**, because the test needs about 200 tokens.
3. **Wrong key or wrong scheme.** Check a reply from one model with the other model selected in the app. Expect
   **Not detected**, because each endpoint uses a different scheme (`llmwm` or vLLM's built-in `vllm_gumbel`). The
   detector only recognises a mark made with its own key and scheme.
4. **Edit the text.** Paste a watermarked reply into **Robustness** and click **Run edits**. Shuffling sentences
   matters least, because the mark is keyed on the previous four tokens, which mostly stay together within a sentence.
   Truncating to 150 tokens roughly halves the z-score and may leave too little text. Deleting 20% of the words does
   the most damage, because each deletion also breaks the context of the next few tokens; expect it to fall below the
   threshold most of the time. Then paraphrase the reply by hand, or ask another model to rewrite it, and check
   again: expect the score to fall towards chance. This is the limit of every statistical text watermark.
5. **Temperature 0.** Set temperature to 0 and generate. vLLM's built-in watermark (the default) leaves greedy
   requests unmarked, so expect **Not detected**: block or override temperature 0 in front of a production endpoint.
   The optional `llmwm` processor still samples, at `WATERMARK_GREEDY_TEMPERATURE` (0.7), so its replies are marked.
6. **Through the gateway.** Switch on AI Gateway for the endpoint (inference table, usage tracking, rate limits; see
   `notebooks/03_gateway_check`), send a few requests, wait for rows to reach the inference table (up to an hour) and
   run notebook 03. Expect every logged watermarked response to be detected, with the same z-score the client saw.

## Run the tests

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest -m "not slow"   # unit tests, seconds
.venv/bin/python -m pytest -m slow -s      # generates with SmolLM2-135M on CPU, about a minute
```

The vLLM processor is tested against a stub of the vLLM 0.33 interface, since vLLM needs CUDA. The notebook's local
test (step 5) is the first run against real vLLM.

## Limitations

- **No shared detector.** Neither mode is SynthID: vLLM's watermark is a Gumbel-max scheme, and `llmwm` follows the
  published SynthID-Text tournament with this project's own hashing. Google's and Anthropic's detectors will not
  recognise either mark, and this detector will not recognise theirs. No interoperable standard for text watermark
  detection exists yet.
- **Robustness.** Like all statistical text watermarks, the mark survives light edits and truncation but is weakened
  or removed by paraphrasing and translation. Short and low-entropy outputs (code, numbers, fixed formats) carry
  little signal.
- **vLLM features (`llmwm`).** Speculative decoding is rejected. `top_p`, `min_p` and per-request seeds are not applied to
  watermarked rows. The processor targets vLLM 0.33's Model Runner V2 interface, which is still evolving.
- **Tokenization.** Detection re-tokenizes decoded text; non-canonical tokenizations slightly weaken the signal.
- **Key management.** One key from a Databricks secret. The detector supports several keys (`detect_with_keys`) so
  retired keys still detect older text, but rotation is manual.
- **Endpoint request fields.** The control option relies on the serving endpoint passing `watermarking: false` (vLLM-native modes) or `vllm_xargs` (`{"watermark": 0}`, `llmwm`) through to vLLM. Both pass through on Custom LLM Serving.

## License

Apache 2.0. Model weights are downloaded from Hugging Face under their own licenses (both Apache 2.0 at the time of
writing).
