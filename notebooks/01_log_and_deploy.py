# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy a watermarked open-weight LLM with Custom LLM Serving
# MAGIC
# MAGIC This notebook:
# MAGIC
# MAGIC 1. Installs vLLM and the `llm_watermark` package from this repo.
# MAGIC 2. Downloads the model weights from Hugging Face and copies the tokenizer to a Unity Catalog volume (for the detection app).
# MAGIC 3. Optionally starts vLLM in the notebook with the watermark processor and checks that its output is detected.
# MAGIC 4. Logs the model with a vLLM entrypoint that loads the processor, and registers it in Unity Catalog.
# MAGIC 5. Creates or updates a GPU serving endpoint, with the watermark key injected from a Databricks secret.
# MAGIC 6. Grants the detection app permission to query the endpoint.
# MAGIC
# MAGIC **Run it on AI Runtime (serverless GPU).** Custom LLM Serving requires models to be logged from AI Runtime.
# MAGIC Use the **AI** environment, version 6, which bundles vLLM, torch and flash-attn, or the **Standard** environment,
# MAGIC version 6, where the notebook installs vLLM 0.33.0 itself. Never install a second vLLM on top of the AI environment:
# MAGIC a different torch breaks its bundled flash-attn binary.
# MAGIC Set the `catalog` widget before running all.
# MAGIC In Azure EU regions the only in-geo accelerator is 8x H100, which costs roughly $56 per hour at list price:
# MAGIC detach as soon as the notebook finishes.
# MAGIC
# MAGIC **Status:** Custom LLM Serving is in Public Preview and documented as US-only. Deploying it in an EU region is untested.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install the watermark package
# MAGIC
# MAGIC Packages installed with `%pip` here are captured by `env_pack` when the model is registered,
# MAGIC and restored on the serving endpoint. The relative path installs the package from the bundle's root folder.
# MAGIC It depends only on numpy, so it leaves the environment's vLLM and torch untouched.

# COMMAND ----------

# Remove packages left in this notebook's environment overlay by earlier installs (for example a pip-installed
# vLLM or torch), so the AI environment's own vLLM, torch and flash-attn are used. A fresh environment has none.
import importlib.metadata as _md
import subprocess
import sys

_overlay = [p for p in sys.path if ".ephemeral_nfs/envs" in p and p.endswith("site-packages")]
_leftovers = sorted({d.metadata["Name"] for d in _md.distributions(path=_overlay)} - {"llm-watermark", "llm_watermark"})
if _leftovers:
    print("removing from the environment overlay:", ", ".join(_leftovers))
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", *_leftovers], check=True)
else:
    print("environment overlay is clean")

# COMMAND ----------

# Resolve the bundle root (the folder with pyproject.toml) from this notebook's path, so the install
# does not depend on the working directory.
import os

_nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
BUNDLE_ROOT = os.path.dirname(os.path.dirname(_nb if _nb.startswith("/Workspace") else "/Workspace" + _nb))
assert os.path.exists(os.path.join(BUNDLE_ROOT, "pyproject.toml")), f"pyproject.toml not found under {BUNDLE_ROOT}"
import importlib.util

# The AI environment bundles vLLM; the Standard environment does not, so install a pinned vLLM there.
VLLM_PIN = "" if importlib.util.find_spec("vllm") else "vllm==0.33.0\n"
with open("/tmp/llm_watermark_requirements.txt", "w") as f:
    f.write(VLLM_PIN + BUNDLE_ROOT + "\n")
print("installing", (VLLM_PIN.strip() + " and ") if VLLM_PIN else "", "llm_watermark from", BUNDLE_ROOT)

# COMMAND ----------

# MAGIC %pip install -q -r /tmp/llm_watermark_requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Settings
# MAGIC
# MAGIC The defaults match `databricks.yml`. `model` selects one of the two supported models; both fit on one 80 GB A100
# MAGIC (`GPU_LARGE` on Azure). gpt-oss-120b is tight, so its context window is capped.
# MAGIC
# MAGIC `watermark_mode` selects the watermark: `llmwm` is this repo's SynthID-style logits processor; `vllm_gumbel` and
# MAGIC `vllm_dual_key_gumbel` use vLLM's built-in watermark (`--watermark-config`), started through
# MAGIC `llm_watermark.launch_native` so the key comes from the secret at start-up. With vLLM 0.33, callers can switch the
# MAGIC built-in watermark off per request (`watermarking: false`); vLLM 0.34 adds `enforce`, which the launcher sets.

# COMMAND ----------

dbutils.widgets.dropdown("model", "qwen3_8_27b", ["qwen3_8_27b", "gpt_oss_120b"])
dbutils.widgets.dropdown("watermark_mode", "llmwm", ["llmwm", "vllm_gumbel", "vllm_dual_key_gumbel"])
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "llm_watermark")
dbutils.widgets.text("volume", "artifacts")
dbutils.widgets.text("secret_scope", "llm-watermark")
dbutils.widgets.text("secret_key", "watermark-key")
dbutils.widgets.text("endpoint_name", "watermarked-llm")
dbutils.widgets.text("workload_type", "GPU_LARGE")
dbutils.widgets.dropdown("allow_opt_out", "false", ["false", "true"])
dbutils.widgets.text("app_name", "llm-watermark-detector")
dbutils.widgets.dropdown("run_local_test", "true", ["true", "false"])
dbutils.widgets.dropdown("stage", "deploy", ["test", "deploy"])   # test: stop after the local vLLM test
dbutils.widgets.dropdown("wait_for_endpoint", "true", ["true", "false"])

# Flags follow the Databricks Custom LLM Serving starter notebooks, except that speculative decoding (MTP) is
# left out because the watermark processor does not support it.
MODELS = {
    "qwen3_8_27b": {
        "repo": "Qwen/Qwen3.8-27B",
        "vllm_args": [
            "--dtype bfloat16",
            "--gpu-memory-utilization 0.92",
            "--max-model-len 32768",
            "--max-num-seqs 8",
            "--max-num-batched-tokens 8192",
            "--enable-prefix-caching",
            "--language-model-only",              # text chat only; skips the vision encoder
            "--mamba-cache-mode align",           # Qwen3.8 is a hybrid (linear + full attention) model
            "--gdn-prefill-backend triton",
            "--reasoning-parser qwen3",
            "--default-chat-template-kwargs '{\"enable_thinking\":false}'",
        ],
        "ignore_patterns": [],
    },
    "gpt_oss_120b": {
        "repo": "openai/gpt-oss-120b",
        # The vLLM gpt-oss recipe advises these settings for a single 80 GB GPU.
        "vllm_args": ["--max-model-len 16384", "--gpu-memory-utilization 0.95", "--max-num-batched-tokens 1024"],
        "ignore_patterns": ["original/*", "metal/*"],
    },
}

w_ = dbutils.widgets.get
MODEL_KEY = w_("model")
SPEC = MODELS[MODEL_KEY]
CATALOG, SCHEMA, VOLUME = w_("catalog"), w_("schema"), w_("volume")
assert CATALOG, "Set the catalog widget at the top of the notebook, then use Run all below from this cell (no need to re-run the pip install)"
SECRET_SCOPE, SECRET_KEY = w_("secret_scope"), w_("secret_key")
ENDPOINT = w_("endpoint_name")
ALLOW_OPT_OUT = w_("allow_opt_out") == "true"
MODE = w_("watermark_mode")
UC_MODEL = f"{CATALOG}.{SCHEMA}.{MODEL_KEY}_{MODE}"
MODEL_DIR = MODEL_KEY      # weights folder, relative to the working directory; also the path vLLM loads when serving
TOKENIZER_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{MODEL_KEY}"
print(UC_MODEL, ENDPOINT, TOKENIZER_DIR, MODE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Check the environment and the watermark key
# MAGIC
# MAGIC The key is read here only for the optional local test. The endpoint gets it through a secret reference,
# MAGIC so it is never written into the model or the endpoint configuration.

# COMMAND ----------

import os
import subprocess

import torch
import vllm

import llm_watermark
from llm_watermark import WatermarkConfig, WatermarkKey

print("vLLM", vllm.__version__, "| torch", torch.__version__, "| llm_watermark", llm_watermark.__version__)
print("vLLM from", os.path.dirname(vllm.__file__), "| torch from", os.path.dirname(torch.__file__))
if ".ephemeral_nfs" in vllm.__file__ or ".ephemeral_nfs" in torch.__file__:
    raise RuntimeError("vLLM or torch is loading from the notebook overlay, not the AI environment. Re-run the cleanup "
                       "cell at the top, or reset the environment, then run all again.")
_vllm = tuple(int(x) for x in vllm.__version__.split(".")[:2])
_needed = (0, 31) if MODE == "llmwm" else (0, 30)
if _vllm < _needed:
    raise RuntimeError(f"watermark_mode={MODE} needs vLLM {'.'.join(map(str, _needed))} or later, found "
                       f"{vllm.__version__}. Use the AI or Standard environment, version 6 or later.")
if MODE != "llmwm" and _vllm < (0, 34) and not ALLOW_OPT_OUT:
    print(f"NOTE: vLLM {vllm.__version__} cannot block per-request watermarking=false (needs 0.34).")
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
key = WatermarkKey(dbutils.secrets.get(SECRET_SCOPE, SECRET_KEY))
print("watermark key_id:", key.key_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Download the weights and publish the tokenizer
# MAGIC
# MAGIC Weights go to a temporary directory on local disk, as in the Databricks starter notebooks: the notebook's own
# MAGIC folder does not accept multi-GB files. The tokenizer is copied to the UC volume, where the detection app reads it.

# COMMAND ----------

import os
import shutil
import tempfile

from huggingface_hub import snapshot_download

WORK_DIR = os.path.join(tempfile.gettempdir(), "llm_watermark")   # fixed, so re-runs reuse downloaded weights
os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
print("working directory:", os.getcwd())

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

snapshot_download(SPEC["repo"], local_dir=MODEL_DIR, ignore_patterns=SPEC["ignore_patterns"] or None)
os.makedirs(TOKENIZER_DIR, exist_ok=True)
for name in ("tokenizer.json", "tokenizer_config.json"):
    shutil.copy(os.path.join(MODEL_DIR, name), os.path.join(TOKENIZER_DIR, name))
print(subprocess.run(["du", "-sh", MODEL_DIR], capture_output=True, text=True).stdout)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Optional: test vLLM with the watermark in this notebook
# MAGIC
# MAGIC Starts the same command the endpoint will run, on one GPU and on port 3080 (AI Runtime notebooks allow ports
# MAGIC 3000 to 3999; the endpoint uses 8080), sends a prompt to `/invocations`, the route Model Serving calls, and checks
# MAGIC the reply with the detector. This catches flag, memory and processor errors in minutes rather than after a
# MAGIC deployment. Set `run_local_test` to `false` to skip it.

# COMMAND ----------

import os
import signal
import time

import requests
from tokenizers import Tokenizer

from llm_watermark.modes import detect, launch_prefix, opt_out_fields

TEST_PORT = 3080


def vllm_command(port: int) -> str:
    """Shell command run by the endpoint (and by the local test). Environment prefixes are allowed."""
    return " ".join([
        "VLLM_USE_FLASHINFER_SAMPLER=0",      # FlashInfer's sampler fails at startup on Serverless GPU
        "VLLM_USE_V2_MODEL_RUNNER=1",         # the processor implements the Model Runner V2 interface
        *launch_prefix(MODE),                 # vLLM server, plus the watermark processor or launcher
        f"--model {MODEL_DIR}",
        f"--served-model-name {MODEL_KEY}",
        "--host 0.0.0.0",
        f"--port {port}",
        *SPEC["vllm_args"],
    ])


def invoke(prompt: str, watermark: bool = True) -> str:
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 400, "temperature": 0.7}
    if not watermark:
        body.update(opt_out_fields(MODE))
    r = requests.post(f"http://localhost:{TEST_PORT}/invocations", json=body, timeout=600)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r.json()["choices"][0]["message"]["content"]


def log_tail() -> str:
    return open("vllm.log").read()[-5000:]


import json

RESULTS = {"model": MODEL_KEY, "mode": MODE}
if w_("run_local_test") == "true" or w_("stage") == "test":
    env = dict(os.environ, WATERMARK_KEY=key.secret, WATERMARK_ALLOW_OPT_OUT="true", CUDA_VISIBLE_DEVICES="0")
    with open("vllm.log", "w") as log:
        server = subprocess.Popen(vllm_command(TEST_PORT), shell=True, env=env, stdout=log, stderr=log,
                                  start_new_session=True)
    try:
        deadline = time.time() + 30 * 60
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {server.returncode}. End of its log:\n{log_tail()}")
            try:
                if requests.get(f"http://localhost:{TEST_PORT}/health", timeout=2).ok:
                    break
            except requests.ConnectionError:
                pass
            if time.time() > deadline:
                raise RuntimeError(f"vLLM did not start within 30 minutes. End of its log:\n{log_tail()}")
            time.sleep(5)

        tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        prompt = "Write a 300-word blog post about why provenance matters for AI-generated content."
        for label, marked in (("watermarked", True), ("control (opt-out)", False)):
            text = invoke(prompt, watermark=marked)
            result = detect(tok.encode(text, add_special_tokens=False).ids, key, MODE, WatermarkConfig())
            print(f"{label:>18}: {result.verdict:<13} z={result.z_score:6.1f} p={result.p_value:.2e} "
                  f"tokens={result.num_scored}")
            RESULTS[label] = {"verdict": result.verdict, "z": round(result.z_score, 2), "p": result.p_value,
                              "tokens": result.num_scored, "text": text[:400]}
            print("   ", text[:300].replace("\n", " "), "...")
        RESULTS["log"] = [line.strip()[:300] for line in open("vllm.log") if "atermark" in line][-5:]
        print(RESULTS["log"])
    finally:
        if server.poll() is None:                      # still running: stop vLLM and its workers
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            server.wait(timeout=120)

# COMMAND ----------

if w_("stage") == "test":
    dbutils.notebook.exit(json.dumps(RESULTS, default=str))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Log and register the model
# MAGIC
# MAGIC The model's metadata holds the task and the entrypoint. The endpoint runs the entrypoint in the model's artifacts
# MAGIC folder, so `--model` names the logged weights folder. The placeholder class is never called.

# COMMAND ----------

import mlflow
from mlflow.pyfunc.model import ChatCompletionResponse, ChatModel


class Placeholder(ChatModel):
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})


entrypoint = vllm_command(8080)
print(entrypoint)

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run(run_name=f"{MODEL_KEY}-{MODE}"):
    model_info = mlflow.pyfunc.log_model(
        name=MODEL_KEY,
        python_model=Placeholder(),
        artifacts={"model_dir": MODEL_DIR},
        metadata={"task": "llm/v1/chat", "entrypoint": entrypoint},
    )
registered = mlflow.register_model(model_info.model_uri, UC_MODEL, env_pack="databricks_model_serving")
print("registered", UC_MODEL, "version", registered.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Create or update the serving endpoint
# MAGIC
# MAGIC The key is injected with a `{{secrets/scope/key}}` reference, so it never appears in the model or the
# MAGIC endpoint configuration. Scale-to-zero keeps idle cost near zero, but a cold
# MAGIC start reloads the weights and takes several minutes.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType

w = WorkspaceClient()
served = ServedEntityInput(
    entity_name=UC_MODEL,
    entity_version=str(registered.version),
    workload_type=ServingModelWorkloadType(w_("workload_type")),
    workload_size="Small",
    scale_to_zero_enabled=True,
    environment_vars={
        "WATERMARK_KEY": f"{{{{secrets/{SECRET_SCOPE}/{SECRET_KEY}}}}}",
        "WATERMARK_ALLOW_OPT_OUT": "true" if ALLOW_OPT_OUT else "false",
    },
)
existing = {e.name for e in w.serving_endpoints.list()}
if ENDPOINT in existing:
    w.serving_endpoints.update_config(name=ENDPOINT, served_entities=[served])
    print("updating", ENDPOINT)
else:
    w.serving_endpoints.create(name=ENDPOINT, config=EndpointCoreConfigInput(served_entities=[served]))
    print("creating", ENDPOINT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Let the detection app query the endpoint
# MAGIC
# MAGIC The app is deployed by the bundle before this notebook runs, so its service principal already exists.

# COMMAND ----------

from databricks.sdk.service.serving import ServingEndpointAccessControlRequest, ServingEndpointPermissionLevel

try:
    app_sp = w.apps.get(w_("app_name")).service_principal_client_id
    endpoint_id = w.serving_endpoints.get(ENDPOINT).id
    w.serving_endpoints.update_permissions(endpoint_id, access_control_list=[
        ServingEndpointAccessControlRequest(service_principal_name=app_sp,
                                            permission_level=ServingEndpointPermissionLevel.CAN_QUERY)])
    print("granted CAN_QUERY to app service principal", app_sp)
except Exception as e:  # noqa: BLE001
    print("could not grant the app access (deploy the bundle first?):", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Wait for the endpoint
# MAGIC
# MAGIC Building the serving container and loading the weights can take 30 minutes or more. This cell can be re-run,
# MAGIC and you can detach from AI Runtime now: the endpoint keeps deploying without the notebook.

# COMMAND ----------

for _ in range(120 if w_("wait_for_endpoint") == "true" else 0):
    state = w.serving_endpoints.get(ENDPOINT).state
    print(time.strftime("%H:%M:%S"), state.ready, state.config_update)
    if str(state.config_update).endswith("NOT_UPDATING") and str(state.ready).endswith("READY"):
        break
    if str(state.config_update).endswith("UPDATE_FAILED"):
        raise RuntimeError("endpoint update failed; check the endpoint's build and service logs")
    time.sleep(60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Summary

# COMMAND ----------

RESULTS.update({"uc_model": UC_MODEL, "version": registered.version, "endpoint": ENDPOINT})
dbutils.notebook.exit(json.dumps(RESULTS, default=str))
