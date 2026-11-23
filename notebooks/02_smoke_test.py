# Databricks notebook source
# MAGIC %md
# MAGIC # Smoke test: query the watermarked endpoint and detect the mark
# MAGIC
# MAGIC Runs on serverless CPU. Sends a few prompts to the endpoint, then checks each reply with the same detector the app
# MAGIC uses. If the endpoint allows opt-out (`allow_opt_out=true` in notebook 01), it also generates unwatermarked
# MAGIC controls, which should not be detected.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install the detector
# MAGIC
# MAGIC Only numpy and the tokenizer are needed: detection does not load the model.

# COMMAND ----------

# Resolve the bundle root (the folder with pyproject.toml) from this notebook's path, so the install
# does not depend on the working directory.
import os

_nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
BUNDLE_ROOT = os.path.dirname(os.path.dirname(_nb if _nb.startswith("/Workspace") else "/Workspace" + _nb))
assert os.path.exists(os.path.join(BUNDLE_ROOT, "pyproject.toml")), f"pyproject.toml not found under {BUNDLE_ROOT}"
with open("/tmp/llm_watermark_requirements.txt", "w") as f:
    f.write('tokenizers\nopenai\ndatabricks-sdk>=0.102.0\n' + BUNDLE_ROOT + "\n")
print("installing llm_watermark from", BUNDLE_ROOT)

# COMMAND ----------

# MAGIC %pip install -q -r /tmp/llm_watermark_requirements.txt

# COMMAND ----------

# Environments can be cached between runs, so always reinstall this package from the current bundle files.
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", "--no-cache-dir",
                BUNDLE_ROOT], check=True)

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Settings

# COMMAND ----------

dbutils.widgets.dropdown("model", "qwen3_8_27b", ["qwen3_8_27b", "gpt_oss_120b"])
dbutils.widgets.dropdown("watermark_mode", "llmwm", ["llmwm", "vllm_gumbel", "vllm_dual_key_gumbel"])
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "llm_watermark")
dbutils.widgets.text("volume", "artifacts")
dbutils.widgets.text("secret_scope", "llm-watermark")
dbutils.widgets.text("secret_key", "watermark-key")
dbutils.widgets.text("endpoint_name", "watermarked-llm")
dbutils.widgets.dropdown("test_control", "false", ["false", "true"])

w_ = dbutils.widgets.get
MODEL_KEY, ENDPOINT = w_("model"), w_("endpoint_name")
assert w_("catalog"), "Set the catalog widget at the top of the notebook, then use Run all below from this cell"
TOKENIZER = f"/Volumes/{w_('catalog')}/{w_('schema')}/{w_('volume')}/{MODEL_KEY}/tokenizer.json"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generate and detect
# MAGIC
# MAGIC Expect `watermarked` with a very small p-value for replies of a few hundred tokens, and `not_detected` or
# MAGIC `too_short` for controls.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from tokenizers import Tokenizer

from llm_watermark import WatermarkConfig, WatermarkKey
from llm_watermark.modes import detect, opt_out_fields

w = WorkspaceClient()
client = w.serving_endpoints.get_open_ai_client()
tok = Tokenizer.from_file(TOKENIZER)
key = WatermarkKey(dbutils.secrets.get(w_("secret_scope"), w_("secret_key")))

PROMPTS = [
    "Write a 300-word blog post about why provenance matters for AI-generated content.",
    "Explain how a heat pump works to someone with no engineering background.",
    "Draft a friendly email inviting the team to a quarterly planning session.",
]


def generate(prompt: str, watermark: bool) -> str:
    extra = {} if watermark else opt_out_fields(w_("watermark_mode"))
    resp = client.chat.completions.create(model=ENDPOINT, messages=[{"role": "user", "content": prompt}],
                                          max_tokens=400, temperature=0.7, extra_body=extra or None)
    return resp.choices[0].message.content or ""


rows = []
for prompt in PROMPTS:
    for marked in ([True, False] if w_("test_control") == "true" else [True]):
        text = generate(prompt, marked)
        r = detect(tok.encode(text, add_special_tokens=False).ids, key, w_("watermark_mode"), WatermarkConfig())
        rows.append((prompt[:40], "watermarked" if marked else "control", r.verdict, round(r.z_score, 1),
                     f"{r.p_value:.1e}", r.num_scored))
display(spark.createDataFrame(rows, "prompt string, arm string, verdict string, z double, p string, tokens int"))

# COMMAND ----------

import json

dbutils.notebook.exit(json.dumps(rows))
