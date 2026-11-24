# Databricks notebook source
# MAGIC %md
# MAGIC # Gateway check: is the watermark still in the logged responses?
# MAGIC
# MAGIC Runs on serverless CPU. With AI Gateway (the endpoint-level gateway in the Serving tab) switched on for the
# MAGIC watermarked endpoint, every request and response is written to an inference table in Unity Catalog. This notebook
# MAGIC reads that table and runs the detector on each logged response, so you can confirm that the gateway path leaves
# MAGIC the text, and so the watermark, unchanged.
# MAGIC
# MAGIC Switch the gateway on first, for example:
# MAGIC
# MAGIC ```
# MAGIC databricks serving-endpoints put-ai-gateway watermarked-llm --json '{
# MAGIC   "inference_table_config": {"enabled": true, "catalog_name": "<catalog>", "schema_name": "llm_watermark",
# MAGIC                              "table_name_prefix": "watermarked_llm"},
# MAGIC   "usage_tracking_config": {"enabled": true},
# MAGIC   "rate_limits": [{"calls": 60, "key": "user", "renewal_period": "minute"}]}'
# MAGIC ```
# MAGIC
# MAGIC Then send some traffic (notebook 02 or the app) and wait: rows usually reach the table within an hour.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install the detector

# COMMAND ----------

import os

_nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
BUNDLE_ROOT = os.path.dirname(os.path.dirname(_nb if _nb.startswith("/Workspace") else "/Workspace" + _nb))
assert os.path.exists(os.path.join(BUNDLE_ROOT, "pyproject.toml")), f"pyproject.toml not found under {BUNDLE_ROOT}"
with open("/tmp/llm_watermark_requirements.txt", "w") as f:
    f.write("tokenizers\nscipy\n" + BUNDLE_ROOT + "\n")

# COMMAND ----------

# MAGIC %pip install -q -r /tmp/llm_watermark_requirements.txt

# COMMAND ----------

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
dbutils.widgets.dropdown("watermark_mode", "vllm_dual_key_gumbel", ["vllm_dual_key_gumbel", "vllm_gumbel", "llmwm"])
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "llm_watermark")
dbutils.widgets.text("volume", "artifacts")
dbutils.widgets.text("secret_scope", "llm-watermark")
dbutils.widgets.text("secret_key", "watermark-key")
dbutils.widgets.text("payload_table", "watermarked_llm_payload")
dbutils.widgets.text("lookback_hours", "24")

w_ = dbutils.widgets.get
assert w_("catalog"), "Set the catalog widget at the top of the notebook, then use Run all below from this cell"
TABLE = f"{w_('catalog')}.{w_('schema')}.{w_('payload_table')}"
TOKENIZER = f"/Volumes/{w_('catalog')}/{w_('schema')}/{w_('volume')}/{w_('model')}/tokenizer.json"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Read the inference table
# MAGIC
# MAGIC Each row holds the request and response bodies as JSON strings. A request that asked for an unwatermarked
# MAGIC control (`vllm_xargs.watermark = 0`, or `watermarking: false` in the vLLM-native modes) is labelled `control`.

# COMMAND ----------

import json

rows = spark.sql(f"""
    SELECT request_time, databricks_request_id, status_code, request, response
    FROM {TABLE}
    WHERE request_time >= current_timestamp() - INTERVAL {int(w_('lookback_hours'))} HOURS AND status_code = 200
    ORDER BY request_time
""").collect()
print(len(rows), "successful requests logged in", TABLE)


def arm(request: dict) -> str:
    xargs = request.get("vllm_xargs") or {}
    off = str(xargs.get("watermark", "1")).lower() in ("0", "false") or request.get("watermarking") is False
    return "control" if off else "watermarked"


def reply_text(response: dict) -> str:
    try:
        return response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Detect the watermark in each logged response
# MAGIC
# MAGIC Expect `watermarked` for every watermarked row with a few hundred tokens of text, and `not_detected` for
# MAGIC controls. Compare the z-scores with the ones the client saw (notebook 02's output): they should be identical,
# MAGIC because the logged text is the text the client received.

# COMMAND ----------

from tokenizers import Tokenizer

from llm_watermark import WatermarkConfig, WatermarkKey
from llm_watermark.modes import detect

tok = Tokenizer.from_file(TOKENIZER)
key = WatermarkKey(dbutils.secrets.get(w_("secret_scope"), w_("secret_key")))

results = []
for r in rows:
    request, response = json.loads(r.request or "{}"), json.loads(r.response or "{}")
    text = reply_text(response)
    d = detect(tok.encode(text, add_special_tokens=False).ids, key, w_("watermark_mode"), WatermarkConfig())
    prompt = (request.get("messages") or [{}])[-1].get("content", "")
    results.append((str(r.request_time), prompt[:40], arm(request), d.verdict, round(d.z_score, 1),
                    f"{d.p_value:.1e}", d.num_scored))
display(spark.createDataFrame(results, "request_time string, prompt string, arm string, verdict string, z double, "
                                       "p string, tokens int"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Summary

# COMMAND ----------

marked = [x for x in results if x[2] == "watermarked"]
controls = [x for x in results if x[2] == "control"]
summary = {
    "table": TABLE,
    "watermarked_rows": len(marked),
    "watermarked_detected": sum(x[3] == "watermarked" for x in marked),
    "control_rows": len(controls),
    "control_detected": sum(x[3] == "watermarked" for x in controls),
    "rows": results,
}
dbutils.notebook.exit(json.dumps(summary))
