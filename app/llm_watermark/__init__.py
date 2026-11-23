"""SynthID-style text watermarking for vLLM on Databricks, with a numpy-only detector."""

from .config import SCHEME, WatermarkConfig, WatermarkKey
from .detector import DetectionResult, detect_tokens, detect_with_keys

__all__ = ["SCHEME", "WatermarkConfig", "WatermarkKey", "DetectionResult", "detect_tokens", "detect_with_keys"]
__version__ = "0.1.0"
