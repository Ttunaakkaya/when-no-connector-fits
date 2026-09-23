"""When No Connector Fits (wncf).

Importing this package pins the Hugging Face caches before transformers is imported.
The user-level HF_HOME / HF_HUB_CACHE / HF_XET_CACHE point at D:, which is not always
mounted, so the project uses its own cache unless WNCF_HF_HOME overrides it.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_hf_home = Path(os.environ.get("WNCF_HF_HOME", Path.home() / ".cache" / "huggingface"))
os.environ["HF_HOME"] = str(_hf_home)
os.environ["HF_HUB_CACHE"] = str(_hf_home / "hub")
os.environ["HF_XET_CACHE"] = str(_hf_home / "xet")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Pinned model revisions (commit hashes on the Hub, recorded 2026-09-22).
MODELS = {
    "7b": ("llava-hf/llava-onevision-qwen2-7b-ov-hf", "0d50680527681998e456c7b78950205bedd8a068"),
    "0.5b": ("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", "74dd0bf867a4cda7950c17663794267c60cf4b40"),
}
