"""Helpers shared by run_eval.py and quantize.py.

Only light imports here: quantize.py also runs in the quantization environment, which has neither vLLM nor BFCL.
"""

import os
import platform
import re
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PACKAGES = ["vllm", "torch", "transformers", "bfcl-eval", "xgrammar", "bitsandbytes", "accelerate", "gptqmodel"]


def resolve_revision(model, revision):
    """Commit hash of the model on Hugging Face. Configs pin a hash; a branch name is looked up."""
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    from huggingface_hub import HfApi
    return HfApi().model_info(model, revision=revision).sha


def checkpoint_path(config_path):
    """Folder of the quantized checkpoint that src/quantize.py makes for a config with a `quantize` section."""
    return Path(os.environ.get("CHECKPOINT_DIR", REPO / "checkpoints")) / Path(config_path).stem


def package_versions():
    versions = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions
