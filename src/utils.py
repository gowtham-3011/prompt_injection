"""Shared utilities: paths, seeding, logging, config loading, stable hashing."""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

# Environment variables that override config keys. Secrets (API keys) are NOT here:
# they are read directly by the SDKs and never copied into config dicts or logs.
_ENV_OVERRIDES = {
    "PIF_LLM_BACKEND": "llm.backend",
}


def get_logger(name: str) -> logging.Logger:
    """Return a module logger; level comes from the LOG_LEVEL env var (default INFO)."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    """Seed every RNG we use so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is heavy; only seed it if it is installed
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def stable_hash(*parts: Any) -> int:
    """Deterministic 64-bit hash of `parts`.

    Why not built-in hash()? Python salts str hashes per process, so hash("x")
    differs between runs, which would silently break reproducibility.
    """
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def set_by_path(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set cfg["a"]["b"] = value for dotted_key "a.b", creating dicts as needed."""
    *parents, leaf = dotted_key.split(".")
    node = cfg
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def parse_overrides(pairs: Iterable[str]) -> dict:
    """Turn CLI strings like ["llm.backend=mock", "seed=7"] into a nested dict.

    Values are parsed as YAML so numbers/bools/null get their proper types.
    """
    result: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override must look like key=value, got: {pair!r}")
        key, raw = pair.split("=", 1)
        set_by_path(result, key.strip(), yaml.safe_load(raw))
    return result


def _load_dotenv() -> None:
    """Load a local .env if python-dotenv is installed (optional convenience)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> dict:
    """Load a YAML config, then apply env-var overrides, then CLI overrides (highest priority)."""
    _load_dotenv()
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    for env_var, dotted in _ENV_OVERRIDES.items():
        if os.environ.get(env_var):
            set_by_path(cfg, dotted, os.environ[env_var])
    # Model override applies to whichever backend is selected.
    if os.environ.get("PIF_LLM_MODEL"):
        set_by_path(cfg, f"llm.{cfg['llm']['backend']}.model", os.environ["PIF_LLM_MODEL"])

    return deep_merge(cfg, parse_overrides(overrides))


def results_dir(*parts: str) -> Path:
    """Return (and create) a directory under results/."""
    path = PROJECT_ROOT.joinpath("results", *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
