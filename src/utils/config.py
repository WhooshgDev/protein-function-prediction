import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 fingerprint of a configuration dictionary."""
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def apply_overrides(config: Dict[str, Any], overrides: list) -> Dict[str, Any]:
    """Apply ``dotted.key=value`` overrides (values parsed as YAML) to a config dict in place."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_val = item.split("=", 1)
        node = config
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw_val)
    return config


def load_config_with_overrides(config_path: Union[str, Path], overrides: list = None) -> Dict[str, Any]:
    return apply_overrides(load_config(config_path), overrides or [])
