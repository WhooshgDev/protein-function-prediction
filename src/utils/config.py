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
