from typing import Any, Dict
from src.utils.config import load_config
from src.utils.seed import set_seed


def run_training_from_config(config_path: str, aspect: str = "F") -> Dict[str, Any]:
    """High-level training helper invoked by scripts/train.py."""
    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))
    return {"status": "configured", "aspect": aspect, "config": config}
