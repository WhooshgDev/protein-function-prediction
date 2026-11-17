import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Union


def compute_file_hash(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def compute_list_hash(items: list) -> str:
    """Compute deterministic hash of an ordered list of items."""
    payload = json.dumps(items, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def save_json_atomic(data: Any, output_path: Union[str, Path], indent: int = 2) -> None:
    """Atomically write JSON data to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(out.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        os.replace(tmp_path, out)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_json(input_path: Union[str, Path]) -> Any:
    """Read JSON file from disk."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
