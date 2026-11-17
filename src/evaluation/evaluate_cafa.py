from pathlib import Path
from typing import Any, Dict, Union


def run_cafa_evaluator_smoke_test(
    predictions_path: Union[str, Path],
    ground_truth_path: Union[str, Path],
    obo_path: Union[str, Path],
    ia_path: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """Wrapper for official CAFA evaluator / smoke test verification."""
    pred_file = Path(predictions_path)
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")

    return {
        "protocol": "evaluator_smoke_test",
        "predictions_path": str(pred_file),
        "ground_truth_path": str(ground_truth_path),
        "obo_path": str(obo_path),
        "ia_path": str(ia_path) if ia_path else None,
        "status": "ready_for_official_cafa_evaluator",
    }
