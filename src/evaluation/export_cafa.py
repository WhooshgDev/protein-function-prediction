import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

GO_ID_PATTERN = re.compile(r"^GO:\d{7}$")


def export_predictions(
    protein_ids: List[str],
    go_vocab: Dict[str, int],
    scores: np.ndarray,
    output_path: Union[str, Path],
    schema_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Contract entrypoint: Export predictions in CAFA TSV/CSV format and validate schema.

    Args:
        protein_ids: Ordered list of N protein IDs
        go_vocab: Mapping of GO_ID -> column index (size K)
        scores: Array of shape (N, K) with values in [0, 1]
        output_path: Destination file path
        schema_config: Optional dict with min_score, max_terms_per_protein, sep, float_format
    """
    cfg = schema_config or {}
    min_score = float(cfg.get("min_score", 0.001))
    max_terms = int(cfg.get("max_terms_per_protein", 1500))
    sep = str(cfg.get("sep", "\t"))

    if len(protein_ids) != scores.shape[0]:
        raise ValueError(f"protein_ids length ({len(protein_ids)}) != scores rows ({scores.shape[0]})")
    if len(go_vocab) != scores.shape[1]:
        raise ValueError(f"go_vocab size ({len(go_vocab)}) != scores cols ({scores.shape[1]})")
    if not np.isfinite(scores).all():
        raise ValueError("Scores contain NaN or Inf.")
    if (scores < 0.0).any() or (scores > 1.0).any():
        raise ValueError("Scores must lie strictly within [0.0, 1.0].")

    idx_to_go = {idx: go_id for go_id, idx in go_vocab.items()}
    for go_id in idx_to_go.values():
        if not GO_ID_PATTERN.match(go_id):
            raise ValueError(f"Invalid GO ID in vocabulary: {go_id}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_rows_written = 0
    proteins_with_predictions: Set[str] = set()
    seen_pairs: Set[Tuple[str, str]] = set()

    with open(out_path, "w", encoding="utf-8") as f:
        for i, pid in enumerate(protein_ids):
            row_scores = scores[i]
            candidate_indices = np.where(row_scores >= min_score)[0]
            if candidate_indices.size == 0 and row_scores.size > 0:
                # Keep top-1 candidate so every target protein is represented if required
                candidate_indices = np.array([int(np.argmax(row_scores))])

            sorted_order = candidate_indices[np.argsort(-row_scores[candidate_indices])][:max_terms]
            for col_idx in sorted_order:
                go_id = idx_to_go[int(col_idx)]
                val = float(row_scores[col_idx])
                if (pid, go_id) in seen_pairs:
                    continue
                seen_pairs.add((pid, go_id))
                proteins_with_predictions.add(pid)
                f.write(f"{pid}{sep}{go_id}{sep}{val:.4f}\n")
                num_rows_written += 1

    return {
        "output_path": str(out_path),
        "num_target_proteins": len(protein_ids),
        "num_covered_proteins": len(proteins_with_predictions),
        "num_rows_written": num_rows_written,
    }
