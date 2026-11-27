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
    float_format = int(cfg.get("float_format", 4))

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
                f.write(f"{pid}{sep}{go_id}{sep}{val:.{float_format}f}\n")
                num_rows_written += 1

    return {
        "output_path": str(out_path),
        "num_target_proteins": len(protein_ids),
        "num_covered_proteins": len(proteins_with_predictions),
        "num_rows_written": num_rows_written,
    }


class StreamingCafaWriter:
    """Write CAFA rows protein-by-protein with schema checks (plan §11).

    Format matches ``sample_submission.tsv``: no header, ``protein<TAB>GO<TAB>score``.
    Guarantees: GO IDs match ``GO:`` + 7 digits, scores finite in [0, 1], no duplicate (protein, GO) pairs,
    every target protein gets at least one row.
    """

    def __init__(self, output_path: Union[str, Path], sep: str = "\t", float_format: int = 3):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self.f = open(self.tmp, "w", encoding="utf-8", newline="\n")
        self.sep = sep
        self.fmt = f"{{:.{int(float_format)}f}}"
        self.rows = 0
        self.proteins: Set[str] = set()

    def write_protein(self, pid: str, go_ids: List[str], scores: np.ndarray) -> None:
        if pid in self.proteins:
            raise ValueError(f"Protein {pid} written twice (rows must be grouped per protein)")
        if len(go_ids) != len(scores) or len(go_ids) == 0:
            raise ValueError(f"Protein {pid}: need >=1 (GO, score) pair")
        if len(set(go_ids)) != len(go_ids):
            raise ValueError(f"Duplicate GO IDs for {pid}")
        scores = np.asarray(scores, dtype=np.float64)
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError(f"Scores for {pid} outside [0, 1] or non-finite")
        sep, fmt = self.sep, self.fmt
        self.f.write("".join(f"{pid}{sep}{g}{sep}{fmt.format(s)}\n" for g, s in zip(go_ids, scores)))
        self.rows += len(go_ids)
        self.proteins.add(pid)

    def close(self) -> Dict[str, Any]:
        self.f.close()
        self.tmp.replace(self.path)
        return {"output_path": str(self.path), "num_rows_written": self.rows,
                "num_covered_proteins": len(self.proteins)}


def validate_submission(
    path: Union[str, Path],
    expected_ids: Optional[List[str]] = None,
    valid_go_ids: Optional[Set[str]] = None,
    sep: str = "\t",
    max_terms_per_protein: Optional[int] = None,
) -> Dict[str, Any]:
    """Read a submission back line by line and verify schema, ranges, duplicates and coverage."""
    n_rows = 0
    seen_proteins: Set[str] = set()
    current, current_terms = None, set()
    errors: List[str] = []
    max_terms_seen = 0
    min_s, max_s = 1.0, 0.0

    def _close_block():
        nonlocal max_terms_seen
        max_terms_seen = max(max_terms_seen, len(current_terms))
        if max_terms_per_protein and len(current_terms) > max_terms_per_protein:
            errors.append(f"{current}: {len(current_terms)} terms > limit")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(sep)
            n_rows += 1
            if len(parts) != 3:
                errors.append(f"line {n_rows}: expected 3 columns")
                continue
            pid, go, sc = parts
            if pid != current:
                if current is not None:
                    _close_block()
                if pid in seen_proteins:
                    errors.append(f"{pid}: rows not contiguous")
                seen_proteins.add(pid)
                current, current_terms = pid, set()
            if go in current_terms:
                errors.append(f"duplicate pair {pid} {go}")
            current_terms.add(go)
            if not GO_ID_PATTERN.match(go) or (valid_go_ids is not None and go not in valid_go_ids):
                errors.append(f"invalid GO id {go}")
            s = float(sc)
            if not (0.0 <= s <= 1.0):
                errors.append(f"score out of range {pid} {go} {sc}")
            min_s, max_s = min(min_s, s), max(max_s, s)
            if len(errors) > 50:
                break
    if current is not None:
        _close_block()
    missing = 0
    unexpected = 0
    if expected_ids is not None:
        exp = set(expected_ids)
        missing = len(exp - seen_proteins)
        unexpected = len(seen_proteins - exp)
        if missing:
            errors.append(f"{missing} target proteins missing")
        if unexpected:
            errors.append(f"{unexpected} unexpected proteins")
    return {
        "path": str(path), "num_rows": n_rows, "num_proteins": len(seen_proteins),
        "missing_targets": missing, "unexpected_proteins": unexpected,
        "max_terms_per_protein_seen": max_terms_seen, "score_min": min_s, "score_max": max_s,
        "valid": not errors, "errors": errors[:50],
    }
