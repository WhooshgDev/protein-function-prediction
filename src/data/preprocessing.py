from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

VALID_AMINO_ACIDS: Set[str] = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def normalize_protein_id(raw_header: str) -> str:
    """Normalize FASTA header to canonical protein ID.
    Handles standard UniProt formats (e.g. sp|P12345|NAME -> P12345) or plain whitespace-delimited IDs.
    """
    token = raw_header.strip().lstrip(">").split()[0]
    if "|" in token:
        parts = token.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return token


def parse_fasta(fasta_path: Union[str, Path]) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Parse FASTA file, normalize IDs, validate characters, and collect QC stats."""
    protein_ids: List[str] = []
    sequences: List[str] = []
    seen_ids: Set[str] = set()
    seen_seqs: Set[str] = set()

    duplicate_ids = 0
    duplicate_seqs = 0
    empty_records = 0
    invalid_char_records = 0

    current_id = None
    current_seq_parts: List[str] = []

    def _commit_record(pid: str, seq_parts: List[str]):
        nonlocal duplicate_ids, duplicate_seqs, empty_records, invalid_char_records
        seq = "".join(seq_parts).upper().strip()
        if not seq:
            empty_records += 1
            return
        if pid in seen_ids:
            duplicate_ids += 1
            return
        if any(c not in VALID_AMINO_ACIDS for c in seq):
            invalid_char_records += 1
        if seq in seen_seqs:
            duplicate_seqs += 1
        seen_ids.add(pid)
        seen_seqs.add(seq)
        protein_ids.append(pid)
        sequences.append(seq)

    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    _commit_record(current_id, current_seq_parts)
                current_id = normalize_protein_id(line)
                current_seq_parts = []
            else:
                current_seq_parts.append(line)
        if current_id is not None:
            _commit_record(current_id, current_seq_parts)

    lengths = [len(s) for s in sequences]
    qc_stats = {
        "num_proteins": len(protein_ids),
        "duplicate_ids": duplicate_ids,
        "duplicate_sequences": duplicate_seqs,
        "empty_records": empty_records,
        "invalid_char_records": invalid_char_records,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "mean_length": float(sum(lengths) / len(lengths)) if lengths else 0.0,
    }
    return protein_ids, sequences, qc_stats


def parse_annotations(tsv_path: Union[str, Path]) -> List[Tuple[str, str, str]]:
    """Parse annotation TSV file into deduplicated (protein_id, go_id, aspect) tuples."""
    records: List[Tuple[str, str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    with open(tsv_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split("\t")
            if not parts or len(parts) < 2:
                continue
            if idx == 0 and not parts[1].strip().startswith("GO:"):
                continue  # header row
            pid = parts[0].strip()
            go_id = parts[1].strip()
            aspect = parts[2].strip() if len(parts) > 2 else ""
            if (pid, go_id) not in seen:
                seen.add((pid, go_id))
                records.append((pid, go_id, aspect))
    return records


def parse_fasta_header_field(fasta_path: Union[str, Path], field_index: int = 1) -> Dict[str, str]:
    """Return {normalized_id: whitespace-separated header field ``field_index``}.

    The CAFA 5 test superset stores the NCBI taxon ID in the header (``>Q9CQV8<TAB>10090``).
    """
    out: Dict[str, str] = {}
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].strip().split()
                if len(parts) > field_index:
                    out[normalize_protein_id(line)] = parts[field_index]
    return out


def prepare_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """Contract entrypoint: orchestrate data ingestion, ontology, label matrices, and splits.

    Writes all artifacts to ``config['data']['processed_dir']`` (see ``src/data/pipeline.py``).
    """
    from src.data.pipeline import prepare_data as _prepare

    return _prepare(config)
