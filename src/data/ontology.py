from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
from scipy import sparse

NAMESPACE_TO_ASPECT = {
    "molecular_function": "F",
    "biological_process": "P",
    "cellular_component": "C",
    "F": "F",
    "P": "P",
    "C": "C",
}


@dataclass
class GOTerm:
    go_id: str
    name: str = ""
    aspect: str = ""
    is_obsolete: bool = False
    replaced_by: List[str] = field(default_factory=list)
    consider: List[str] = field(default_factory=list)
    parents: List[Tuple[str, str]] = field(default_factory=list)  # (parent_go_id, relation_type)


class GOOntology:
    """GO DAG representation supporting parsing, obsolete term resolution, vocabulary, and ancestor propagation."""

    def __init__(self, allowed_relations: Tuple[str, ...] = ("is_a", "part_of")):
        self.allowed_relations = set(allowed_relations)
        self.terms: Dict[str, GOTerm] = {}

    def parse_obo(self, obo_path: Union[str, Path]) -> None:
        """Parse go-basic.obo file into GOTerm records."""
        current_term: Optional[GOTerm] = None

        with open(obo_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if line == "[Term]":
                    if current_term is not None and current_term.go_id:
                        self.terms[current_term.go_id] = current_term
                    current_term = GOTerm(go_id="")
                elif line.startswith("[") and line.endswith("]"):
                    if current_term is not None and current_term.go_id:
                        self.terms[current_term.go_id] = current_term
                    current_term = None
                elif current_term is not None and ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.split("!")[0].strip()
                    if key == "id":
                        current_term.go_id = val
                    elif key == "name":
                        current_term.name = val
                    elif key == "namespace":
                        current_term.aspect = NAMESPACE_TO_ASPECT.get(val, val)
                    elif key == "is_obsolete" and val.lower() == "true":
                        current_term.is_obsolete = True
                    elif key == "replaced_by":
                        current_term.replaced_by.append(val)
                    elif key == "consider":
                        current_term.consider.append(val)
                    elif key == "is_a":
                        if "is_a" in self.allowed_relations:
                            current_term.parents.append((val, "is_a"))
                    elif key == "relationship":
                        rel_parts = val.split()
                        if len(rel_parts) >= 2:
                            rel_type, target_id = rel_parts[0], rel_parts[1]
                            if rel_type in self.allowed_relations:
                                current_term.parents.append((target_id, rel_type))

        if current_term is not None and current_term.go_id:
            self.terms[current_term.go_id] = current_term

    def resolve_term(self, go_id: str) -> Optional[str]:
        """Resolve a GO ID, handling obsolete terms with single unambiguous replaced_by."""
        term = self.terms.get(go_id)
        if term is None:
            return None
        if not term.is_obsolete:
            return go_id
        if len(term.replaced_by) == 1:
            replacement = term.replaced_by[0]
            rep_term = self.terms.get(replacement)
            if rep_term is not None and not rep_term.is_obsolete:
                return replacement
        return None

    def get_ancestors(self, go_id: str, same_aspect_only: bool = True) -> Set[str]:
        """Return transitive closure of valid ancestors for a GO term (preventing cycles)."""
        resolved = self.resolve_term(go_id)
        if resolved is None:
            return set()
        root_aspect = self.terms[resolved].aspect
        visited: Set[str] = set()
        stack: List[str] = [resolved]
        while stack:
            curr = stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            term = self.terms.get(curr)
            if term is None:
                continue
            for parent_id, _ in term.parents:
                p_resolved = self.resolve_term(parent_id)
                if p_resolved is None or p_resolved in visited:
                    continue
                if same_aspect_only and self.terms[p_resolved].aspect != root_aspect:
                    continue
                stack.append(p_resolved)
        return visited

    def build_aspect_vocabulary(self, aspect: str, observed_terms: Optional[Set[str]] = None) -> Dict[str, int]:
        """Build deterministic lexicographically sorted GO-ID -> index mapping for an aspect."""
        candidates: Set[str] = set()
        for go_id, term in self.terms.items():
            if not term.is_obsolete and term.aspect == aspect:
                if observed_terms is None or go_id in observed_terms:
                    candidates.add(go_id)
        sorted_terms = sorted(candidates)
        return {go_id: idx for idx, go_id in enumerate(sorted_terms)}

    def build_go_edges(self, vocab: Dict[str, int], aspect: str) -> np.ndarray:
        """Build parent-to-child edge index array of shape (E, 2) where col 0 = parent_idx, col 1 = child_idx."""
        edges: List[Tuple[int, int]] = []
        for child_go, child_idx in vocab.items():
            term = self.terms.get(child_go)
            if term is None or term.aspect != aspect:
                continue
            for parent_go, _ in term.parents:
                p_resolved = self.resolve_term(parent_go)
                if p_resolved in vocab:
                    parent_idx = vocab[p_resolved]
                    if parent_idx != child_idx:
                        edges.append((parent_idx, child_idx))
        edges = sorted(set(edges))
        if not edges:
            return np.zeros((0, 2), dtype=np.int64)
        return np.array(edges, dtype=np.int64)


def build_sparse_label_matrix(
    protein_ids: List[str],
    annotations: Dict[str, Set[str]],
    vocab: Dict[str, int],
) -> sparse.csr_matrix:
    """Build sparse CSR label matrix of shape (N, K) aligned with protein_ids and vocab."""
    rows: List[int] = []
    cols: List[int] = []
    for row_idx, pid in enumerate(protein_ids):
        terms = annotations.get(pid, set())
        for go_id in terms:
            if go_id in vocab:
                rows.append(row_idx)
                cols.append(vocab[go_id])
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(protein_ids), len(vocab)), dtype=np.float32)
