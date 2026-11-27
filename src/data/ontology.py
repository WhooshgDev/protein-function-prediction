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
    alt_ids: List[str] = field(default_factory=list)
    parents: List[Tuple[str, str]] = field(default_factory=list)  # (parent_go_id, relation_type)


class GOOntology:
    """GO DAG representation supporting parsing, obsolete term resolution, vocabulary, and ancestor propagation."""

    def __init__(self, allowed_relations: Tuple[str, ...] = ("is_a", "part_of")):
        self.allowed_relations = set(allowed_relations)
        self.terms: Dict[str, GOTerm] = {}
        self.alt_id_map: Dict[str, str] = {}
        self.data_version: str = ""
        self._ancestor_cache: Dict[str, frozenset] = {}

    def parse_obo(self, obo_path: Union[str, Path]) -> None:
        """Parse go-basic.obo file into GOTerm records."""
        current_term: Optional[GOTerm] = None

        with open(obo_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if current_term is None and line.startswith("data-version:"):
                    self.data_version = line.split(":", 1)[1].strip()
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
                    elif key == "alt_id":
                        current_term.alt_ids.append(val)
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
        for go_id, term in self.terms.items():
            for alt in term.alt_ids:
                if alt not in self.terms:
                    self.alt_id_map[alt] = go_id
        self._ancestor_cache = {}

    def resolve_term(self, go_id: str) -> Optional[str]:
        """Resolve a GO ID, handling obsolete terms with single unambiguous replaced_by."""
        term = self.terms.get(go_id)
        if term is None and go_id in self.alt_id_map:
            go_id = self.alt_id_map[go_id]
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
        if same_aspect_only and resolved in self._ancestor_cache:
            return set(self._ancestor_cache[resolved])
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
        if same_aspect_only:
            self._ancestor_cache[resolved] = frozenset(visited)
        return visited

    def propagate(self, go_ids: Set[str]) -> Set[str]:
        """Close a set of GO IDs under the allowed relations (same-aspect ancestors). Idempotent."""
        out: Set[str] = set()
        for go_id in go_ids:
            out |= self.get_ancestors(go_id)
        return out

    def find_cycles(self) -> int:
        """Return number of nodes participating in a cycle of allowed relations (0 for a valid DAG)."""
        import networkx as nx

        g = nx.DiGraph()
        for go_id, term in self.terms.items():
            if term.is_obsolete:
                continue
            for parent_id, _ in term.parents:
                g.add_edge(go_id, parent_id)
        if nx.is_directed_acyclic_graph(g):
            return 0
        return sum(len(c) for c in nx.strongly_connected_components(g) if len(c) > 1)

    def build_aspect_vocabulary(self, aspect: str, observed_terms: Optional[Set[str]] = None) -> Dict[str, int]:
        """Build deterministic lexicographically sorted GO-ID -> index mapping for an aspect."""
        candidates: Set[str] = set()
        for go_id, term in self.terms.items():
            if not term.is_obsolete and term.aspect == aspect:
                if observed_terms is None or go_id in observed_terms:
                    candidates.add(go_id)
        sorted_terms = sorted(candidates)
        return {go_id: idx for idx, go_id in enumerate(sorted_terms)}

    def build_go_edges(
        self, vocab: Dict[str, int], aspect: str, return_relations: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        """Build parent-to-child edge index array of shape (E, 2) where col 0 = parent_idx, col 1 = child_idx.

        Only direct edges whose both endpoints are in ``vocab`` are kept. With
        ``return_relations=True`` the relation type of each edge is returned as well.
        """
        edge_rel: Dict[Tuple[int, int], str] = {}
        for child_go, child_idx in vocab.items():
            term = self.terms.get(child_go)
            if term is None or term.aspect != aspect:
                continue
            for parent_go, rel in term.parents:
                p_resolved = self.resolve_term(parent_go)
                if p_resolved in vocab:
                    parent_idx = vocab[p_resolved]
                    if parent_idx != child_idx:
                        key = (parent_idx, child_idx)
                        # is_a wins over part_of if both exist for the same pair
                        if key not in edge_rel or rel == "is_a":
                            edge_rel[key] = rel
        keys = sorted(edge_rel)
        arr = np.array(keys, dtype=np.int64) if keys else np.zeros((0, 2), dtype=np.int64)
        if return_relations:
            return arr, [edge_rel[k] for k in keys]
        return arr


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
