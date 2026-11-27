from pathlib import Path
from src.data.ontology import GOOntology, build_sparse_label_matrix


SAMPLE_OBO = """format-version: 1.2
[Term]
id: GO:0003674
name: molecular_function
namespace: molecular_function

[Term]
id: GO:0005488
name: binding
namespace: molecular_function
is_a: GO:0003674 ! molecular_function

[Term]
id: GO:0005515
name: protein binding
namespace: molecular_function
is_a: GO:0005488 ! binding

[Term]
id: GO:0099999
name: old binding term
namespace: molecular_function
is_obsolete: true
replaced_by: GO:0005515

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process
"""


def test_ontology_parsing_propagation_and_edges(tmp_path: Path):
    obo_path = tmp_path / "go-basic.obo"
    obo_path.write_text(SAMPLE_OBO, encoding="utf-8")

    onto = GOOntology()
    onto.parse_obo(obo_path)

    # Obsolete resolution
    assert onto.resolve_term("GO:0099999") == "GO:0005515"

    # Ancestor propagation (idempotent & aspect-isolated)
    ancestors = onto.get_ancestors("GO:0005515")
    assert ancestors == {"GO:0005515", "GO:0005488", "GO:0003674"}
    assert "GO:0008150" not in ancestors

    # Vocabulary & Parent->Child edges
    vocab_f = onto.build_aspect_vocabulary("F")
    assert list(vocab_f.keys()) == ["GO:0003674", "GO:0005488", "GO:0005515"]

    edges = onto.build_go_edges(vocab_f, "F")
    # (GO:0003674 -> GO:0005488) and (GO:0005488 -> GO:0005515)
    assert edges.shape == (2, 2)
    assert (edges[0, 0], edges[0, 1]) == (vocab_f["GO:0003674"], vocab_f["GO:0005488"])
    assert (edges[1, 0], edges[1, 1]) == (vocab_f["GO:0005488"], vocab_f["GO:0005515"])

    # Sparse matrix conversion preserves values
    mat = build_sparse_label_matrix(["P1", "P2"], {"P1": ancestors, "P2": {"GO:0003674"}}, vocab_f)
    dense = mat.toarray()
    assert dense.shape == (2, 3)
    assert dense[0].sum() == 3.0
    assert dense[1].sum() == 1.0


ALT_OBO = SAMPLE_OBO + """
[Term]
id: GO:0005634
name: nucleus
namespace: cellular_component
alt_id: GO:0000001
relationship: part_of GO:0005575 ! cellular_component

[Term]
id: GO:0005575
name: cellular_component
namespace: cellular_component
"""


def test_alt_id_relation_types_and_idempotent_propagation(tmp_path: Path):
    obo = tmp_path / "go.obo"
    obo.write_text(ALT_OBO, encoding="utf-8")
    onto = GOOntology()
    onto.parse_obo(obo)
    assert onto.resolve_term("GO:0000001") == "GO:0005634"
    assert onto.resolve_term("GO:1234567") is None
    vocab_c = onto.build_aspect_vocabulary("C")
    edges, rels = onto.build_go_edges(vocab_c, "C", return_relations=True)
    assert rels == ["part_of"]
    assert (edges[0, 0], edges[0, 1]) == (vocab_c["GO:0005575"], vocab_c["GO:0005634"])  # parent, child
    once = onto.propagate({"GO:0005515"})
    assert onto.propagate(once) == once
    assert onto.find_cycles() == 0


def test_relation_policy_excludes_part_of_when_configured(tmp_path: Path):
    obo = tmp_path / "go.obo"
    obo.write_text(ALT_OBO, encoding="utf-8")
    onto = GOOntology(allowed_relations=("is_a",))
    onto.parse_obo(obo)
    assert onto.get_ancestors("GO:0005634") == {"GO:0005634"}
