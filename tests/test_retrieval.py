from teras.data import load_investigations
from teras.retrieval import SearchIndex


def test_droplet_extinguishment():
    index = SearchIndex.build(load_investigations())
    results = index.search("droplet extinguishment", top_k=3)
    assert any(inv.acronym in {"FLEX", "FLEX-2"} for inv, _ in results)


def test_smoke_detection():
    index = SearchIndex.build(load_investigations())
    results = index.search("smoke detection aerosol", top_k=3)
    assert any(inv.acronym in {"SAME", "SAME-R"} for inv, _ in results)


def test_empty_query():
    index = SearchIndex.build(load_investigations())
    assert index.search("the and in") == []
