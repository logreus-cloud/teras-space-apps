import pytest

from teras.validation import normalize_attributes, normalize_rank


def test_normalize_rank_filters_and_clamps():
    value = [
        {"accession": "PSI-69", "score": "120.8", "reason": 7},
        {"accession": "PSI-10", "score": -4, "reason": None},
        {"accession": "PSI-999", "score": 50},
        "bad",
    ]
    assert normalize_rank(value, {"PSI-69", "PSI-10"}) == [
        {"accession": "PSI-69", "score": 100, "reason": "7"},
        {"accession": "PSI-10", "score": 0, "reason": ""},
    ]


@pytest.mark.parametrize(
    "value",
    [
        {"items": [{"accession": "PSI-69", "score": 80, "reason": "r"}]},
        {"rankings": [{"accession": "PSI-69", "score": 80, "reason": "r"}], "note": "x"},
        {"accession": "PSI-69", "score": 80, "reason": "r"},
    ],
)
def test_normalize_rank_unwraps_object_responses(value):
    assert normalize_rank(value, {"PSI-69"}) == [
        {"accession": "PSI-69", "score": 80, "reason": "r"}
    ]


@pytest.mark.parametrize(
    "value",
    [None, {}, [], [{"accession": "PSI-999"}], {"a": [], "b": []}],
)
def test_normalize_rank_rejects_empty_results(value):
    with pytest.raises(ValueError):
        normalize_rank(value, {"PSI-69"})


def test_normalize_attributes_coerces_and_filters():
    value = normalize_attributes({
        "fuel_phase": "PLASMA",
        "fuels": [" methane ", 3, ""],
        "sample_geometry": 42,
        "gravity": "MICROGRAVITY",
        "measurements": "temperature",
        "fire_safety_area": ["Detection", "other", None],
        "key_findings": ["one", "two", "three", "four"],
    })
    assert value == {
        "fuel_phase": "unknown",
        "fuels": ["methane"],
        "sample_geometry": "42",
        "oxygen_conditions": "unknown",
        "flow_conditions": "unknown",
        "gravity": "microgravity",
        "measurements": [],
        "fire_safety_area": ["detection"],
        "key_findings": ["one", "two", "three"],
    }


def test_normalize_attributes_requires_object():
    with pytest.raises(ValueError):
        normalize_attributes([])
