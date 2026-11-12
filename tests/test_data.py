from teras.data import DATA_PATH, load_investigations, parse_investigation, split_multi


def test_split_multi():
    assert split_multi("FLEX     Droplet   FLEX  ") == ("FLEX", "Droplet")
    assert split_multi(None) == ()


def test_parse_minimal():
    inv = parse_investigation({"Accession": "PSI-69", "Acronym": "FLEX", "Title": "Flame"})
    assert inv.label == "FLEX — Flame"
    assert inv.related == ()
    assert inv.start_year is None
    assert inv.hardware_name == ""
    assert inv.thumbnail_url == ""


def test_parse_malformed_fields():
    inv = parse_investigation({
        "Accession": "PSI-2",
        "Experiment Hardware": "invalid",
        "Person": None,
        "Keywords": ["invalid"],
        "Investigation Start Date": 123,
        "Thumbnail": 42,
    })
    assert inv.hardware_name == ""
    assert inv.people == ""
    assert inv.keywords == ()
    assert inv.start_year is None
    assert inv.thumbnail_url == ""


def test_load_real_data():
    items = load_investigations(DATA_PATH)
    assert len(items) == 24
    assert any(inv.accession == "PSI-69" and inv.acronym == "FLEX" for inv in items)
    assert [int(inv.accession.split("-")[-1]) for inv in items] == sorted(
        int(inv.accession.split("-")[-1]) for inv in items
    )
