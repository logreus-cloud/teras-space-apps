import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "psi_combustion.json"
PSI_BASE = "https://psi.nasa.gov"


@dataclass(frozen=True)
class Investigation:
    accession: str
    acronym: str
    title: str
    project_type: str
    platform: str
    sub_area: str
    related: tuple[str, ...]
    start_year: int | None
    end_year: int | None
    center: str
    keywords: tuple[str, ...]
    objectives: str
    approach: str
    hypothesis: str
    impacts: str
    hardware_name: str
    hardware_link: str
    people: str
    publications: tuple[str, ...]
    thumbnail_url: str
    page_url: str

    @property
    def label(self) -> str:
        return f"{self.acronym} — {self.title}" if self.acronym else self.accession


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def split_multi(value) -> tuple[str, ...]:
    parts = re.split(r"\s{2,}", _text(value))
    return tuple(dict.fromkeys(part.strip() for part in parts if part.strip()))


def _year(value) -> int | None:
    match = re.fullmatch(r"\d{2}-\d{2}-(\d{4})", _text(value))
    return int(match.group(1)) if match else None


def parse_investigation(raw: dict) -> Investigation:
    raw = raw if isinstance(raw, dict) else {}
    hardware = raw.get("Experiment Hardware")
    hardware = hardware if isinstance(hardware, dict) else {}
    person = raw.get("Person")
    person = person if isinstance(person, dict) else {}
    accession = _text(raw.get("Accession"))
    thumb = _text(raw.get("Thumbnail"))
    thumbnail_url = ""
    if thumb.startswith(("http://", "https://")):
        thumbnail_url = thumb
    elif thumb.startswith("/"):
        thumbnail_url = f"{PSI_BASE}{thumb}"
    related = _text(raw.get("Related Investigations"))
    people = " ".join(filter(None, (
        _text(person.get("First Name")),
        _text(person.get("Middle Initials")),
        _text(person.get("Last Name")),
    )))
    return Investigation(
        accession=accession,
        acronym=_text(raw.get("Acronym")),
        title=_text(raw.get("Title")),
        project_type=_text(raw.get("Project Type")),
        platform=_text(raw.get("Flight Platform")),
        sub_area=_text(raw.get("SubResearch Area")),
        related=tuple(dict.fromkeys(related.split())) if related else (),
        start_year=_year(raw.get("Investigation Start Date")),
        end_year=_year(raw.get("Investigation End Date")),
        center=_text(raw.get("Monitoring Center")),
        keywords=split_multi(raw.get("Keywords")),
        objectives=_text(raw.get("Objectives")),
        approach=_text(raw.get("Approach")),
        hypothesis=_text(raw.get("Hypothesis")),
        impacts=_text(raw.get("Research Impacts/Earth Benefits")),
        hardware_name=_text(hardware.get("Name")),
        hardware_link=_text(hardware.get("Link")),
        people=people,
        publications=split_multi(raw.get("Publication Title")),
        thumbnail_url=thumbnail_url,
        page_url=f"{PSI_BASE}/physci/repo/data/investigations/{accession}",
    )


def load_investigations(path=DATA_PATH) -> tuple[Investigation, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return ()
    items = tuple(parse_investigation(item) for item in raw if isinstance(item, dict))

    def sort_key(inv):
        match = re.search(r"(\d+)$", inv.accession)
        return (int(match.group(1)) if match else float("inf"), inv.accession)

    return tuple(sorted(items, key=sort_key))


def to_dataframe(items) -> pd.DataFrame:
    rows = (
        {
            "accession": inv.accession,
            "acronym": inv.acronym,
            "title": inv.title,
            "project_type": inv.project_type,
            "platform": inv.platform,
            "hardware_name": inv.hardware_name,
            "start_year": inv.start_year,
            "end_year": inv.end_year,
            "keywords": ", ".join(inv.keywords),
        }
        for inv in items
    )
    return pd.DataFrame(rows, columns=[
        "accession", "acronym", "title", "project_type", "platform",
        "hardware_name", "start_year", "end_year", "keywords",
    ])


def full_text(inv: Investigation) -> str:
    fields = (
        ("Accession", inv.accession),
        ("Acronym", inv.acronym),
        ("Title", inv.title),
        ("Project type", inv.project_type),
        ("Platform", inv.platform),
        ("Sub-area", inv.sub_area),
        ("Related investigations", ", ".join(inv.related)),
        ("Start year", inv.start_year),
        ("End year", inv.end_year),
        ("Center", inv.center),
        ("Keywords", ", ".join(inv.keywords)),
        ("Objectives", inv.objectives),
        ("Approach", inv.approach),
        ("Hypothesis", inv.hypothesis),
        ("Impacts and Earth benefits", inv.impacts),
        ("Hardware", inv.hardware_name),
        ("Hardware link", inv.hardware_link),
        ("People", inv.people),
        ("Publications", "; ".join(inv.publications)),
    )
    return "\n\n".join(f"{name}: {value}" for name, value in fields if value)
