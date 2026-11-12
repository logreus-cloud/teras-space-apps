FUEL_PHASES = {"solid", "liquid", "gas", "mixed", "unknown"}
GRAVITY_LEVELS = {"microgravity", "partial", "normal", "mixed", "unknown"}
FIRE_AREAS = {"prevention", "detection", "suppression"}


def normalize_rank(value, known_accessions) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("Rank response must be a JSON array.")
    known = set(known_accessions)
    rows = []
    for item in value:
        if not isinstance(item, dict):
            continue
        accession = item.get("accession")
        if not isinstance(accession, str) or accession not in known:
            continue
        try:
            score = int(float(item.get("score", 0)))
        except (TypeError, ValueError, OverflowError):
            score = 0
        rows.append({
            "accession": accession,
            "score": max(0, min(100, score)),
            "reason": str(item.get("reason") or ""),
        })
    if not rows:
        raise ValueError("Rank response contains no known investigations.")
    return rows


def _text(value):
    result = str(value).strip() if value is not None else ""
    return result or "unknown"


def _string_list(value):
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def normalize_attributes(value) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Attributes response must be a JSON object.")
    phase = _text(value.get("fuel_phase")).lower()
    gravity = _text(value.get("gravity")).lower()
    areas = [area.lower() for area in _string_list(value.get("fire_safety_area"))]
    return {
        "fuel_phase": phase if phase in FUEL_PHASES else "unknown",
        "fuels": _string_list(value.get("fuels")),
        "sample_geometry": _text(value.get("sample_geometry")),
        "oxygen_conditions": _text(value.get("oxygen_conditions")),
        "flow_conditions": _text(value.get("flow_conditions")),
        "gravity": gravity if gravity in GRAVITY_LEVELS else "unknown",
        "measurements": _string_list(value.get("measurements")),
        "fire_safety_area": [area for area in areas if area in FIRE_AREAS],
        "key_findings": _string_list(value.get("key_findings"))[:3],
    }
