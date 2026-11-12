import json
import os
import sys
import tempfile
from pathlib import Path

import requests

URL = "https://psi.nasa.gov/geode-py/ws/repo/search"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "psi_combustion.json"


def fetch():
    try:
        response = requests.get(
            URL,
            params={"q": "", "data_source": "psi", "data_type": "investigation", "size": 200},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()["hits"]
        hits = payload["hits"]
        total = payload["total"]
        total = total["value"] if isinstance(total, dict) else total
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Could not fetch PSI investigations: {exc}") from exc
    if not isinstance(hits, list):
        raise RuntimeError("PSI response does not contain a list of hits.")
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        raise RuntimeError("PSI response has an invalid hit count.")
    if len(hits) < total:
        raise RuntimeError(f"PSI returned {len(hits)} of {total} hits; data was not saved.")
    investigations = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        source = hit.get("_source")
        if isinstance(source, dict) and "Combustion" in str(source.get("Research Area", "")):
            investigations.append(source)
    _save(investigations)
    return len(investigations)


def _save(investigations):
    temp_path = None
    try:
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=DATA_PATH.parent, delete=False
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(investigations, handle, indent=1, ensure_ascii=False)
        os.replace(temp_path, DATA_PATH)
    except OSError as exc:
        raise RuntimeError(f"Could not save PSI investigations: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        count = fetch()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    sys.stdout.write(f"Saved {count} combustion investigations to {DATA_PATH}\n")
