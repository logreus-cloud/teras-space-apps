import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/psi_combustion.json"
RAW = ROOT / "data/raw"
BASE = "https://psi.nasa.gov/geode-py/ws/repo"


def _safe_name(name):
    name = Path(str(name).replace("\\", "/")).name.replace("..", "")
    if not name or name in (".", ".."):
        raise ValueError("Invalid file name")
    return name


def _download_table(acc, entry, force):
    name = _safe_name(entry["file_name"])
    target = RAW / name
    if target.exists() and not force:
        return "skipped"
    response = requests.get(
        f"https://psi.nasa.gov/geode-py/ws/studies/{acc}/download",
        params={"file": entry["file_name"], "version": int(entry["version"]), "redirect": "false"},
        timeout=30,
    )
    response.raise_for_status()
    signed_url = response.text.strip()
    parsed = urlparse(signed_url)
    if parsed.scheme != "https" or not (parsed.hostname or "").endswith(".amazonaws.com"):
        raise ValueError("Invalid signed download URL")
    RAW.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with requests.get(signed_url, timeout=60, stream=True, allow_redirects=False) as download:
            download.raise_for_status()
            with tempfile.NamedTemporaryFile(dir=RAW, delete=False) as temp:
                temp_path = Path(temp.name)
                for chunk in download.iter_content(chunk_size=65536):
                    if chunk:
                        temp.write(chunk)
        os.replace(temp_path, target)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return "downloaded"


def _accessions():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    records = data.get("investigations", []) if isinstance(data, dict) else data
    return sorted({row["Accession"] for row in records if row.get("Accession")})


def main():
    parser = argparse.ArgumentParser(description="Fetch PSI experimental tables")
    parser.add_argument("--force", action="store_true", help="Replace existing files")
    args = parser.parse_args()
    counts = {"downloaded": 0, "skipped": 0, "errors": 0}
    for acc in _accessions():
        try:
            response = requests.get(f"{BASE}/investigations/{acc}", timeout=30)
            response.raise_for_status()
            tables = response.json().get("experimentalTable") or []
        except (requests.RequestException, ValueError, AttributeError) as exc:
            counts["errors"] += 1
            sys.stderr.write(f"{acc}: {exc}\n")
            continue
        for entry in tables:
            try:
                counts[_download_table(acc, entry, args.force)] += 1
            except (requests.RequestException, ValueError, KeyError, OSError, TypeError) as exc:
                counts["errors"] += 1
                sys.stderr.write(f"{acc} / {entry}: {exc}\n")
    sys.stdout.write(
        f"Downloaded: {counts['downloaded']}, skipped: {counts['skipped']}, errors: {counts['errors']}\n"
    )


if __name__ == "__main__":
    main()
