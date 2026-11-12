import hashlib
import json
import os
import tempfile
from pathlib import Path

from .data import DATA_PATH
from .llm import model_name

CACHE_PATH = DATA_PATH.parent / "cache"


def _path(kind, prompt) -> Path:
    source = f"{model_name()}\0{kind}\0{prompt}"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return CACHE_PATH / f"{digest}.json"


def get(kind, prompt):
    path = _path(kind, prompt)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def put(kind, prompt, value):
    path = _path(kind, prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False)
    temp = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False)
    temp_path = Path(temp.name)
    try:
        with temp:
            temp.write(payload)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
