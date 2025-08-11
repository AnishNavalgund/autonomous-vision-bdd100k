import json
from pathlib import Path
from typing import Dict, Iterable


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def scan_images(images_dir: Path) -> Dict[str, Path]:
    """
    Build a filename
    Data structure: dict[str, Path]
    """
    idx: Dict[str, Path] = {}
    for p in images_dir.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            idx[p.name] = p
    return idx


def load_json_records(path: Path) -> Iterable[dict]:
    """
    Supports:
      - .json
    Outputs dicts (one per image record).
    """
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        yield from data
    else:
        yield from data.get("images", [])
