import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def scan_images(images_dir: Path) -> Dict[str, Path]:
    """Build a filename to path mapping for images.

    Returns:
        Dictionary mapping filename to full path
    """
    idx: Dict[str, Path] = {}
    for path in images_dir.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            idx[path.name] = path
    return idx


def load_json_records(path: Path) -> Iterable[dict]:
    """Load JSON records from file.

    Supports:
        - .json files
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


def coco_xywh_to_yolo(
    center_x, center_y, width, height, img_width, img_height
):
    """Convert COCO center format to YOLO normalized format."""
    return (
        center_x / img_width,
        center_y / img_height,
        width / img_width,
        height / img_height,
    )


def coco_bbox_to_yolo_norm(x, y, width, height, img_width, img_height):
    """Convert COCO bbox format to YOLO normalized format."""
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    return coco_xywh_to_yolo(
        center_x, center_y, width, height, img_width, img_height
    )


def load_val_parquet(parquet_path: Path) -> pd.DataFrame:
    """
    Load BDD100K validation metadata parquet file.

    Args:
        parquet_path (Path): Path to val_data.parquet

    Returns:
        pd.DataFrame: Data with columns like scene, weather, time, etc.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} does not exist.")

    df = pd.read_parquet(parquet_path)
    return df
