import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_coco(
    coco_json: Path,
) -> Tuple[Dict[int, dict], Dict[int, list], List[str]]:
    """Load COCO format annotations and convert to internal format."""
    data = json.loads(coco_json.read_text())

    # Preserves 1..10 order
    cats = sorted(data["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]

    images = {
        im["id"]: {
            "file_name": im["file_name"],
            "width": im["width"],
            "height": im["height"],
        }
        for im in data["images"]
    }

    anns_by_image: Dict[int, List[dict]] = {k: [] for k in images.keys()}

    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        if ann["image_id"] not in anns_by_image:
            continue

        ann["_cls"] = cat_id_to_idx[ann["category_id"]]
        anns_by_image[ann["image_id"]].append(ann)

    return images, anns_by_image, names
