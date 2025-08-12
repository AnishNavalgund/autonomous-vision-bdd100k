from pathlib import Path
from typing import Dict, List

import yaml

from autonomous_vision.utils.helper import coco_bbox_to_yolo_norm


def write_yolo_labels(
    images: Dict[int, dict],
    anns_by_image: Dict[int, List[dict]],
    labels_dir: Path,
) -> int:
    """Write YOLO format labels to files."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    n_files = 0

    for img_id, im in images.items():
        width, height = im["width"], im["height"]
        lines: List[str] = []

        for ann in anns_by_image.get(img_id, []):
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            cxn, cyn, wn, hn = coco_bbox_to_yolo_norm(
                x, y, w, h, width, height
            )
            cxn = min(max(cxn, 0.0), 1.0)
            cyn = min(max(cyn, 0.0), 1.0)
            wn = min(max(wn, 0.0), 1.0)
            hn = min(max(hn, 0.0), 1.0)

            lines.append(
                f"{ann['_cls']} {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f}"
            )

        out = labels_dir / (Path(im["file_name"]).stem + ".txt")
        out.write_text("\n".join(lines))
        n_files += 1

    return n_files


def create_empty_labels_for_unlabeled_images(
    unlabeled_list_path: Path, labels_dir: Path
) -> int:
    """Create empty label files for images in unlabeled list."""
    labels_dir.mkdir(parents=True, exist_ok=True)

    with open(unlabeled_list_path, "r", encoding="utf-8") as f:
        unlabeled_images = [line.strip() for line in f if line.strip()]

    created_count = 0
    for image_name in unlabeled_images:
        base_name = Path(image_name).stem
        label_path = labels_dir / (base_name + ".txt")

        with open(label_path, "w", encoding="utf-8") as f:
            pass  # Creates empty file

        created_count += 1

    return created_count


def make_yolo_yaml(
    train_images_dir: Path,
    val_images_dir: Path,
    names: List[str],
    out_yaml: Path,
) -> Path:
    """Create YOLO dataset YAML configuration."""
    data = {
        "path": ".",
        "train": str(train_images_dir.resolve()),
        "val": str(val_images_dir.resolve()),
        "names": dict(enumerate(names)),
        "autodownload": False,
    }

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(data, sort_keys=False))

    return out_yaml
