"""
BDD100K to COCO converter for object detection.
"""

import json
from pathlib import Path

from autonomous_vision.config import Config
from autonomous_vision.utils.helper import load_json_records


def create_coco_dataset(
    bdd_labels_path: Path, output_path: Path, split_name: str
) -> None:
    """
    Convert BDD100K labels to COCO format.

    Args:
        bdd_labels_path: Path to BDD100K JSON labels file
        output_path: Path to save the COCO JSON file
        split_name: Name of the split (train/val)
    """

    # Initialize COCO structure
    coco_data = {
        "info": {"description": f"BDD100K {split_name}"},
        "licenses": [{"id": 1, "name": "BDD100K"}],
        "categories": [
            {"id": i + 1, "name": class_name}
            for i, class_name in enumerate(Config.detection_classes)
        ],
        "images": [],
        "annotations": [],
    }

    # Create category name to ID mapping
    category_map = {
        name: i + 1 for i, name in enumerate(Config.detection_classes)
    }

    # Process each image
    annotation_id = 1
    for record in load_json_records(bdd_labels_path):
        image_name = record["name"]

        w, h = Config.BDD100K_WIDTH, Config.BDD100K_HEIGHT

        # Add image info
        image_info = {
            "id": len(coco_data["images"]) + 1,
            "file_name": image_name,
            "width": w,
            "height": h,
        }
        coco_data["images"].append(image_info)

        # Process labels
        for obj in record["labels"]:
            # Skip lane and drivable area
            if obj["category"] not in Config.detection_classes:
                continue

            # Skip labels without bbox
            if obj.get("box2d") is None:
                continue

            # Get bbox coordinates from BDD
            box2d = obj["box2d"]
            x1, y1, x2, y2 = box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]

            width = x2 - x1
            height = y2 - y1

            # Add annotation only for needed for obj det
            annotation = {
                "id": annotation_id,
                "image_id": image_info["id"],
                "category_id": category_map[obj["category"]],
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
            }
            coco_data["annotations"].append(annotation)
            annotation_id += 1

    # Save COCO json file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=1)


def main():
    """Convert both train and val splits to COCO format."""

    # Convert training split
    create_coco_dataset(
        bdd_labels_path=Config.train_labels,
        output_path=Config.coco_data / "bdd100k_train_coco.json",
        split_name="train",
    )

    # Convert validation split
    create_coco_dataset(
        bdd_labels_path=Config.val_labels,
        output_path=Config.coco_data / "bdd100k_val_coco.json",
        split_name="val",
    )


if __name__ == "__main__":
    main()
