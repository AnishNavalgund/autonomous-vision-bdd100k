from pathlib import Path
from typing import List

import pandas as pd

from autonomous_vision.config import Config
from autonomous_vision.utils.helper import ensure_dir, load_json_records
from autonomous_vision.utils.schemas import ImageAnnotation

ALLOWED_CLASSES = Config.detection_classes


def parse_split(labels_path: Path, split: str) -> List[dict]:
    """
    Parse one split (train/val/test):
      - Validate each image record with Pydantic
      - one object per row
    Returns:
      flat_rows: List of flattened object annotations
    """
    flat_rows: List[dict] = []

    for raw in load_json_records(labels_path):
        record = ImageAnnotation.model_validate(raw)

        w, h = Config.BDD100K_WIDTH, Config.BDD100K_HEIGHT

        # Flatten all object labels in this image
        scene = record.attributes.scene if record.attributes else None
        tod = record.attributes.timeofday if record.attributes else None
        weather = record.attributes.weather if record.attributes else None

        for obj in record.labels:
            if obj.category not in ALLOWED_CLASSES:
                continue

            # Skiping labels with no bbox
            if obj.box2d is None:
                continue

            # extract bbox
            x1, y1, x2, y2 = (
                obj.box2d.x1,
                obj.box2d.y1,
                obj.box2d.x2,
                obj.box2d.y2,
            )

            flat_rows.append(
                {
                    "image_name": record.name,
                    "split": split,
                    "label_id": getattr(obj, "id", None),
                    "category": obj.category,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": w,
                    "height": h,
                    "scene": scene,
                    "timeofday": tod,
                    "weather": weather,
                    "traffic_light_color": obj.attributes.get(
                        "trafficLightColor"
                    ),
                    "occluded": obj.attributes.get("occluded"),
                    "truncated": obj.attributes.get("truncated"),
                }
            )

    return flat_rows


def main():
    ensure_dir(Config.parsed_data)

    # Parse training split
    train_rows = parse_split(Config.train_labels, "train")

    # Parse val split
    val_rows = parse_split(Config.val_labels, "val")

    if not train_rows and not val_rows:
        return

    # separate files for train and val
    if train_rows:
        train_df = pd.DataFrame(train_rows)
        train_csv = Config.parsed_data / "train_data.csv"
        train_parquet = Config.parsed_data / "train_data.parquet"
        train_df.to_csv(train_csv, index=False)
        train_df.to_parquet(train_parquet)

    if val_rows:
        val_df = pd.DataFrame(val_rows)
        val_csv = Config.parsed_data / "val_data.csv"
        val_parquet = Config.parsed_data / "val_data.parquet"
        val_df.to_csv(val_csv, index=False)
        val_df.to_parquet(val_parquet)

    print(f"Total train annotations: {len(train_rows)}")
    print(f"Total val annotations: {len(val_rows)}")
    print(f"Total annotations: {len(train_rows) + len(val_rows)}")


if __name__ == "__main__":
    main()
