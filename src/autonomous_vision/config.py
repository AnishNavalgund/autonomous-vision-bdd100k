from pathlib import Path
from typing import ClassVar

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # Get the project root directory (two levels up from this config file)
    project_root: ClassVar[Path] = Path(__file__).parent.parent.parent

    train_labels: Path = (
        project_root / "data/labels/bdd100k_labels_images_train.json"
    )
    val_labels: Path = (
        project_root / "data/labels/bdd100k_labels_images_val.json"
    )

    train_images: Path = project_root / "data/images/train"
    val_images: Path = project_root / "data/images/val"

    parsed_data: Path = project_root / "results/parsed_data"
    coco_data: Path = project_root / "results/coco_data"

    detection_classes: list[str] = [
        "person",
        "rider",
        "car",
        "bus",
        "truck",
        "bike",
        "motor",
        "traffic light",
        "traffic sign",
        "train",
    ]

    # BDD100K dataset constants
    BDD100K_WIDTH: int = 1280
    BDD100K_HEIGHT: int = 720


Config = Config()
