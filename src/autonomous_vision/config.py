from pathlib import Path
from typing import ClassVar

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Configuration class for autonomous vision project."""

    # Get the project root directory (two levels up from this config file)
    project_root: ClassVar[Path] = Path(__file__).parent.parent.parent

    train_labels: Path = (
        project_root / "data/BDD_labels/bdd100k_labels_images_train.json"
    )
    val_labels: Path = (
        project_root / "data/BDD_labels/bdd100k_labels_images_val.json"
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

    train_json: Path = (
        project_root / "results/coco_data/bdd100k_train_coco.json"
    )
    val_json: Path = project_root / "results/coco_data/bdd100k_val_coco.json"

    # Training output
    project: Path = project_root / "runs/train"
    run_name: str = "yolo11s_NoFreeze"

    # Model & schedule
    model_name: str = "yolo11s.pt"
    imgsz: int = 640
    epochs: int = 20
    batch: int = -1  # auto-batch to VRAM
    device: int = 0

    # Data loading / geometry
    rect: bool = True

    # Reproducibility
    seed: int = 42


Config = Config()
