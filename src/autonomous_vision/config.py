from pathlib import Path
from typing import ClassVar

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Configuration class for autonomous vision project."""

    # Get the project root directory (two levels up from this config file)
    project_root: ClassVar[Path] = Path(__file__).parent.parent.parent

    # Raw BDD100K JSON files
    train_labels: Path = (
        project_root / "data/raw_bdd_jsons/bdd100k_labels_images_train.json"
    )
    val_labels: Path = (
        project_root / "data/raw_bdd_jsons/bdd100k_labels_images_val.json"
    )

    # YOLO training data
    train_images: Path = project_root / "data/yolo_data/images/train"
    val_images: Path = project_root / "data/yolo_data/images/val"
    train_labels_yolo: Path = project_root / "data/yolo_data/labels/train"
    val_labels_yolo: Path = project_root / "data/yolo_data/labels/val"
    dataset_yaml: Path = project_root / "data/yolo_data/dataset.yaml"

    # Parsed data (CSV/Parquet)
    parsed_data: Path = project_root / "data/parsed_data"

    # COCO format data
    coco_data: Path = project_root / "data/coco_data"

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

    # COCO JSON files
    train_json: Path = project_root / "data/coco_data/bdd100k_train_coco.json"
    val_json: Path = project_root / "data/coco_data/bdd100k_val_coco.json"

    # Training output
    project: Path = project_root / "runs/train"
    run_name: str = "yolo11m_30epochs"

    # Model & schedule
    model_name: str = "yolo11m.pt"
    imgsz: int = 640
    epochs: int = 50
    batch: int = -1  # auto-batch to VRAM
    device: int = 0

    # Data loading / geometry
    rect: bool = True

    # Reproducibility
    seed: int = 42


Config = Config()
