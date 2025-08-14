"""
YOLO Training Pipeline for Autonomous Vision
"""

from pathlib import Path

from ultralytics import YOLO

from autonomous_vision.config import Config as C
from autonomous_vision.object_detection.data_loader import load_coco
from autonomous_vision.object_detection.label_utils import (
    create_empty_labels_for_unlabeled_images,
    make_yolo_yaml,
    write_yolo_labels,
)


def sanity_check_paths():
    """Check if all required paths exist."""
    paths = [C.train_images, C.val_images, C.train_json, C.val_json]
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Path not found: {path}")


def main():
    """Main training function."""
    sanity_check_paths()

    train_images_dir = Path(C.train_images)
    val_images_dir = Path(C.val_images)

    # Create YOLO-compatible label structure
    train_labels_dir = train_images_dir.parent.parent / "labels" / "train"
    val_labels_dir = val_images_dir.parent.parent / "labels" / "val"

    print("==> Converting COCO JSON to YOLO TXT labels (train)...")
    tr_images, tr_anns_by_img, names = load_coco(Path(C.train_json))
    n_tr = write_yolo_labels(tr_images, tr_anns_by_img, train_labels_dir)
    print(f"   Wrote {n_tr} label files -> {train_labels_dir}")

    print("==> Converting COCO JSON to YOLO TXT labels (val)...")
    va_images, va_anns_by_img, names_val = load_coco(Path(C.val_json))
    if names_val != names:
        raise RuntimeError("Train/Val category lists differ")
    n_va = write_yolo_labels(va_images, va_anns_by_img, val_labels_dir)
    print(f"   Wrote {n_va} label files -> {val_labels_dir}")

    # Create empty labels for unlabeled images to prevent background issue
    print("==> Creating empty labels for unlabeled images...")
    unlabeled_list_path = Path("data/lists/unlabeled_train.txt")
    if unlabeled_list_path.exists():
        n_empty = create_empty_labels_for_unlabeled_images(
            unlabeled_list_path, train_labels_dir
        )
        print(
            f"Created {n_empty} empty label files to prevent "
            f"background issue"
        )
    else:
        print("No unlabeled list found, skipping empty label creation")

    tmp_yaml = make_yolo_yaml(
        train_images_dir,
        val_images_dir,
        names,
        out_yaml=Path("data/yolo_data/dataset.yaml"),
    )
    print(f"==> Dataset YAML: {tmp_yaml}")

    print("==> Starting training ...")
    model = YOLO(C.model_name)

    model.train(
        model=C.model_name,
        data=str(tmp_yaml),
        imgsz=C.imgsz,
        epochs=C.epochs,
        batch=C.batch,
        device=C.device,
        workers=8,
        rect=C.rect,
        cos_lr=True,
        lr0=0.0025,
        lrf=0.01,
        optimizer="AdamW",
        amp=True,
        weight_decay=0.01,
        close_mosaic=4,
        patience=4,
        label_smoothing=0.01,
        overlap_mask=False,
        plots=True,
        save=True,
        project=C.project,
        name=C.run_name,
        verbose=True,
        seed=C.seed,
    )

    print("==> Validating best checkpoint...")
    model.val(
        data=str(tmp_yaml),
        save_json=True,
        plots=False,
        device=C.device,
        verbose=True,
    )

    print("Training complete. Check runs/train for curves & weights.")


if __name__ == "__main__":
    main()
