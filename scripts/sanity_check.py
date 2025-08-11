import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def sanity_check():
    """Perform sanity check across json, csv and parquet files after parsing"""

    # File paths for both train and val
    train_files = {
        "COCO": Path("results/coco_data/bdd100k_train_coco.json"),
        "CSV": Path("results/parsed_data/train_data.csv"),
        "Parquet": Path("results/parsed_data/train_data.parquet"),
    }

    val_files = {
        "COCO": Path("results/coco_data/bdd100k_val_coco.json"),
        "CSV": Path("results/parsed_data/val_data.csv"),
        "Parquet": Path("results/parsed_data/val_data.parquet"),
    }

    # Check if all files exist
    print("File Existence Check:")
    all_files_exist = True

    print("  Training data:")
    for format_name, file_path in train_files.items():
        if file_path.exists():
            print(f"{format_name}: {file_path.name}")
        else:
            print(f"Mismatch: {format_name}: {file_path.name} - MISSING!")
            all_files_exist = False

    print("  Validation data:")
    for format_name, file_path in val_files.items():
        if file_path.exists():
            print(f"{format_name}: {file_path.name}")
        else:
            print(f"Mismatch: {format_name}: {file_path.name} - MISSING!")
            all_files_exist = False

    if not all_files_exist:
        return False

    print("\n **************************** \n")

    # Load all data
    print("Loading data...")

    # Training data
    print("  Loading training data...")
    with open(train_files["COCO"], "r", encoding="utf-8") as f:
        train_coco = json.load(f)
    train_csv = pd.read_csv(train_files["CSV"])
    train_parquet = pd.read_parquet(train_files["Parquet"])

    # Validation data
    print("  Loading validation data...")
    with open(val_files["COCO"], "r", encoding="utf-8") as f:
        val_coco = json.load(f)
    val_csv = pd.read_csv(val_files["CSV"])
    val_parquet = pd.read_parquet(val_files["Parquet"])

    print("\n **************************** \n")

    # Image Count Check
    print("Image Count Check:")

    # Training data
    print("  Training data:")
    train_coco_images = len(train_coco["images"])
    train_csv_images = train_csv["image_name"].nunique()
    train_parquet_images = train_parquet["image_name"].nunique()

    print(f"COCO JSON: {train_coco_images:,} images")
    print(f"CSV: {train_csv_images:,} unique images")
    print(f"Parquet: {train_parquet_images:,} unique images")

    if train_coco_images == train_csv_images == train_parquet_images:
        print(f"All Match: {train_coco_images:,} images")
    else:
        print("Mismatch Detected!")
        return False

    # Validation data
    print("  Validation data:")
    val_coco_images = len(val_coco["images"])
    val_csv_images = val_csv["image_name"].nunique()
    val_parquet_images = val_parquet["image_name"].nunique()

    print(f"COCO JSON: {val_coco_images:,} images")
    print(f"CSV: {val_csv_images:,} unique images")
    print(f"Parquet: {val_parquet_images:,} unique images")

    if val_coco_images == val_csv_images == val_parquet_images:
        print(f"All Match: {val_coco_images:,} images")
    else:
        print("Mismatch Detected!")
        return False

    print("\n **************************** \n")

    # Annotation Count Check
    print("Annotation Count Check:")

    # Training data
    print("Training data:")
    train_coco_annotations = len(train_coco["annotations"])
    train_csv_annotations = len(train_csv)
    train_parquet_annotations = len(train_parquet)

    print(f"COCO JSON: {train_coco_annotations:,} annotations")
    print(f"CSV: {train_csv_annotations:,} annotations")
    print(f"Parquet: {train_parquet_annotations:,} annotations")

    if (
        train_coco_annotations
        == train_csv_annotations
        == train_parquet_annotations
    ):
        print(f"All Match: {train_coco_annotations:,} annotations")
    else:
        print("Mismatch Detected!")
        return False

    # Validation data
    print("Validation data:")
    val_coco_annotations = len(val_coco["annotations"])
    val_csv_annotations = len(val_csv)
    val_parquet_annotations = len(val_parquet)

    print(f"COCO JSON: {val_coco_annotations:,} annotations")
    print(f"CSV: {val_csv_annotations:,} annotations")
    print(f"Parquet: {val_parquet_annotations:,} annotations")

    if val_coco_annotations == val_csv_annotations == val_parquet_annotations:
        print(f"All Match: {val_coco_annotations:,} annotations")
    else:
        print("Mismatch Detected!")
        return False

    print("\n **************************** \n")

    # Class Distribution Check
    print("Class Distribution Check:")

    # Training data class counts
    print("Training data:")
    train_coco_class_counts = Counter()
    for ann in train_coco["annotations"]:
        category_id = ann["category_id"]
        category_name = train_coco["categories"][category_id - 1]["name"]
        train_coco_class_counts[category_name] += 1

    train_csv_class_counts = train_csv["category"].value_counts().to_dict()
    train_parquet_class_counts = (
        train_parquet["category"].value_counts().to_dict()
    )

    print("Class distribution comparison:")
    print(
        f"{'Class':<15} {'COCO':<10} {'CSV':<10} {'Parquet':<10} {'Status':<8}"
    )
    print(
        "=" * 15
        + " "
        + "=" * 10
        + " "
        + "=" * 10
        + " "
        + "=" * 10
        + " "
        + "=" * 8
    )

    train_all_classes = (
        set(train_coco_class_counts.keys())
        | set(train_csv_class_counts.keys())
        | set(train_parquet_class_counts.keys())
    )

    for class_name in sorted(train_all_classes):
        coco_count = train_coco_class_counts.get(class_name, 0)
        csv_count = train_csv_class_counts.get(class_name, 0)
        parquet_count = train_parquet_class_counts.get(class_name, 0)

        if coco_count == csv_count == parquet_count:
            status = "Match"
        else:
            status = "Mismatch"

        print(
            f"{class_name:<15} {coco_count:<10} {csv_count:<10} "
            f"{parquet_count:<10} {status:<8}"
        )

    # Verify training total counts match
    if (
        sum(train_coco_class_counts.values())
        == sum(train_csv_class_counts.values())
        == sum(train_parquet_class_counts.values())
    ):
        print(
            f"Total class counts match: "
            f"{sum(train_coco_class_counts.values()):,}"
        )
    else:
        print("Training total class count mismatch!")
        return False

    # Validation data class counts
    print("Validation data:")
    val_coco_class_counts = Counter()
    for ann in val_coco["annotations"]:
        category_id = ann["category_id"]
        category_name = val_coco["categories"][category_id - 1]["name"]
        val_coco_class_counts[category_name] += 1

    val_csv_class_counts = val_csv["category"].value_counts().to_dict()
    val_parquet_class_counts = val_parquet["category"].value_counts().to_dict()

    print("Class distribution comparison:")
    print(
        f"{'Class':<15} {'COCO':<10} {'CSV':<10} {'Parquet':<10} {'Status':<8}"
    )
    print(
        "=" * 15
        + " "
        + "=" * 10
        + " "
        + "=" * 10
        + " "
        + "=" * 10
        + " "
        + "=" * 8
    )

    val_all_classes = (
        set(val_coco_class_counts.keys())
        | set(val_csv_class_counts.keys())
        | set(val_parquet_class_counts.keys())
    )

    for class_name in sorted(val_all_classes):
        coco_count = val_coco_class_counts.get(class_name, 0)
        csv_count = val_csv_class_counts.get(class_name, 0)
        parquet_count = val_parquet_class_counts.get(class_name, 0)

        if coco_count == csv_count == parquet_count:
            status = "Match"
        else:
            status = "Mismatch"

        print(
            f"{class_name:<15} {coco_count:<10} {csv_count:<10} "
            f"{parquet_count:<10} {status:<8}"
        )

    # Verify validation total counts match
    if (
        sum(val_coco_class_counts.values())
        == sum(val_csv_class_counts.values())
        == sum(val_parquet_class_counts.values())
    ):
        print(
            f"Total class counts match: "
            f"{sum(val_coco_class_counts.values()):,}"
        )
    else:
        print("Validation total class count mismatch!")
        return False

    print("\n **************************** \n")

    print("Summary:")

    print("Training data:")
    print(f"Total images: {train_coco_images:,}")
    print(f"Total annotations: {train_coco_annotations:,}")
    print(f"Classes: {len(train_coco_class_counts)}")

    print("Validation data:")
    print(f"Total images: {val_coco_images:,}")
    print(f"Total annotations: {val_coco_annotations:,}")
    print(f"Classes: {len(val_coco_class_counts)}")

    print("\n Data is consistent across all formats.")
    return True


if __name__ == "__main__":

    SUCCESS = sanity_check()
    if not SUCCESS:
        print("Sanity check failed!")
        sys.exit(1)
