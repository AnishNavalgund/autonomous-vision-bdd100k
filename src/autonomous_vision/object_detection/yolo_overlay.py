from pathlib import Path

import cv2
from ultralytics import YOLO

from autonomous_vision.config import Config

MODEL_PATH = "./models/yolo11s_bdd.pt"
VAL_IMAGES_DIR = Config.val_images
VAL_LABELS_DIR = Config.val_labels_yolo
OUTPUT_DIR = "./outputs/viz_yolo11_today"
IMG_SIZE = (1280, 720)  # Original image size
N_IMAGES = 10

model = YOLO(MODEL_PATH)
class_names = model.names


def draw_boxes(image, boxes, color, labels=None, scores=None):
    """Draw bounding boxes on image with optional labels and scores."""
    for i, box in enumerate(boxes):
        box_x1, box_y1, box_x2, box_y2 = map(int, box)
        cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), color, 2)
        text = ""
        if labels is not None:
            cls_id = int(labels[i])
            name = class_names.get(cls_id, str(cls_id))
            text += name
        if scores is not None:
            text += f" {scores[i]:.2f}"
        if text:
            cv2.putText(
                image,
                text,
                (box_x1, box_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
    return image


output_dir_path = Path(OUTPUT_DIR)
output_dir_path.mkdir(parents=True, exist_ok=True)
image_paths = sorted(Path(VAL_IMAGES_DIR).glob("*.jpg"))[:N_IMAGES]

print("Starting YOLO overlay generation...")
print(f"Model: {MODEL_PATH}")
print(f"Processing {N_IMAGES} images...")

for img_path in image_paths:
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, IMG_SIZE)

    # Load ground truth
    label_file = Path(VAL_LABELS_DIR) / (img_path.stem + ".txt")
    gt_boxes, gt_classes = [], []
    if label_file.exists():
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                cls, x, y, bw, bh = map(float, line.strip().split())
                gt_x1 = int((x - bw / 2) * IMG_SIZE[0])
                gt_y1 = int((y - bh / 2) * IMG_SIZE[1])
                gt_x2 = int((x + bw / 2) * IMG_SIZE[0])
                gt_y2 = int((y + bh / 2) * IMG_SIZE[1])
                gt_boxes.append([gt_x1, gt_y1, gt_x2, gt_y2])
                gt_classes.append(int(cls))

    # Run YOLO prediction
    results = model.predict(
        str(img_path), imgsz=IMG_SIZE[0], conf=0.25, iou=0.45, verbose=True
    )
    for r in results:
        pred_boxes = r.boxes.xyxy.cpu().numpy()
        pred_classes = r.boxes.cls.cpu().numpy()
        pred_scores = r.boxes.conf.cpu().numpy()

    img = draw_boxes(img, gt_boxes, (0, 0, 255), gt_classes)  # RED = GT
    img = draw_boxes(
        img, pred_boxes, (0, 255, 0), pred_classes, pred_scores
    )  # GREEN = Pred

    out_path = output_dir_path / f"{img_path.stem}_viz.jpg"
    cv2.imwrite(str(out_path), img)

print(f"Saved overlays to: {output_dir_path}")
