# pylint: disable=no-member
import os
import sys
from typing import Any, List, Optional

import cv2

IMG_DIR = "./data/yolo_data/images/test"
OUTPUT_VIDEO = "./outputs/videos/test.mp4"
FPS = 0.5

# make output directory
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

# Get sorted image list
images: List[str] = sorted(
    [img for img in os.listdir(IMG_DIR) if img.endswith(".jpg")]
)

print(f"{len(images)} images")

# get size
frame: Optional[Any] = cv2.imread(os.path.join(IMG_DIR, images[0]))
if frame is None:
    print(f"Could not read first image: {images[0]}")
    sys.exit(1)

h, w, _ = frame.shape
print(f"Image dim: {w}x{h}")

# VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"XVID")
TEMP_OUTPUT = OUTPUT_VIDEO.replace(".mp4", ".avi")
video_writer = cv2.VideoWriter(TEMP_OUTPUT, fourcc, FPS, (w, h))

if not video_writer.isOpened():
    print("Error: Could not open video writer")
    sys.exit(1)

print(f"Processing {len(images)} images...")
for i, img_name in enumerate(images):
    img_path = os.path.join(IMG_DIR, img_name)
    frame = cv2.imread(img_path)
    if frame is not None:
        video_writer.write(frame)
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(images)} images")
    else:
        print(f"Could not read image {img_name}")

video_writer.release()
print(f"Video saved as {TEMP_OUTPUT}")
