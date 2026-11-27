# -*- coding: utf-8 -*-
"""Detect and count three target classes with YOLO; return image-level counts and inference durations."""
import json
import os
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = str(ROOT / "models" / "detector.pt")
IMGSZ = 640
CONF = 0.25
IOU = 0.70
CLASSES = ["overflow", "no_overflow", "waste"]


def main():
    args = sys.argv[1:]
    annotated_dir = None
    if "--annotated-dir" in args:
        k = args.index("--annotated-dir")
        annotated_dir = args[k + 1]
        del args[k:k + 2]
    if annotated_dir:
        os.makedirs(annotated_dir, exist_ok=True)
    img_dir, ids = args[0], args[1:]
    t0 = time.perf_counter()
    model = YOLO(WEIGHTS)
    load_time = time.perf_counter() - t0

    per_image, total = [], {c: 0 for c in CLASSES}
    for image_id in ids:
        path = os.path.join(img_dir, image_id + ".jpg")
        ts = time.perf_counter()
        res = model.predict(path, imgsz=IMGSZ, conf=CONF, iou=IOU,
                            device=os.environ.get("YOLO_DEVICE", "cpu"), verbose=False)[0]
        dt = time.perf_counter() - ts
        counts = {c: 0 for c in CLASSES}
        for cid in res.boxes.cls.tolist():
            counts[model.names[int(cid)]] += 1
        for c in CLASSES:
            total[c] += counts[c]
        annotated_path = ""
        if annotated_dir:
            annotated = res.plot()
            annotated_path = os.path.join(annotated_dir, image_id + ".jpg")
            cv2.imwrite(annotated_path, annotated)
        per_image.append({
            "image_id": image_id + ".jpg",
            "objects": [{"class": c, "count": counts[c]} for c in CLASSES],
            "annotated": annotated_path,
            "infer_seconds": round(dt, 4),
        })
    print(json.dumps({
        "params": {"imgsz": IMGSZ, "conf": CONF, "iou": IOU, "device": os.environ.get("YOLO_DEVICE", "cpu")},
        "weight_load_seconds": round(load_time, 4),
        "per_image": per_image,
        "total": total,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
