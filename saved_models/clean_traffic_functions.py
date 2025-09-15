
# Clean Traffic Analysis Functions
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import yaml

def process_detections(results, conf_threshold=0.25):
    counts = {'mobil': 0, 'motor': 0, 'total': 0}
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf)
            if conf < conf_threshold:
                continue
            cls = int(box.cls)
            class_name = ['mobil', 'motor'][cls]
            counts[class_name] += 1
            counts['total'] += 1
    return counts

def calculate_green_time(car_count, motorcycle_count, base_time=10, time_per_car=6, time_per_motorcycle=5, min_time=15, max_time=60):
    green_time = base_time + (car_count * time_per_car) + (motorcycle_count * time_per_motorcycle)
    green_time = max(min_time, min(green_time, max_time))
    return green_time

def process_image(image_path, model, conf_threshold=0.25):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    results = model.predict(img, conf=conf_threshold)
    counts = process_detections(results, conf_threshold)
    return counts

# Usage:
# model = YOLO('saved_models/clean_traffic_model.pt')
# counts = process_image('image.jpg', model)
# green_time = calculate_green_time(counts['mobil'], counts['motor'])
