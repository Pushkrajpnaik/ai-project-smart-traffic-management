import numpy as np
from ultralytics import YOLO

DETECTED_CLASSES = {"car", "bus", "truck", "motorcycle", "ambulance", "fire truck"}
EMERGENCY_CLASSES = {"ambulance", "fire truck"}
MODEL_PATH = "yolov8n.pt"


class VehicleDetector:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model = YOLO(model_path)

    def detect(self, frame):
        """Return a list of vehicle detections for the provided frame."""
        results = self.model(frame, verbose=False)[0]
        detections = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for box, cls in zip(boxes, classes):
            label = self.model.names.get(int(cls), str(int(cls)))
            if label not in DETECTED_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
            })

        return detections


def is_emergency_label(label: str) -> bool:
    return label in EMERGENCY_CLASSES
