import numpy as np
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, config: dict):
        model_path = config.get("model", {}).get("path", "yolov8n.pt")
        self.model = YOLO(model_path)
        self.detected_classes = set(config.get("model", {}).get("detected_classes", ["car", "bus", "truck", "motorcycle", "ambulance", "fire truck"]))

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
            if label not in self.detected_classes:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
            })

        return detections
