from detector import EMERGENCY_CLASSES


def detect_emergency(detections):
    """Return whether an emergency vehicle is present and the lane it occupies."""
    emergency_lanes = set()
    for det in detections:
        if det["label"] in EMERGENCY_CLASSES and "lane" in det:
            emergency_lanes.add(det["lane"])

    if not emergency_lanes:
        return False, None

    if "lane1" in emergency_lanes:
        return True, "lane1"
    return True, "lane2"
