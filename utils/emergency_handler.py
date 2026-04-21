class EmergencyHandler:
    def __init__(self, config: dict):
        self.emergency_classes = set(config.get("model", {}).get("emergency_classes", ["ambulance", "fire truck"]))

    def check_emergency(self, tracked_objects, labels_dict):
        """
        Check if any of the tracked objects is an emergency vehicle.
        tracked_objects: list of [x1, y1, x2, y2, obj_id, lane_id]
        labels_dict: dict mapping obj_id to label
        Returns: (True/False, lane_id)
        """
        for obj in tracked_objects:
            obj_id = obj[4]
            lane_id = obj[5]
            label = labels_dict.get(obj_id, "")
            if label in self.emergency_classes and lane_id is not None:
                return True, lane_id
        return False, None
