import cv2
import numpy as np


def compute_area(polygon):
    """Compute the polygon area in pixels."""
    return abs(cv2.contourArea(np.array(polygon, np.int32)))


def compute_density(vehicle_count, lane_area):
    """Return a simple density score for a lane."""
    if lane_area <= 0:
        return 0.0
    return vehicle_count / lane_area
