import cv2
import numpy as np

class DensityEstimator:
    def __init__(self, config: dict):
        self.lane1_polygon = np.array(config["lanes"]["lane1"]["polygon"], np.int32)
        self.lane2_polygon = np.array(config["lanes"]["lane2"]["polygon"], np.int32)
        self.lane1_capacity = config["lanes"]["lane1"]["capacity"]
        self.lane2_capacity = config["lanes"]["lane2"]["capacity"]
        self.low_thresh = config["thresholds"]["density"]["low"]
        self.medium_thresh = config["thresholds"]["density"]["medium"]

    def is_in_lane(self, pt, lane_id):
        polygon = self.lane1_polygon if lane_id == "lane1" else self.lane2_polygon
        return cv2.pointPolygonTest(polygon, pt, False) > 0

    def get_density_tier(self, density_value):
        if density_value < self.low_thresh:
            return 0 # Low
        elif density_value < self.medium_thresh:
            return 1 # Medium
        else:
            return 2 # High

    def estimate_density(self, count_lane1, count_lane2):
        d1 = count_lane1 / self.lane1_capacity
        d2 = count_lane2 / self.lane2_capacity
        return {
            "lane1": d1,
            "lane2": d2,
            "tier_lane1": self.get_density_tier(d1),
            "tier_lane2": self.get_density_tier(d2)
        }
