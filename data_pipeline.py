import cv2
import numpy as np
import json
import os

class DataPipeline:
    """
    Data Preparation & Feature Engineering
    - Frame resizing
    - Normalization
    - Feature extraction
    """
    def __init__(self, config):
        self.width = config["video"]["width"]
        self.height = config["video"]["height"]
        self.frame_skip = config["video"]["frame_skip"]

    def preprocess_frame(self, frame):
        """
        Resize and normalize frame.
        """
        resized = cv2.resize(frame, (self.width, self.height))
        # Additional preprocessing like noise reduction could be added here
        # e.g., blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        return resized

    def extract_features(self, agent_state):
        """
        Feature extraction: Vehicle count per lane, density tier, emergency flag.
        """
        features = {
            "lane1_count": agent_state.get("lane1_count", 0),
            "lane2_count": agent_state.get("lane2_count", 0),
            "lane1_density_tier": agent_state.get("tier_lane1", 0),
            "lane2_density_tier": agent_state.get("tier_lane2", 0),
            "emergency_present": 1 if agent_state.get("emergency_flag") else 0
        }
        
        # Feature engineering: Time-of-day weighting (mocked for this project)
        # e.g., rush hour gets higher priority
        # features["rush_hour_weight"] = 1.5
        
        return features
