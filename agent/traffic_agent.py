import time
from vision.detector import VehicleDetector
from vision.tracker import Tracker
from vision.density_estimator import DensityEstimator
from agent.mdp_controller import MDPController
from utils.signal_controller import SignalController
from utils.emergency_handler import EmergencyHandler

class TrafficAgent:
    """
    Autonomous agent that manages traffic signals using a
    Perceive → Reason → Act → Reflect loop.
    """
    def __init__(self, config):
        self.detector = VehicleDetector(config)
        self.tracker = Tracker()
        self.density_estimator = DensityEstimator(config)
        self.mdp = MDPController(config)
        self.signal = SignalController(config)
        self.emergency_handler = EmergencyHandler(config)
        self.memory = []  # stores (state, action, reward) history

    def perceive(self, frame):
        """Tool: detect + track + estimate density + check emergency"""
        # Detect
        raw_detections = self.detector.detect(frame)
        det_boxes = [d["bbox"] for d in raw_detections]
        labels = [d["label"] for d in raw_detections]
        
        # Track
        tracked_objects = self.tracker.update(det_boxes)
        
        # Determine lanes and labels
        labels_dict = {}
        lane_counts = {"lane1": set(), "lane2": set()}
        
        for i, obj in enumerate(tracked_objects):
            # obj = [x1, y1, x2, y2, id]
            x1, y1, x2, y2, obj_id = obj
            
            # Use bottom center of bounding box for more accurate lane detection
            # as the base of the vehicle is what touches the road/polygon
            cx, cy_bottom = (x1 + x2) // 2, y2
            
            label = labels[i] if i < len(labels) else "unknown"
            labels_dict[obj_id] = label
            
            # Determine lane
            lane_id = None
            if self.density_estimator.is_in_lane((cx, cy_bottom), "lane1"):
                lane_counts["lane1"].add(obj_id)
                lane_id = "lane1"
            elif self.density_estimator.is_in_lane((cx, cy_bottom), "lane2"):
                lane_counts["lane2"].add(obj_id)
                lane_id = "lane2"
                
            # Append lane_id to tracked object for emergency check
            tracked_objects[i].append(lane_id)
            
        # Estimate density
        density_data = self.density_estimator.estimate_density(len(lane_counts["lane1"]), len(lane_counts["lane2"]))
        
        # Check emergency
        has_emergency, emergency_lane = self.emergency_handler.check_emergency(tracked_objects, labels_dict)
        
        state = {
            "lane1_density": density_data["lane1"],
            "lane2_density": density_data["lane2"],
            "tier_lane1": density_data["tier_lane1"],
            "tier_lane2": density_data["tier_lane2"],
            "lane1_count": len(lane_counts["lane1"]),
            "lane2_count": len(lane_counts["lane2"]),
            "emergency_flag": has_emergency,
            "emergency_lane": emergency_lane,
            "tracked_objects": tracked_objects,
            "labels_dict": labels_dict
        }
        return state

    def reason(self, state):
        """MDP policy lookup or calculation"""
        action = self.mdp.get_action(state)
        return action

    def act(self, action):
        """Set signal timings"""
        signal_state = self.signal.update(action)
        return signal_state

    def compute_reward(self, state):
        """Calculate reward: -(total_vehicles_waiting)"""
        # Negative because we want to minimize waiting
        waiting_vehicles = state["lane1_count"] + state["lane2_count"]
        return -waiting_vehicles

    def reflect(self, state, action, reward):
        """Log decision for analysis"""
        log_entry = {
            "timestamp": time.time(),
            "state": {k: v for k, v in state.items() if k not in ["tracked_objects", "labels_dict"]},
            "action": action,
            "reward": reward
        }
        self.memory.append(log_entry)
        return log_entry
