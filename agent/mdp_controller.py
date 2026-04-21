class MDPController:
    def __init__(self, config: dict):
        self.min_green = config["timing"]["min_green"]
        self.max_green = config["timing"]["max_green"]
        self.emergency_green = config["timing"]["emergency_green"]

    def get_action(self, state):
        """
        state = {
            "lane1_density": 0.5,
            "lane2_density": 0.8,
            "tier_lane1": 1,
            "tier_lane2": 2,
            "emergency_flag": True/False,
            "emergency_lane": "lane1" or "lane2" or None
        }
        """
        if state.get("emergency_flag") and state.get("emergency_lane"):
            return {
                "green_lane": state["emergency_lane"],
                "duration": self.emergency_green,
                "reason": "EMERGENCY"
            }

        # MDP Policy lookup or heuristic calculation based on density
        d1 = state.get("lane1_density", 0.0)
        d2 = state.get("lane2_density", 0.0)
        
        # Action space: durations based on density
        if d1 >= d2:
            chosen_lane = "lane1"
            max_d = d1
        else:
            chosen_lane = "lane2"
            max_d = d2

        # Simplified policy mapping density tier to duration
        # Low -> 15s, Med -> 30s, High -> 45s-60s
        tier = state.get(f"tier_{chosen_lane}", 0)
        if tier == 0:
            duration = self.min_green
        elif tier == 1:
            duration = 30
        else:
            duration = min(self.max_green, int(30 + max_d * 30))

        return {
            "green_lane": chosen_lane,
            "duration": duration,
            "reason": "DENSITY"
        }
