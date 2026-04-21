class SignalController:
    def __init__(self, config: dict):
        self.yellow_time = config["timing"]["yellow_time"]
        self.min_green = config["timing"]["min_green"]
        self.max_green = config["timing"]["max_green"]
        self.emergency_green = config["timing"]["emergency_green"]
        
        self.current_green_lane = "lane1"
        self.state = "GREEN" # GREEN or YELLOW
        self.timer = self.min_green

    def update(self, action: dict):
        """
        action = {"green_lane": "lane1", "duration": 30, "reason": "DENSITY"}
        Returns the current visual state for UI.
        """
        # Decrement timer
        self.timer -= 1
        
        if action.get("reason") == "EMERGENCY":
            # Override immediately
            self.current_green_lane = action["green_lane"]
            self.state = "GREEN"
            self.timer = action["duration"]
        elif self.timer <= 0:
            if self.state == "GREEN":
                # Transition to yellow
                self.state = "YELLOW"
                self.timer = self.yellow_time
            else:
                # Transition to next green
                self.state = "GREEN"
                self.current_green_lane = action["green_lane"]
                self.timer = action["duration"]

        return {
            "green_lane": self.current_green_lane,
            "state": self.state,
            "timer": self.timer
        }
