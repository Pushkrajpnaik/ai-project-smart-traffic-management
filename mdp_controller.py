def get_next_signal(state):
    """Choose the next green lane and duration based on the current state."""
    if state.get("emergency_lane"):
        green_lane = state["emergency_lane"]
        density = state.get(f"density_{green_lane}", 0.0)
        duration = max(8, min(20, int(12 + density * 20)))
        return {"green_lane": green_lane, "duration": duration, "reason": "EMERGENCY"}

    lane_priority = "lane1"
    if state.get("density_lane2", 0.0) > state.get("density_lane1", 0.0):
        lane_priority = "lane2"

    green_lane = lane_priority
    density = state.get(f"density_{green_lane}", 0.0)
    duration = max(6, min(18, int(8 + density * 25)))
    return {"green_lane": green_lane, "duration": duration, "reason": "DENSITY"}
