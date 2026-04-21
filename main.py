import cv2
import numpy as np
import yaml
import argparse
from agent.traffic_agent import TrafficAgent
from data_pipeline import DataPipeline
from evaluation import Evaluator

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_agentic_loop(video_source, config, demo=False):
    agent = TrafficAgent(config)
    pipeline = DataPipeline(config)
    evaluator = Evaluator(output_dir="results")

    cap = cv2.VideoCapture(video_source)
    count = 0
    frame_skip = config["video"]["frame_skip"]

    # Drawing areas
    area1 = config["lanes"]["lane1"]["polygon"]
    area2 = config["lanes"]["lane2"]["polygon"]

    def show_coordinates(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            print(f"Mouse Coordinates: x={x}, y={y}")

    cv2.namedWindow("Smart Traffic Management")
    cv2.setMouseCallback("Smart Traffic Management", show_coordinates)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % frame_skip != 0:
            continue

        # Data Prep
        frame = pipeline.preprocess_frame(frame)

        # Perceive
        state = agent.perceive(frame)
        
        # Feature Extraction
        features = pipeline.extract_features(state)
        
        # Reason
        action = agent.reason(state)
        
        # Act
        signal_state = agent.act(action)
        
        # Reflect
        reward = agent.compute_reward(state)
        log_entry = agent.reflect(state, action, reward)
        evaluator.add_record(log_entry)

        # ================= DISPLAY =================
        # Draw areas
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 255), 2)
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 255), 2)

        # Draw bounding boxes and ids
        for obj in state["tracked_objects"]:
            x1, y1, x2, y2, obj_id, lane_id = obj
            label = state["labels_dict"].get(obj_id, "unknown")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{obj_id}:{label}", (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
            cv2.circle(frame, (x2, y2), 4, (0, 255, 0), -1)

        # Density display
        cong1 = ["LOW", "MEDIUM", "HIGH"][state["tier_lane1"]]
        cong2 = ["LOW", "MEDIUM", "HIGH"][state["tier_lane2"]]
        
        cv2.putText(frame, f"Lane1: {state['lane1_count']} ({cong1})", (50, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(frame, f"Lane2: {state['lane2_count']} ({cong2})", (50, 90),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Signal Display
        color = (0, 255, 0) if signal_state["state"] == "GREEN" else (0, 255, 255)
        if signal_state["green_lane"] == "lane1":
            cv2.putText(frame, f"Lane1: {signal_state['state']} ({signal_state['timer']}s)", (50, 140),
                        cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            cv2.putText(frame, "Lane2: RED", (50, 180),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Lane2: {signal_state['state']} ({signal_state['timer']}s)", (50, 180),
                        cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            cv2.putText(frame, "Lane1: RED", (50, 140),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        # Emergency Display
        if state["emergency_flag"]:
            cv2.putText(frame, "EMERGENCY VEHICLE DETECTED", (250, 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

        cv2.imshow("Smart Traffic Management", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Generate Results
    evaluator.generate_metrics()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Smart Traffic Management System")
    parser.add_argument("--source", type=str, default="highway.mp4", help="Path to video source")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    args = parser.parse_args()

    config = load_config()
    
    if args.demo:
        print("Running in DEMO mode.")
        # If running demo, we could use a specific video or settings
        
    run_agentic_loop(args.source, config, args.demo)
