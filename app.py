from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import yaml
import time
from agent.traffic_agent import TrafficAgent
from data_pipeline import DataPipeline

app = Flask(__name__)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Global state for the API
system_state = {
    "lane1_signal": "RED",
    "lane2_signal": "RED",
    "lane1_timer": 0,
    "lane2_timer": 0,
    "lane1_count": 0,
    "lane2_count": 0,
    "lane1_density": "LOW",
    "lane2_density": "LOW",
    "emergency": False
}

def generate_frames():
    global system_state
    agent = TrafficAgent(config)
    pipeline = DataPipeline(config)
    cap = cv2.VideoCapture(config["video"]["source"])
    frame_skip = config["video"]["frame_skip"]
    count = 0

    area1 = config["lanes"]["lane1"]["polygon"]
    area2 = config["lanes"]["lane2"]["polygon"]

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop the video if it ends
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        count += 1
        if count % frame_skip != 0:
            continue

        # Data Prep
        frame = pipeline.preprocess_frame(frame)

        # Perceive
        state = agent.perceive(frame)
        
        # Reason
        action = agent.reason(state)
        
        # Act
        signal_state = agent.act(action)

        # Update global state for the frontend dashboard
        system_state["lane1_count"] = state['lane1_count']
        system_state["lane2_count"] = state['lane2_count']
        system_state["lane1_density"] = ["LOW", "MEDIUM", "HIGH"][state["tier_lane1"]]
        system_state["lane2_density"] = ["LOW", "MEDIUM", "HIGH"][state["tier_lane2"]]
        system_state["emergency"] = state["emergency_flag"]

        # Signal Logic
        if signal_state["green_lane"] == "lane1":
            system_state["lane1_signal"] = signal_state["state"]
            system_state["lane1_timer"] = signal_state["timer"]
            system_state["lane2_signal"] = "RED"
            system_state["lane2_timer"] = signal_state["timer"]
        else:
            system_state["lane2_signal"] = signal_state["state"]
            system_state["lane2_timer"] = signal_state["timer"]
            system_state["lane1_signal"] = "RED"
            system_state["lane1_timer"] = signal_state["timer"]

        # Draw overlays on the frame for the video stream
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 255), 2)
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 255), 2)

        for obj in state["tracked_objects"]:
            x1, y1, x2, y2, obj_id, lane_id = obj
            label = state["labels_dict"].get(obj_id, "unknown")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{obj_id}:{label}", (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()

        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Renders the main dashboard HTML."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state')
def get_state():
    """Returns the current state as JSON for the frontend to update UI."""
    return jsonify(system_state)

if __name__ == "__main__":
    print("Starting Flask Web Server on http://127.0.0.1:2245")
    app.run(host='0.0.0.0', port=2245, debug=True, threaded=True)
