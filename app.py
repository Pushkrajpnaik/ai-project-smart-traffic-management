import os
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import numpy as np
import yaml
import time
from werkzeug.utils import secure_filename
from agent.traffic_agent import TrafficAgent
from data_pipeline import DataPipeline

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Global variables
current_video_path = config["video"]["source"]
system_state = {
    "lane1_signal": "RED",
    "lane2_signal": "RED",
    "lane1_timer": 0,
    "lane2_timer": 0,
    "lane1_count": 0,
    "lane2_count": 0,
    "lane1_density": "LOW",
    "lane2_density": "LOW",
    "emergency": False,
    "current_video": os.path.basename(current_video_path)
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_frames():
    global system_state, current_video_path
    agent = TrafficAgent(config)
    pipeline = DataPipeline(config)
    
    cap = cv2.VideoCapture(current_video_path)
    frame_skip = config["video"]["frame_skip"]
    count = 0

    area1 = config["lanes"]["lane1"]["polygon"]
    area2 = config["lanes"]["lane2"]["polygon"]

    while True:
        # Check if video path changed
        if system_state["current_video"] != os.path.basename(current_video_path):
            cap.release()
            cap = cv2.VideoCapture(current_video_path)
            system_state["current_video"] = os.path.basename(current_video_path)

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        count += 1
        if count % frame_skip != 0:
            continue

        frame = pipeline.preprocess_frame(frame)
        state = agent.perceive(frame)
        action = agent.reason(state)
        signal_state = agent.act(action)

        system_state["lane1_count"] = state['lane1_count']
        system_state["lane2_count"] = state['lane2_count']
        system_state["lane1_density"] = ["LOW", "MEDIUM", "HIGH"][state["tier_lane1"]]
        system_state["lane2_density"] = ["LOW", "MEDIUM", "HIGH"][state["tier_lane2"]]
        system_state["emergency"] = state["emergency_flag"]

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

        cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 255), 2)
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 255), 2)

        for obj in state["tracked_objects"]:
            x1, y1, x2, y2, obj_id, lane_id = obj
            label = state["labels_dict"].get(obj_id, "unknown")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{obj_id}:{label}", (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    # List videos in root and uploads
    videos = [f for f in os.listdir('.') if f.endswith('.mp4')]
    uploaded_videos = [os.path.join('uploads', f) for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.mp4')]
    all_videos = videos + uploaded_videos
    return render_template('index.html', videos=all_videos, current=os.path.basename(current_video_path))

@app.route('/select_video', methods=['POST'])
def select_video():
    global current_video_path
    video = request.form.get('video')
    if video:
        current_video_path = video
    return redirect(url_for('index'))

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global current_video_path
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        current_video_path = save_path
        return redirect(url_for('index'))
    return redirect(request.url)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state')
def get_state():
    return jsonify(system_state)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2254, debug=True, threaded=True)
