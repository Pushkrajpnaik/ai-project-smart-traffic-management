# AI-Based Smart Traffic Management System

## Project Overview
**Topic:** AI-Based Smart Traffic Management System  
**GitHub Repo:** [ai-project-smart-traffic-management](https://github.com/Pushkrajpnaik/ai-project-smart-traffic-management)  
**Team:** Om Dhamame (23070122155), Pritika Kurup (23070122167), Pushkraj Naik (23070122169)  
**Institution:** Symbiosis Institute of Technology, Pune — TYCSE-B  

## Problem Statement
Traffic congestion in urban areas leads to increased waiting times, higher emissions, and delayed emergency responses. Traditional fixed-time traffic signals fail to adapt to real-time traffic density. This project implements an Agentic AI system that dynamically controls traffic signals based on real-time computer vision analysis, optimizing traffic flow and prioritizing emergency vehicles.

## System Architecture
The system architecture follows a 3-layer design:
1. **Input Layer:** CCTV/live feed processing at 30fps.
2. **Processing Layer (Agentic Core):** YOLOv8 detection → Object tracking → Density estimation → MDP controller reasoning.
3. **Output Layer:** Dynamic signal control + Emergency routing.

### Agentic AI Approach
Unlike a simple pipeline, this system is governed by a `TrafficAgent` that operates on a continuous **Perceive → Reason → Act → Reflect** loop:
- **Perceive:** Analyzes video frames to detect vehicles, track movements, estimate lane densities, and identify emergency vehicles.
- **Reason:** Utilizes a Markov Decision Process (MDP) based heuristic to determine the optimal signal timing given the current traffic state.
- **Act:** Modifies the traffic signal states (GREEN/YELLOW/RED) and timings.
- **Reflect:** Logs the state, action, and reward (negative wait time) for evaluation and continuous monitoring.

## AI Techniques Used
- **YOLOv8 (You Only Look Once):** Selected for its state-of-the-art real-time object detection capabilities. Used to classify vehicles (cars, buses, trucks, ambulances, etc.).
- **Object Tracking:** Centroid-based tracking maintains unique vehicle IDs across frames to prevent double-counting.
- **Markov Decision Process (MDP):** A mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision maker. Used here to select the optimal green signal duration.
- **Agentic Loop:** Encapsulates the system into an autonomous agent capable of multi-step reasoning and state-based action.

## Dataset
The system is designed to work with real-time video feeds or pre-recorded videos (like `highway.mp4`). For training and benchmarking, datasets like UA-DETRAC or VIRAT can be utilized to fine-tune the YOLOv8 model for specific angles and vehicle types.

## Installation & Setup

To run this project on a different PC, follow these detailed steps to ensure all dependencies and environments are correctly configured.

### Prerequisites
- **Python 3.8 or higher**: Ensure Python is installed. You can check this by running `python --version` in your terminal.
- **Git**: To clone the repository.
- **FFmpeg (Optional)**: For advanced video processing, though not strictly required for the basic agent loop.

### 1. Environment Setup
It is highly recommended to use a virtual environment to avoid dependency conflicts:
```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Required Libraries
Install the core AI and web dependencies using the provided requirements file:
```bash
pip install -r requirements.txt
```

If you encounter any issues, you can manually install the critical packages:
```bash
pip install opencv-python ultralytics pyyaml flask matplotlib
```

### 3. YOLOv8 Model Weights
The system uses the `yolov8n.pt` (Nano) model by default for speed. The first time you run the script, it will automatically download the weights from Ultralytics. Ensure you have an active internet connection for the first run.

### 4. Running the Project

#### **A. Web Dashboard (Recommended)**
To launch the modern web-based command center:
```bash
python app.py
```
Then open your browser and navigate to `http://127.0.0.1:5000`.

#### **B. CLI Demo Mode**
To run the processing loop directly in a CV2 window:
```bash
python main.py --demo
```

#### **C. Custom Video Source**
To run against a specific video file:
```bash
python main.py --source your_video.mp4
```

---

## Project Structure
```text
ai-project-smart-traffic-management/
├── README.md                 # Full project description
├── main.py                   # Entry point — runs the TrafficAgent
├── agent/                    
│   ├── traffic_agent.py      # Agentic AI loop (Perceive→Reason→Act→Reflect)
│   └── mdp_controller.py     # MDP state-action-reward logic
├── vision/                   
│   ├── detector.py           # YOLOv8 detection wrapper
│   ├── tracker.py            # Object tracking (unique IDs)
│   └── density_estimator.py  # Lane density calculation
├── utils/                    
│   ├── signal_controller.py  # Signal timing output
│   └── emergency_handler.py  # Emergency vehicle override
├── data_pipeline.py          # Data prep & feature engineering
├── evaluation.py             # Metrics + plots
├── results/                  
│   ├── metrics.json          # Evaluation metrics
│   └── plots/                # Evaluation plots
├── requirements.txt          # Python dependencies
└── config.yaml               # Thresholds, model paths, timing params
```

## References
1. Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection." CVPR.
2. Jocher, G. "YOLOv8." Ultralytics.
3. Puterman, M. L. "Markov Decision Processes: Discrete Stochastic Dynamic Programming."
4. (Add additional relevant research papers referenced in your report)
