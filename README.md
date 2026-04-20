# Smart Traffic Light System

## Overview
This repository implements an AI-based Smart Traffic Management system using YOLOv8 and OpenCV. The system reads video input, detects vehicles, tracks them across frames, computes lane densities, and chooses the next green lane using a simplified MDP-style decision module.

## Features
- YOLOv8-based vehicle detection.
- Vehicle tracking with consistent object IDs.
- Lane-wise counting and density calculation.
- Emergency vehicle detection and emergency-priority override.
- Simplified AI decision logic for dynamic signal timing.
- On-screen display of lane counts, densities, green signal lane, and timer.

## Project Structure
- `main.py` — application entry point.
- `detector.py` — YOLOv8 detection module.
- `tracker.py` — tracker module for object ID persistence.
- `density.py` — lane area and density calculations.
- `mdp_controller.py` — decision logic for green signal selection.
- `emergency.py` — emergency detection helper.

## How It Works
1. Reads video frames from `highway.mp4`.
2. Detects vehicles using YOLOv8.
3. Tracks vehicles and assigns IDs.
4. Computes lane-specific vehicle counts and densities.
5. Uses a simplified MDP decision function to choose the next green lane.
6. Overrides normal decisions when an emergency vehicle is present.
7. Displays signal status and timer on the output window.

## Usage
1. Change to the project folder:
```bash
cd '/Users/pushkrajpnaik/Desktop/Pushkraj /Smart-Traffic-Management-System'
```
2. Activate the virtual environment:
```bash
source .venv/bin/activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the system:
```bash
python main.py
```

## Dependencies
- opencv-python
- numpy
- torch
- ultralytics

## Notes
- The default entrypoint is `main.py`.
- If you have a custom YOLOv8 model with emergency vehicle labels, update `detector.py` accordingly.

## Contributors
- Arpit Rai

## Acknowledgments
Thanks to Noida Institute of Engineering and Technology and the AIML department for supporting this project.
