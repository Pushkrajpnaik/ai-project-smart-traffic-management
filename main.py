import cv2
import numpy as np
from tracker import *
from ultralytics import YOLO

# ================= MODEL =================
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture('highway.mp4')

count = 0
tracker = Tracker()

# ================= AREAS =================
area1 = [(275,445),(275,485),(530,485),(530,445)]
area2 = [(580,445),(580,485),(870,485),(870,445)]

area_1 = set()
area_2 = set()

# ================= SIGNAL =================
current_lane = 1
signal_timer = 0

# ================= MDP =================
# State -> density
# Action -> signal timing
# Reward -> reduced congestion
# Policy -> choose lane with higher density

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print([x, y])

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))

    # ================= YOLOv8 DETECTION =================
    results = model(frame)

    detections = []
    labels = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            detections.append([x1, y1, x2, y2])

            label_id = int(cls[i])
            label_name = model.names[label_id]
            labels.append(label_name)

    # ================= TRACKING =================
    idx_bbox = tracker.update(detections)

    # ================= COUNT + EMERGENCY =================
    emergency = False
    emergency_lane = None

    for i, bbox in enumerate(idx_bbox):
        x2, y2, x3, y3, id = bbox
        label = labels[i] if i < len(labels) else "unknown"

        cv2.rectangle(frame, (x2,y2), (x3,y3), (0,0,255), 2)
        cv2.putText(frame, f"{id}:{label}", (x2,y2),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,0), 2)

        cv2.circle(frame, (x3,y3), 4, (0,255,0), -1)

        # Lane detection
        if cv2.pointPolygonTest(np.array(area1,np.int32),(x3,y3),False) > 0:
            area_1.add(id)
            if label in ['truck', 'bus']:   # approx emergency
                emergency = True
                emergency_lane = 1

        if cv2.pointPolygonTest(np.array(area2,np.int32),(x3,y3),False) > 0:
            area_2.add(id)
            if label in ['truck', 'bus']:
                emergency = True
                emergency_lane = 2

    # ================= COUNT =================
    a1 = len(area_1)
    a2 = len(area_2)

    # ================= DENSITY =================
    lane_capacity = 20
    density1 = a1 / lane_capacity
    density2 = a2 / lane_capacity

    # ================= CONGESTION =================
    def get_congestion(d):
        if d < 0.3:
            return "LOW"
        elif d < 0.7:
            return "MEDIUM"
        else:
            return "HIGH"

    cong1 = get_congestion(density1)
    cong2 = get_congestion(density2)

    # ================= AI SIGNAL CONTROL =================
    if emergency:
        current_lane = emergency_lane
        signal_timer = 60
    else:
        if signal_timer <= 0:
            if density1 > density2:
                current_lane = 1
                signal_timer = 50
            else:
                current_lane = 2
                signal_timer = 50

    signal_timer -= 1

    # ================= DRAW AREAS =================
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,255),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,255),2)

    # ================= DISPLAY =================
    cv2.putText(frame, f"Lane1: {a1} ({cong1})", (50,50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)

    cv2.putText(frame, f"Lane2: {a2} ({cong2})", (50,90),
                cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)

    # Signal status
    if current_lane == 1:
        cv2.putText(frame, "Lane1: GREEN", (50,140),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
        cv2.putText(frame, "Lane2: RED", (50,180),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    else:
        cv2.putText(frame, "Lane1: RED", (50,140),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        cv2.putText(frame, "Lane2: GREEN", (50,180),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

    # Emergency display
    if emergency:
        cv2.putText(frame, "EMERGENCY VEHICLE DETECTED", (300,50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 3)

    print(f"L1:{a1}, L2:{a2}, Density1:{density1:.2f}, Density2:{density2:.2f}, Active:{current_lane}")

    cv2.imshow("FRAME", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()