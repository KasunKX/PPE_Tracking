import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response, jsonify
from collections import deque
import threading
from flask_cors import CORS
import supervision as sv

app = Flask(__name__)
CORS(app)

# Load YOLO model
model = YOLO("best.pt")

# Last 10 predictions (counts)
last_predictions = deque(maxlen=10)
pred_lock = threading.Lock()

# Store trails for persons
person_trails = {}
MAX_TRAIL_LENGTH = 40  # Number of points to keep for trail

# Supervision label annotator
label_annotator = sv.LabelAnnotator(
    text_scale=0.5,
    text_thickness=1,
    text_padding=4,
)

# Classes to track and annotate
track_classes = ["Person", "Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest", "Machinery"]

def interpolate_trail(trail, smooth_factor=3):
    """Interpolate points for smoother trails"""
    if len(trail) < 2:
        return trail
    interp_points = []
    for i in range(1, len(trail)):
        x0, y0 = trail[i-1]
        x1, y1 = trail[i]
        for t in np.linspace(0, 1, smooth_factor):
            xi = int(x0 + (x1 - x0) * t)
            yi = int(y0 + (y1 - y0) * t)
            interp_points.append((xi, yi))
    return interp_points

def gen_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Initialize counts
        counts = {cls: 0 for cls in track_classes}

        results = model.track(frame, persist=True)
        other_boxes = []
        other_labels = []

        if results and len(results[0].boxes) > 0:
            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            ids = (
                r.boxes.id.cpu().numpy().astype(int)
                if r.boxes.id is not None
                else np.arange(len(boxes))
            )

            for box, cls_id, track_id in zip(boxes, classes, ids):
                x1, y1, x2, y2 = map(int, box[:4])
                class_name = model.names[cls_id]

                if class_name not in track_classes:
                    continue

                counts[class_name] += 1

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if class_name == "Person":
                    # Update trail
                    if track_id not in person_trails:
                        person_trails[track_id] = []
                    person_trails[track_id].append((cx, cy))
                    person_trails[track_id] = person_trails[track_id][-MAX_TRAIL_LENGTH:]

                    # Interpolate for smooth trail
                    smooth_trail = interpolate_trail(person_trails[track_id], smooth_factor=4)

                    # Draw neon trail
                    for i in range(1, len(smooth_trail)):
                        pt1 = smooth_trail[i - 1]
                        pt2 = smooth_trail[i]
                        alpha = i / len(smooth_trail)
                        color = (0, int(255 * alpha), 255)
                        thickness = max(1, int(2 * alpha))
                        cv2.line(frame, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

                    # Draw box around person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, lineType=cv2.LINE_AA)

                    # Draw live dot
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)

                else:
                    # Other tracked classes: only labels
                    other_boxes.append([x1, y1, x2, y2])
                    other_labels.append(class_name)

            # Annotate other objects
            if other_boxes:
                other_detections = sv.Detections(
                    xyxy=np.array(other_boxes),
                    class_id=np.arange(len(other_labels)),  # dummy IDs
                    confidence=np.ones(len(other_labels))  # dummy confidence
                )
                frame = label_annotator.annotate(
                    scene=frame,
                    detections=other_detections,
                    labels=other_labels
                )

        # Update last 10 counts safely
        with pred_lock:
            last_predictions.append(counts.copy())

        # Encode frame
        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

@app.route("/video")
def video():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/counts")
def counts_endpoint():
    with pred_lock:
        return jsonify(list(last_predictions))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
