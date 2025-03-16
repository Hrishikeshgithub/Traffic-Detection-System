import cv2
from ultralytics import YOLO
import torch
import numpy as np

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Nano model for CPU speed

# Set device to CPU
device = 'cpu'
model.to(device)
print(f"Using device: {device}")

# Define vehicle classes (COCO dataset labels)
vehicle_classes = ['car', 'truck', 'bus', 'motorbike']

# Open video file (update path if needed)
cap = cv2.VideoCapture('sample2.mp4')  # Or use 0 for webcam

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Optional: Reduce resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables for vehicle counting
vehicle_count = 0
tracked_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Perform inference with YOLOv8
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7])  # Filter for vehicle classes

    # Process detection results
    current_ids = set()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            if conf > 0.5 and model.names[cls] in vehicle_classes:
                label = f"{model.names[cls]} ID: {track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if track_id != -1:
                    current_ids.add(track_id)

    # Update vehicle count
    new_vehicles = current_ids - tracked_ids
    vehicle_count += len(new_vehicles)
    tracked_ids.update(current_ids)

    # Display vehicle count on frame
    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame in a window
    cv2.imshow('Real-Time Traffic Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"Total vehicles detected: {vehicle_count}")