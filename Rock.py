import os
# Prevent OpenMP runtime errors if multiple copies of the library are loaded
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from ultralytics import YOLO

# Initialize text font settings for OpenCV drawing
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2

# Load the custom Rock-Paper-Scissors YOLOv8 model
model = YOLO('best.pt')
names = model.model.names  # Get class name mapping (e.g., 0 -> 'Rock')

# Open camera using DirectShow (CAP_DSHOW) to fix the MSMF Windows error
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set standard camera resolution (prevents format mismatches)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam. Try changing the index from 0 to 1.")

while cap.isOpened():
    # Read a frame from the webcam
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully finished.")
        break

    # Run object tracking on the current frame using ByteTrack
    results = model.track(frame, iou=0.5, show=False, tracker="bytetrack.yaml")

    # Check if any tracked objects were detected in the frame
    if results[0].boxes.id is not None:
        # Extract object IDs, class labels, bounding box coordinates, and confidence scores
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu()
        conf = results[0].boxes.conf.tolist()

        # Loop through all detected objects to draw boxes and labels
        for box, track_id, cof, c in zip(boxes, track_ids, conf, clss):
            x1, y1, x2, y2 = box.int().tolist()

            # Draw bounding box around the detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=thickness)

            # Display the object's class name above the box
            cv2.putText(frame, names[c], (x1, y1 - 20), font, font_scale, (255, 0, 0), thickness)

            # Display the unique tracking ID below the box
            cv2.putText(frame, str(track_id), (x2, y2 + 20), font, font_scale, (255, 0, 0), thickness)

    # Display the processed frame in a window named 'img'
    cv2.imshow('img', frame)

    # Break the loop immediately if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up and release webcam/window resources
cap.release()
cv2.destroyAllWindows()