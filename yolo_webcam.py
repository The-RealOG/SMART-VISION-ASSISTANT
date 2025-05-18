from ultralytics import YOLO
import cv2

# Load YOLOv8 nano model (smallest and fastest)
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(1)  # or 1 if external

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame)

    # Annotate frame
    annotated = results[0].plot()

    # Show the frame
    cv2.imshow("YOLOv8 Detection", annotated)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
