import sys
from pathlib import Path

import cv2

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.models.yolo_drowsiness import YOLODrowsinessDetector


MODEL_PATH = (
    "runs/detect/training/runs/yolov8_drowsiness/weights/best.pt"
)


def main():
    detector = YOLODrowsinessDetector(MODEL_PATH)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Webcam started.")
    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read frame.")
            break

        # Run YOLO detection with a low confidence threshold
        # for debugging live webcam detection.
        detections = detector.predict(
            frame,
            confidence=0.05
        )

        # Print detections in terminal
        if detections:
            print(detections)

        # Draw detections
        for detection in detections:
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            x1, y1, x2, y2 = detection["bbox"]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display class and confidence
            label = f"{class_name}: {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Display webcam
        cv2.imshow(
            "YOLOv8 Drowsiness Detection",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()