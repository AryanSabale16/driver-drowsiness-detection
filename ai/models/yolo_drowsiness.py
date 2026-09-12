from pathlib import Path
from ultralytics import YOLO


class YOLODrowsinessDetector:
    """
    YOLOv8-based drowsiness detector.

    Class mapping:
        0 -> alert
        1 -> drowsy
    """

    def __init__(self, model_path: str):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {model_path}"
            )

        self.model = YOLO(str(model_path))

        # Expected classes from our trained model
        self.class_names = {
            0: "alert",
            1: "drowsy",
        }

    def predict(self, frame, confidence: float = 0.25):
        """
        Run YOLO inference on a single frame.

        Returns:
            list of dictionaries containing:
                class_id
                class_name
                confidence
                bbox
        """

        results = self.model.predict(
            source=frame,
            conf=confidence,
            verbose=False,
        )

        detections = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            class_name = self.class_names.get(
                class_id,
                str(class_id)
            )

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
            })

        return detections