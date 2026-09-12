import sys
from pathlib import Path

import cv2

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# YOLO
# ============================================================

from ai.models.yolo_drowsiness import YOLODrowsinessDetector


# ============================================================
# MEDIAPIPE / BEHAVIOR DETECTION
# ============================================================

from ai.detection.face_mesh import FaceMeshDetector
from ai.detection.eye_detector import EyeDetector
from ai.detection.mouth_detector import MouthDetector
from ai.detection.head_pose_detector import HeadPoseDetector


# ============================================================
# TEMPORAL ANALYSIS
# ============================================================

from ai.drowsiness.temporal_analyzer import TemporalAnalyzer
from ai.drowsiness.perclos_analyzer import PerclosAnalyzer
from ai.drowsiness.yawn_analyzer import YawnAnalyzer
from ai.drowsiness.head_pose_analyzer import HeadPoseAnalyzer


# ============================================================
# INTELLIGENCE + ALERTS
# ============================================================

from ai.intelligence.drowsiness_analyzer import DrowsinessAnalyzer
from ai.intelligence.hybrid_drowsiness_analyzer import (
    HybridDrowsinessAnalyzer
)

from ai.alerts.alert_engine import AlertEngine
from ai.alerts.alert_controller import AlertController


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    "runs/detect/training/runs/yolov8_drowsiness/weights/best.pt"
)

# Minimum YOLO confidence
YOLO_CONFIDENCE = 0.25

# Extra margin around YOLO bounding box
ROI_MARGIN = 20


# ============================================================
# ROI HELPER
# ============================================================

def get_face_roi(frame, bbox, margin=20):
    """
    Extract the YOLO detected face region with a small margin.

    Returns:
        roi
        x1
        y1
        x2
        y2
    """

    frame_height, frame_width = frame.shape[:2]

    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)

    x2 = min(frame_width, x2 + margin)
    y2 = min(frame_height, y2 + margin)

    if x2 <= x1 or y2 <= y1:
        return None, x1, y1, x2, y2

    roi = frame[y1:y2, x1:x2]

    return roi, x1, y1, x2, y2


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # INITIALIZE CAMERA
    # ========================================================

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # ========================================================
    # INITIALIZE YOLO
    # ========================================================

    print("Loading YOLOv8 drowsiness model...")

    yolo_detector = YOLODrowsinessDetector(
        MODEL_PATH
    )

    print("YOLOv8 loaded successfully.")

    # ========================================================
    # INITIALIZE MEDIAPIPE
    # ========================================================

    face_detector = FaceMeshDetector()

    eye_detector = EyeDetector()
    mouth_detector = MouthDetector()
    head_pose_detector = HeadPoseDetector()

    # ========================================================
    # INITIALIZE TEMPORAL ANALYZERS
    # ========================================================

    temporal_analyzer = TemporalAnalyzer()
    perclos_analyzer = PerclosAnalyzer()
    yawn_analyzer = YawnAnalyzer()
    head_pose_analyzer = HeadPoseAnalyzer()

    # ========================================================
    # INITIALIZE INTELLIGENCE
    # ========================================================

    drowsiness_analyzer = DrowsinessAnalyzer()

    # Hybrid fusion:
    #
    # 70% -> MediaPipe behavioral intelligence
    # 30% -> YOLO learned prediction
    #
    hybrid_analyzer = HybridDrowsinessAnalyzer(
        behavior_weight=0.70,
        yolo_weight=0.30
    )

    # ========================================================
    # INITIALIZE ALERT SYSTEM
    # ========================================================

    alert_engine = AlertEngine()
    alert_controller = AlertController()

    # ========================================================
    # START MESSAGE
    # ========================================================

    print("=" * 70)
    print("YOLO + MEDIAPIPE HYBRID DRIVER DROWSINESS TEST")
    print("=" * 70)
    print("YOLO        -> Driver face + Alert/Drowsy prediction")
    print("MediaPipe   -> EAR + MAR + Head Pose")
    print("Temporal    -> Blink + PERCLOS + Yawn + Head duration")
    print("Behavioral  -> MediaPipe drowsiness score")
    print("Hybrid      -> YOLO + Behavioral fusion")
    print("Alert       -> Final warning / critical decision")
    print("Press Q to exit.")
    print("=" * 70)

    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read webcam frame.")
                break

            # ------------------------------------------------
            # MIRROR WEBCAM
            # ------------------------------------------------

            frame = cv2.flip(frame, 1)

            frame_height, frame_width = frame.shape[:2]

            # =================================================
            # STEP 1: YOLO DETECTION
            # =================================================

            yolo_detections = yolo_detector.predict(
                frame,
                confidence=YOLO_CONFIDENCE
            )

            # =================================================
            # NO YOLO DETECTION
            # =================================================

            if not yolo_detections:

                # Make sure any previous alarm is stopped
                alert_controller.stop()

                cv2.putText(
                    frame,
                    "NO DRIVER DETECTED",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "YOLO confidence below threshold",
                    (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

                cv2.imshow(
                    "YOLO + MediaPipe Drowsiness Detection",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                continue

            # =================================================
            # STEP 2: SELECT BEST YOLO DETECTION
            # =================================================

            best_detection = max(
                yolo_detections,
                key=lambda detection: detection["confidence"]
            )

            yolo_class = best_detection["class_name"]
            yolo_confidence = best_detection["confidence"]
            yolo_bbox = best_detection["bbox"]

            # =================================================
            # STEP 3: EXTRACT YOLO FACE ROI
            # =================================================

            roi, roi_x1, roi_y1, roi_x2, roi_y2 = get_face_roi(
                frame,
                yolo_bbox,
                ROI_MARGIN
            )

            if roi is None or roi.size == 0:

                alert_controller.stop()

                cv2.putText(
                    frame,
                    "INVALID YOLO ROI",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

                cv2.imshow(
                    "YOLO + MediaPipe Drowsiness Detection",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                continue

            # =================================================
            # STEP 4: MEDIAPIPE ON YOLO ROI
            # =================================================

            results = face_detector.detect(roi)

            # =================================================
            # YOLO BOUNDING BOX COLOR
            # =================================================

            if yolo_class == "drowsy":
                yolo_color = (0, 0, 255)
            else:
                yolo_color = (0, 255, 0)

            # =================================================
            # DRAW YOLO BOUNDING BOX
            # =================================================

            cv2.rectangle(
                frame,
                (roi_x1, roi_y1),
                (roi_x2, roi_y2),
                yolo_color,
                2
            )

            cv2.putText(
                frame,
                f"YOLO: {yolo_class.upper()} "
                f"{yolo_confidence:.2f}",
                (roi_x1, max(roi_y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                yolo_color,
                2
            )

            # =================================================
            # MEDIAPIPE FACE FOUND
            # =================================================

            if results.multi_face_landmarks:

                # -------------------------------------------------
                # Use first face detected inside YOLO ROI
                # -------------------------------------------------

                face_landmarks = results.multi_face_landmarks[0]

                roi_height, roi_width = roi.shape[:2]

                # =================================================
                # EYE ANALYSIS
                # =================================================

                left_eye, right_eye = (
                    eye_detector.get_eye_points(
                        face_landmarks,
                        roi_width,
                        roi_height
                    )
                )

                left_ear, right_ear, average_ear = (
                    eye_detector.calculate_eye_ear(
                        face_landmarks,
                        roi_width,
                        roi_height
                    )
                )

                eye_state = (
                    eye_detector.classify_eye_state(
                        average_ear
                    )
                )

                # =================================================
                # TEMPORAL EYE ANALYSIS
                # =================================================

                temporal_data = temporal_analyzer.update(
                    eye_state
                )

                # =================================================
                # PERCLOS
                # =================================================

                perclos = perclos_analyzer.update(
                    eye_state
                )

                # =================================================
                # MOUTH ANALYSIS
                # =================================================

                mouth_points = (
                    mouth_detector.get_mouth_points(
                        face_landmarks,
                        roi_width,
                        roi_height
                    )
                )

                mar = (
                    mouth_detector.calculate_mouth_mar(
                        face_landmarks,
                        roi_width,
                        roi_height
                    )
                )

                mouth_state = (
                    mouth_detector.classify_mouth_state(
                        mar
                    )
                )

                # =================================================
                # YAWN ANALYSIS
                # =================================================

                yawn_data = yawn_analyzer.update(
                    mar
                )

                # =================================================
                # HEAD POSE
                # =================================================

                pose = head_pose_detector.calculate_pose(
                    face_landmarks,
                    roi_width,
                    roi_height
                )

                if pose is not None:

                    pitch = pose["pitch"]
                    yaw = pose["yaw"]
                    roll = pose["roll"]

                    head_state = (
                        head_pose_detector.classify_pose(
                            pitch,
                            yaw
                        )
                    )

                    head_data = (
                        head_pose_analyzer.update(
                            head_state
                        )
                    )

                else:

                    pitch = 0.0
                    yaw = 0.0
                    roll = 0.0

                    head_state = "UNKNOWN"

                    head_data = {
                        "downward_duration": 0.0,
                        "prolonged_downward": False,
                        "total_prolonged_events": 0
                    }

                # =================================================
                # STEP 5: BEHAVIORAL DROWSINESS INTELLIGENCE
                # =================================================

                behavior_data = (
                    drowsiness_analyzer.analyze(
                        perclos=perclos,
                        temporal_data=temporal_data,
                        yawn_data=yawn_data,
                        head_pose_data=head_data
                    )
                )

                # =================================================
                # STEP 6: HYBRID YOLO + MEDIAPIPE FUSION
                # =================================================

                hybrid_data = hybrid_analyzer.analyze(
                    behavior_data=behavior_data,
                    yolo_class_id=best_detection["class_id"],
                    yolo_confidence=yolo_confidence
                )

                # =================================================
                # STEP 7: ALERT ENGINE
                # =================================================

                alert_data = alert_engine.evaluate(
                    hybrid_data
                )

                alert_result = (
                    alert_controller.process(
                        alert_data
                    )
                )

                # =================================================
                # TRANSLATE MEDIAPIPE POINTS
                # FROM ROI -> FULL FRAME
                # =================================================

                full_frame_left_eye = [
                    (
                        x + roi_x1,
                        y + roi_y1
                    )
                    for x, y in left_eye
                ]

                full_frame_right_eye = [
                    (
                        x + roi_x1,
                        y + roi_y1
                    )
                    for x, y in right_eye
                ]

                full_frame_mouth_points = [
                    (
                        x + roi_x1,
                        y + roi_y1
                    )
                    for x, y in mouth_points
                ]

                # =================================================
                # DRAW EYE LANDMARKS
                # =================================================

                eye_detector.draw_eye_points(
                    frame,
                    full_frame_left_eye,
                    full_frame_right_eye
                )

                # =================================================
                # DRAW MOUTH LANDMARKS
                # =================================================

                mouth_detector.draw_mouth_points(
                    frame,
                    full_frame_mouth_points
                )

                # =================================================
                # STATUS COLORS
                # =================================================

                if eye_state == "CLOSED":
                    eye_color = (0, 0, 255)
                else:
                    eye_color = (0, 255, 0)

                if mouth_state == "OPEN":
                    mouth_color = (0, 0, 255)
                else:
                    mouth_color = (0, 255, 0)

                if head_state == "DOWN":
                    head_color = (0, 0, 255)

                elif head_state in [
                    "LEFT",
                    "RIGHT",
                    "UP"
                ]:
                    head_color = (0, 255, 255)

                else:
                    head_color = (0, 255, 0)

                # =================================================
                # LEFT PANEL - MEDIAPIPE
                # =================================================

                x_text = 20

                cv2.putText(
                    frame,
                    "MEDIAPIPE",
                    (x_text, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"EAR: {average_ear:.3f}",
                    (x_text, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    eye_color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Eyes: {eye_state}",
                    (x_text, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    eye_color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Blinks: {temporal_data['total_blinks']}",
                    (x_text, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"PERCLOS: {perclos:.1f}%",
                    (x_text, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"MAR: {mar:.3f}",
                    (x_text, 195),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    mouth_color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Mouth: {mouth_state}",
                    (x_text, 225),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    mouth_color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Yawns: {yawn_data['total_yawns']}",
                    (x_text, 255),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Head: {head_state}",
                    (x_text, 290),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    head_color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Pitch: {pitch:.1f}",
                    (x_text, 320),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    head_color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Yaw: {yaw:.1f}",
                    (x_text, 350),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    head_color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Head Down: "
                    f"{head_data['downward_duration']:.2f}s",
                    (x_text, 380),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                # =================================================
                # RIGHT PANEL - HYBRID INTELLIGENCE
                # =================================================

                intelligence_x = max(
                    frame_width - 350,
                    400
                )

                # -------------------------------------------------
                # Get hybrid values
                # -------------------------------------------------

                behavior_score = (
                    hybrid_data[
                        "behavior_score_normalized"
                    ]
                )

                yolo_drowsiness_score = (
                    hybrid_data[
                        "yolo_drowsiness_score"
                    ]
                )

                hybrid_score = (
                    hybrid_data["score"]
                )

                hybrid_status = (
                    hybrid_data["status"]
                )

                # =================================================
                # STATUS COLOR
                # =================================================

                if hybrid_status == "CRITICAL":

                    status_color = (0, 0, 255)

                elif hybrid_status == "DROWSY":

                    status_color = (0, 165, 255)

                elif hybrid_status == "CAUTION":

                    status_color = (0, 255, 255)

                else:

                    status_color = (0, 255, 0)

                # =================================================
                # INTELLIGENCE HEADER
                # =================================================

                cv2.putText(
                    frame,
                    "HYBRID INTELLIGENCE",
                    (intelligence_x, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                # =================================================
                # BEHAVIOR SCORE
                # =================================================

                cv2.putText(
                    frame,
                    f"Behavior: {behavior_score:.1f}",
                    (intelligence_x, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                # =================================================
                # YOLO SCORE
                # =================================================

                cv2.putText(
                    frame,
                    f"YOLO Score: "
                    f"{yolo_drowsiness_score:.1f}",
                    (intelligence_x, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    yolo_color,
                    2
                )

                # =================================================
                # HYBRID SCORE
                # =================================================

                cv2.putText(
                    frame,
                    f"Hybrid Score: "
                    f"{hybrid_score:.1f}",
                    (intelligence_x, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2
                )

                # =================================================
                # FINAL STATUS
                # =================================================

                cv2.putText(
                    frame,
                    f"Status: {hybrid_status}",
                    (intelligence_x, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2
                )

                # =================================================
                # ALERT SYSTEM
                # =================================================

                cv2.putText(
                    frame,
                    "ALERT SYSTEM",
                    (intelligence_x, 215),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Level: "
                    f"{alert_result['alert_level']}",
                    (intelligence_x, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    status_color,
                    2
                )

                alarm_status = (
                    "ACTIVE"
                    if alert_result["alarm_active"]
                    else "OFF"
                )

                cv2.putText(
                    frame,
                    f"Alarm: {alarm_status}",
                    (intelligence_x, 285),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (
                        (0, 0, 255)
                        if alert_result["alarm_active"]
                        else (0, 255, 0)
                    ),
                    2
                )

                # =================================================
                # REASONS
                # =================================================

                cv2.putText(
                    frame,
                    "Reasons:",
                    (intelligence_x, 330),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                reasons = hybrid_data.get(
                    "reasons",
                    []
                )

                reason_y = 360

                if reasons:

                    for reason in reasons[:3]:

                        reason_text = str(reason)

                        if len(reason_text) > 32:

                            reason_text = (
                                reason_text[:29]
                                + "..."
                            )

                        cv2.putText(
                            frame,
                            f"- {reason_text}",
                            (intelligence_x, reason_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.43,
                            (0, 165, 255),
                            1
                        )

                        reason_y += 23

                else:

                    cv2.putText(
                        frame,
                        "None",
                        (intelligence_x, reason_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )

            # =================================================
            # MEDIAPIPE DID NOT FIND FACE
            # =================================================

            else:

                # Stop alarm because we cannot reliably
                # calculate the behavioral state.
                alert_controller.stop()

                cv2.putText(
                    frame,
                    "MEDIAPIPE: FACE NOT FOUND",
                    (30, 430),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # =================================================
            # DISPLAY
            # =================================================

            cv2.imshow(
                "YOLO + MediaPipe Drowsiness Detection",
                frame
            )

            # =================================================
            # QUIT
            # =================================================

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        # ====================================================
        # CLEANUP
        # ====================================================

        alert_controller.stop()

        camera.release()

        face_detector.close()

        cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()