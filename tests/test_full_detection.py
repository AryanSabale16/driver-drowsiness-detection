import cv2
from ai.intelligence.drowsiness_analyzer import DrowsinessAnalyzer
from ai.detection.face_mesh import FaceMeshDetector
from ai.detection.eye_detector import EyeDetector
from ai.detection.mouth_detector import MouthDetector
from ai.detection.head_pose_detector import HeadPoseDetector

from ai.drowsiness.temporal_analyzer import TemporalAnalyzer
from ai.drowsiness.perclos_analyzer import PerclosAnalyzer
from ai.drowsiness.yawn_analyzer import YawnAnalyzer
from ai.drowsiness.head_pose_analyzer import HeadPoseAnalyzer


def main():

    # =========================================================
    # INITIALIZE CAMERA
    # =========================================================

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # =========================================================
    # INITIALIZE DETECTORS
    # =========================================================

    face_detector = FaceMeshDetector()

    eye_detector = EyeDetector()
    mouth_detector = MouthDetector()
    head_pose_detector = HeadPoseDetector()
    

    # =========================================================
    # INITIALIZE TEMPORAL ANALYZERS
    # =========================================================

    temporal_analyzer = TemporalAnalyzer()
    perclos_analyzer = PerclosAnalyzer()
    yawn_analyzer = YawnAnalyzer()
    head_pose_analyzer = HeadPoseAnalyzer()
    drowsiness_analyzer = DrowsinessAnalyzer()

    print("=" * 60)
    print("FULL DRIVER DETECTION TEST")
    print("=" * 60)
    print("Eye + Mouth + Head Pose")
    print("Press Q to exit.")
    print("=" * 60)

    # =========================================================
    # MAIN LOOP
    # =========================================================

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read webcam frame.")
            break

        # Mirror webcam image
        frame = cv2.flip(frame, 1)

        # -----------------------------------------------------
        # FACE LANDMARK DETECTION
        # -----------------------------------------------------

        results = face_detector.detect(frame)

        if results.multi_face_landmarks:

            # Monitor the first detected face
            face_landmarks = results.multi_face_landmarks[0]

            frame_height, frame_width = frame.shape[:2]

            # =================================================
            # EYE ANALYSIS
            # =================================================

            left_eye, right_eye = eye_detector.get_eye_points(
                face_landmarks,
                frame_width,
                frame_height
            )

            left_ear, right_ear, average_ear = (
                eye_detector.calculate_eye_ear(
                    face_landmarks,
                    frame_width,
                    frame_height
                )
            )

            eye_state = eye_detector.classify_eye_state(
                average_ear
            )

            # Eye temporal analysis
            temporal_data = temporal_analyzer.update(
                eye_state
            )

            # PERCLOS
            perclos = perclos_analyzer.update(
                eye_state
            )

            # =================================================
            # MOUTH ANALYSIS
            # =================================================

            mouth_points = mouth_detector.get_mouth_points(
                face_landmarks,
                frame_width,
                frame_height
            )

            mar = mouth_detector.calculate_mouth_mar(
                face_landmarks,
                frame_width,
                frame_height
            )

            mouth_state = mouth_detector.classify_mouth_state(
                mar
            )

            # Yawn temporal analysis
            yawn_data = yawn_analyzer.update(
                mar
            )

            # =================================================
            # HEAD POSE ANALYSIS
            # =================================================

            pose = head_pose_detector.calculate_pose(
                face_landmarks,
                frame_width,
                frame_height
            )

            if pose is not None:

                pitch = pose["pitch"]
                yaw = pose["yaw"]
                roll = pose["roll"]

                head_state = head_pose_detector.classify_pose(
                    pitch,
                    yaw
                )

                # Head temporal analysis
                head_data = head_pose_analyzer.update(
                    head_state
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
            drowsiness_data = drowsiness_analyzer.analyze(
                perclos=perclos,
                temporal_data=temporal_data,
                yawn_data=yawn_data,
                head_pose_data=head_data
            )
            # =================================================
            # DRAW LANDMARKS
            # =================================================

            eye_detector.draw_eye_points(
                frame,
                left_eye,
                right_eye
            )

            mouth_detector.draw_mouth_points(
                frame,
                mouth_points
            )

            # =================================================
            # COLORS
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
            elif head_state in ["LEFT", "RIGHT", "UP"]:
                head_color = (0, 255, 255)
            else:
                head_color = (0, 255, 0)

            # =================================================
            # EYE INFORMATION
            # =================================================

            cv2.putText(
                frame,
                f"EAR: {average_ear:.3f}",
                (30, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                eye_color,
                2
            )

            cv2.putText(
                frame,
                f"Eyes: {eye_state}",
                (30, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                eye_color,
                2
            )

            cv2.putText(
                frame,
                f"Blinks: {temporal_data['total_blinks']}",
                (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Closure: "
                f"{temporal_data['closure_duration']:.2f}s",
                (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"PERCLOS: {perclos:.1f}%",
                (30, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            # =================================================
            # MOUTH INFORMATION
            # =================================================

            cv2.putText(
                frame,
                f"MAR: {mar:.3f}",
                (30, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                mouth_color,
                2
            )

            cv2.putText(
                frame,
                f"Mouth: {mouth_state}",
                (30, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                mouth_color,
                2
            )

            cv2.putText(
                frame,
                f"Yawn Count: {yawn_data['total_yawns']}",
                (30, 255),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Yawn Duration: "
                f"{yawn_data['mouth_open_duration']:.2f}s",
                (30, 285),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            # =================================================
            # HEAD INFORMATION
            # =================================================

            cv2.putText(
                frame,
                f"Pitch: {pitch:.1f}",
                (30, 325),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                head_color,
                2
            )

            cv2.putText(
                frame,
                f"Yaw: {yaw:.1f}",
                (30, 355),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                head_color,
                2
            )

            cv2.putText(
                frame,
                f"Roll: {roll:.1f}",
                (30, 385),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                head_color,
                2
            )

            cv2.putText(
                frame,
                f"Head: {head_state}",
                (30, 415),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                head_color,
                2
            )

            cv2.putText(
                frame,
                f"Head Down: "
                f"{head_data['downward_duration']:.2f}s",
                (30, 445),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            # =================================================
            # WARNINGS
            # =================================================

            warning_y = 490

            if temporal_data["prolonged_closure"]:

                cv2.putText(
                    frame,
                    "WARNING: PROLONGED EYE CLOSURE",
                    (30, warning_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

                warning_y += 30

            if yawn_data["yawn_active"]:

                cv2.putText(
                    frame,
                    "WARNING: YAWN DETECTED",
                    (30, warning_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

                warning_y += 30

            if head_data["prolonged_downward"]:

                cv2.putText(
                    frame,
                    "WARNING: PROLONGED HEAD DOWN",
                    (30, warning_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

        # -----------------------------------------------------
        # NO FACE DETECTED
        # -----------------------------------------------------

        else:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        # =====================================================
        # DISPLAY
        # =====================================================

        cv2.imshow(
            "Driver Drowsiness Detection - Full Test",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # =========================================================
    # CLEANUP
    # =========================================================

    camera.release()
    face_detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()