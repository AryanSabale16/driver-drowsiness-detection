import cv2

from ai.detection.face_mesh import FaceMeshDetector
from ai.detection.eye_detector import EyeDetector
from ai.detection.mouth_detector import MouthDetector

from ai.drowsiness.temporal_analyzer import TemporalAnalyzer
from ai.drowsiness.yawn_analyzer import YawnAnalyzer


def main():

    # ---------------------------------------------------------
    # 1. INITIALIZE CAMERA
    # ---------------------------------------------------------

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # ---------------------------------------------------------
    # 2. INITIALIZE DETECTORS
    # ---------------------------------------------------------

    face_detector = FaceMeshDetector()

    eye_detector = EyeDetector(
        ear_threshold=0.20
    )

    mouth_detector = MouthDetector(
        mar_threshold=0.50
    )

    temporal_analyzer = TemporalAnalyzer(
        min_blink_duration=0.08,
        max_blink_duration=0.50,
        prolonged_closure_threshold=1.5
    )

    yawn_analyzer = YawnAnalyzer(
        mar_threshold=0.55,
        yawn_duration_threshold=1.5
    )

    print("=" * 50)
    print("Driver Behaviour Analysis Started")
    print("=" * 50)
    print("Press Q to exit.")

    # ---------------------------------------------------------
    # 3. MAIN VIDEO LOOP
    # ---------------------------------------------------------

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read webcam frame.")
            break

        # Mirror webcam image
        frame = cv2.flip(frame, 1)

        # -----------------------------------------------------
        # 4. FACE LANDMARK DETECTION
        # -----------------------------------------------------

        results = face_detector.detect(frame)

        if results.multi_face_landmarks:

            # We only monitor one driver
            face_landmarks = results.multi_face_landmarks[0]

            frame_height, frame_width = frame.shape[:2]

            # =================================================
            # EYE ANALYSIS
            # =================================================

            # Get eye landmark coordinates
            left_eye, right_eye = eye_detector.get_eye_points(
                face_landmarks,
                frame_width,
                frame_height
            )

            # Calculate EAR
            left_ear, right_ear, average_ear = (
                eye_detector.calculate_eye_ear(
                    face_landmarks,
                    frame_width,
                    frame_height
                )
            )

            # Determine OPEN / CLOSED
            eye_state = eye_detector.classify_eye_state(
                average_ear
            )

            # Temporal eye analysis
            temporal_data = temporal_analyzer.update(
                eye_state
            )

            # =================================================
            # MOUTH ANALYSIS
            # =================================================

            # Get mouth landmark coordinates
            mouth_points = mouth_detector.get_mouth_points(
                face_landmarks,
                frame_width,
                frame_height
            )

            # Calculate MAR
            mar = mouth_detector.calculate_mouth_mar(
                face_landmarks,
                frame_width,
                frame_height
            )

            # Determine OPEN / CLOSED
            mouth_state = mouth_detector.classify_mouth_state(
                mar
            )

            # Temporal yawn analysis
            yawn_data = yawn_analyzer.update(
                mar
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

            # =================================================
            # DISPLAY EYE INFORMATION
            # =================================================

            cv2.putText(
                frame,
                f"EAR: {average_ear:.3f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                eye_color,
                2
            )

            cv2.putText(
                frame,
                f"Eyes: {eye_state}",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                eye_color,
                2
            )

            cv2.putText(
                frame,
                f"Blinks: {temporal_data['total_blinks']}",
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                (
                    "Eye Closure: "
                    f"{temporal_data['closure_duration']:.2f}s"
                ),
                (30, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            # =================================================
            # DISPLAY MOUTH INFORMATION
            # =================================================

            cv2.putText(
                frame,
                f"MAR: {mar:.3f}",
                (30, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                mouth_color,
                2
            )

            cv2.putText(
                frame,
                f"Mouth: {mouth_state}",
                (30, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                mouth_color,
                2
            )

            cv2.putText(
                frame,
                f"Yawns: {yawn_data['total_yawns']}",
                (30, 265),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                (
                    "Mouth Open: "
                    f"{yawn_data['mouth_open_duration']:.2f}s"
                ),
                (30, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            # =================================================
            # WARNINGS
            # =================================================

            warning_y = 350

            if temporal_data["prolonged_closure"]:

                cv2.putText(
                    frame,
                    "WARNING: PROLONGED EYE CLOSURE",
                    (30, warning_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                warning_y += 40

            if yawn_data["yawn_active"]:

                cv2.putText(
                    frame,
                    "WARNING: YAWN DETECTED",
                    (30, warning_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
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

        # -----------------------------------------------------
        # DISPLAY FRAME
        # -----------------------------------------------------

        cv2.imshow(
            "Driver Drowsiness Detection - Behaviour Analysis",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------

    camera.release()
    face_detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()