import cv2

from ai.detection.face_mesh import FaceMeshDetector
from ai.detection.mouth_detector import MouthDetector
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

    # Basic mouth landmark/MAR detector
    mouth_detector = MouthDetector(
        mar_threshold=0.50
    )

    # Temporal yawn analyzer
    yawn_analyzer = YawnAnalyzer(
        mar_threshold=0.55,
        yawn_duration_threshold=1.5
    )

    print("MAR + Yawn Detection Test Started")
    print("Press Q to exit.")

    # ---------------------------------------------------------
    # 3. MAIN VIDEO LOOP
    # ---------------------------------------------------------

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        # -----------------------------------------------------
        # 4. FACE LANDMARK DETECTION
        # -----------------------------------------------------

        results = face_detector.detect(frame)

        if results.multi_face_landmarks:

            face_landmarks = (
                results.multi_face_landmarks[0]
            )

            frame_height, frame_width = (
                frame.shape[:2]
            )

            # -------------------------------------------------
            # 5. GET MOUTH LANDMARKS
            # -------------------------------------------------

            mouth_points = (
                mouth_detector.get_mouth_points(
                    face_landmarks,
                    frame_width,
                    frame_height
                )
            )

            # -------------------------------------------------
            # 6. CALCULATE MAR
            # -------------------------------------------------

            mar = (
                mouth_detector.calculate_mouth_mar(
                    face_landmarks,
                    frame_width,
                    frame_height
                )
            )

            # -------------------------------------------------
            # 7. BASIC MOUTH STATE
            # -------------------------------------------------

            mouth_state = (
                mouth_detector.classify_mouth_state(
                    mar
                )
            )

            # -------------------------------------------------
            # 8. TEMPORAL YAWN ANALYSIS
            # -------------------------------------------------

            yawn_data = yawn_analyzer.update(mar)

            # -------------------------------------------------
            # 9. DRAW MOUTH LANDMARKS
            # -------------------------------------------------

            mouth_detector.draw_mouth_points(
                frame,
                mouth_points
            )

            # -------------------------------------------------
            # 10. STATUS COLOR
            # -------------------------------------------------

            if mouth_state == "OPEN":
                status_color = (0, 0, 255)
            else:
                status_color = (0, 255, 0)

            # -------------------------------------------------
            # 11. DISPLAY MAR
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"MAR: {mar:.3f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2
            )

            # -------------------------------------------------
            # 12. DISPLAY MOUTH STATE
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"Mouth: {mouth_state}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2
            )

            # -------------------------------------------------
            # 13. DISPLAY OPEN DURATION
            # -------------------------------------------------

            cv2.putText(
                frame,
                (
                    "Open Duration: "
                    f"{yawn_data['mouth_open_duration']:.2f}s"
                ),
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 0),
                2
            )

            # -------------------------------------------------
            # 14. DISPLAY YAWN COUNT
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"Yawns: {yawn_data['total_yawns']}",
                (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            # -------------------------------------------------
            # 15. YAWN WARNING
            # -------------------------------------------------

            if yawn_data["yawn_active"]:

                cv2.putText(
                    frame,
                    "YAWN DETECTED!",
                    (30, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

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
        # 16. DISPLAY FRAME
        # -----------------------------------------------------

        cv2.imshow(
            "Driver Drowsiness Detection - Yawn Analysis",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ---------------------------------------------------------
    # 17. CLEANUP
    # ---------------------------------------------------------

    camera.release()
    face_detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()