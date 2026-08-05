import cv2

from ai.detection.face_mesh import FaceMeshDetector
from ai.detection.eye_detector import EyeDetector
from ai.drowsiness.temporal_analyzer import TemporalAnalyzer


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
    eye_detector = EyeDetector()

    # Keeps track of eye behaviour over time
    temporal_analyzer = TemporalAnalyzer()

    print("EAR + Temporal Analysis Test Started")
    print("Press Q to exit.")

    # ---------------------------------------------------------
    # 3. MAIN VIDEO LOOP
    # ---------------------------------------------------------

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        # Mirror webcam image
        frame = cv2.flip(frame, 1)

        # -----------------------------------------------------
        # 4. DETECT FACE LANDMARKS
        # -----------------------------------------------------

        results = face_detector.detect(frame)

        if results.multi_face_landmarks:

            # We only monitor one driver
            face_landmarks = results.multi_face_landmarks[0]

            frame_height, frame_width = frame.shape[:2]

            # -------------------------------------------------
            # 5. GET EYE LANDMARKS
            # -------------------------------------------------

            left_eye, right_eye = eye_detector.get_eye_points(
                face_landmarks,
                frame_width,
                frame_height
            )

            # -------------------------------------------------
            # 6. CALCULATE EAR
            # -------------------------------------------------

            left_ear, right_ear, average_ear = (
                eye_detector.calculate_eye_ear(
                    face_landmarks,
                    frame_width,
                    frame_height
                )
            )

            # -------------------------------------------------
            # 7. CLASSIFY EYE STATE
            # -------------------------------------------------

            eye_state = eye_detector.classify_eye_state(
                average_ear
            )

            # -------------------------------------------------
            # 8. TEMPORAL ANALYSIS
            # -------------------------------------------------

            temporal_data = temporal_analyzer.update(
                eye_state
            )

            # -------------------------------------------------
            # 9. DRAW EYE LANDMARKS
            # -------------------------------------------------

            eye_detector.draw_eye_points(
                frame,
                left_eye,
                right_eye
            )

            # -------------------------------------------------
            # 10. SELECT STATUS COLOR
            # -------------------------------------------------

            if eye_state == "CLOSED":
                status_color = (0, 0, 255)       # Red
            else:
                status_color = (0, 255, 0)       # Green

            # -------------------------------------------------
            # 11. DISPLAY EAR
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"EAR: {average_ear:.3f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2
            )

            # -------------------------------------------------
            # 12. DISPLAY EYE STATE
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"Eyes: {eye_state}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2
            )

            # -------------------------------------------------
            # 13. DISPLAY BLINK COUNT
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"Blinks: {temporal_data['total_blinks']}",
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            # -------------------------------------------------
            # 14. DISPLAY EYE CLOSURE DURATION
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"Closure: {temporal_data['closure_duration']:.2f}s",
                (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            # -------------------------------------------------
            # 15. PROLONGED EYE CLOSURE WARNING
            # -------------------------------------------------

            if temporal_data["prolonged_closure"]:

                cv2.putText(
                    frame,
                    "PROLONGED EYE CLOSURE!",
                    (30, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3
                )

        else:

            # No face detected
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
            "Driver Drowsiness Detection - Eye Analysis",
            frame
        )

        # Press Q to exit
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