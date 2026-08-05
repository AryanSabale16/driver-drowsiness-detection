import cv2

from ai.detection.face_mesh import FaceMeshDetector
from ai.detection.eye_detector import EyeDetector


def main():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    face_detector = FaceMeshDetector()
    eye_detector = EyeDetector()

    print("EAR test started.")
    print("Press Q to exit.")

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        # Detect facial landmarks
        results = face_detector.detect(frame)

        if results.multi_face_landmarks:

            face_landmarks = results.multi_face_landmarks[0]

            frame_height, frame_width = frame.shape[:2]

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
            eye_state = eye_detector.classify_eye_state(average_ear)

            # Draw only the selected eye points
            eye_detector.draw_eye_points(
                frame,
                left_eye,
                right_eye
            )

            
            if eye_state == "CLOSED":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            # Display EAR value

            cv2.putText(
                frame,
                f"EAR: {average_ear:.3f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

            # Display Eye State

            cv2.putText(
                frame,
                f"Eyes: {eye_state}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

        cv2.imshow(
            "Driver Drowsiness Detection - EAR Test",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    face_detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()