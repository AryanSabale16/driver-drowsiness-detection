import cv2

from ai.detection.face_mesh import FaceMeshDetector


def main():

    # Open default webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # Create our MediaPipe detector
    detector = FaceMeshDetector()

    print("Camera started successfully.")
    print("Press Q to exit.")

    while True:

        # Capture one frame
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        # Mirror the image
        frame = cv2.flip(frame, 1)

        # Detect facial landmarks
        results = detector.detect(frame)

        # Draw facial landmarks
        frame = detector.draw_landmarks(frame, results)

        # Display video
        cv2.imshow(
            "Driver Drowsiness Detection - Face Mesh Test",
            frame
        )

        # Exit when Q is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up resources
    camera.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()