import cv2

from ai.detection.face_mesh import FaceMeshDetector
from ai.detection.head_pose_detector import HeadPoseDetector


def main():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    face_detector = FaceMeshDetector()
    head_pose_detector = HeadPoseDetector()

    print("Head Pose Test Started")
    print("Press Q to exit.")

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)

        results = face_detector.detect(frame)

        if results.multi_face_landmarks:

            face_landmarks = (
                results.multi_face_landmarks[0]
            )

            frame_height, frame_width = (
                frame.shape[:2]
            )

            pose = head_pose_detector.calculate_pose(
                face_landmarks,
                frame_width,
                frame_height
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

                # ---------------------------------------------
                # DISPLAY ANGLES
                # ---------------------------------------------

                cv2.putText(
                    frame,
                    f"Pitch: {pitch:.1f}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Yaw: {yaw:.1f}",
                    (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Roll: {roll:.1f}",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                # ---------------------------------------------
                # DISPLAY HEAD STATE
                # ---------------------------------------------

                if head_state == "FORWARD":
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cv2.putText(
                    frame,
                    f"Head: {head_state}",
                    (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2
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

        cv2.imshow(
            "Driver Drowsiness Detection - Head Pose Test",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    face_detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()