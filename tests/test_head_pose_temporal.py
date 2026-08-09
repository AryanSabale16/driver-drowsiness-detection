import cv2

from ai.detection.face_mesh import FaceMeshDetector
from ai.detection.head_pose_detector import HeadPoseDetector
from ai.drowsiness.head_pose_analyzer import HeadPoseAnalyzer


def main():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    face_detector = FaceMeshDetector()
    head_pose_detector = HeadPoseDetector()
    head_pose_analyzer = HeadPoseAnalyzer(
        prolonged_threshold=1.5
    )

    print("Head Pose Temporal Test Started")
    print("Press Q to exit.")

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)

        # -------------------------------------------------
        # FACE DETECTION
        # -------------------------------------------------

        results = face_detector.detect(frame)

        if results.multi_face_landmarks:

            face_landmarks = (
                results.multi_face_landmarks[0]
            )

            frame_height, frame_width = (
                frame.shape[:2]
            )

            # -------------------------------------------------
            # HEAD POSE
            # -------------------------------------------------

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

                # -------------------------------------------------
                # TEMPORAL HEAD ANALYSIS
                # -------------------------------------------------

                temporal_data = (
                    head_pose_analyzer.update(
                        head_state
                    )
                )

                # -------------------------------------------------
                # COLORS
                # -------------------------------------------------

                if head_state == "FORWARD":
                    head_color = (0, 255, 0)
                elif head_state == "DOWN":
                    head_color = (0, 0, 255)
                else:
                    head_color = (0, 255, 255)

                # -------------------------------------------------
                # DISPLAY ANGLES
                # -------------------------------------------------

                cv2.putText(
                    frame,
                    f"Pitch: {pitch:.1f}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Yaw: {yaw:.1f}",
                    (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Roll: {roll:.1f}",
                    (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                # -------------------------------------------------
                # HEAD STATE
                # -------------------------------------------------

                cv2.putText(
                    frame,
                    f"Head: {head_state}",
                    (30, 155),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    head_color,
                    2
                )

                # -------------------------------------------------
                # DOWNWARD DURATION
                # -------------------------------------------------

                cv2.putText(
                    frame,
                    (
                        "Down Duration: "
                        f"{temporal_data['downward_duration']:.2f}s"
                    ),
                    (30, 195),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                # -------------------------------------------------
                # PROLONGED DOWNWARD WARNING
                # -------------------------------------------------

                if temporal_data["prolonged_downward"]:

                    cv2.putText(
                        frame,
                        "WARNING: PROLONGED DOWNWARD HEAD",
                        (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
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

        # -------------------------------------------------
        # DISPLAY
        # -------------------------------------------------

        cv2.imshow(
            "Driver Drowsiness Detection - Head Pose Temporal Test",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    face_detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()