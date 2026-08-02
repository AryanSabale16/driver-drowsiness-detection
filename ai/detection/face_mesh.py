import cv2
import mediapipe as mp


class FaceMeshDetector:
    """
    Detects facial landmarks using MediaPipe Face Mesh.
    """

    def __init__(
        self,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ):
        # MediaPipe Face Mesh module
        self.mp_face_mesh = mp.solutions.face_mesh

        # Create the Face Mesh detector
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Utility used to draw landmarks
        self.mp_drawing = mp.solutions.drawing_utils

        self.drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=1,
            circle_radius=1
        )

    def detect(self, frame):
        """
        Detect facial landmarks in an OpenCV frame.
        """

        # OpenCV uses BGR.
        # MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.face_mesh.process(rgb_frame)

        return results

    def draw_landmarks(self, frame, results):
        """
        Draw detected facial landmarks on the frame.
        """

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec
                )

        return frame

    def close(self):
        """
        Release MediaPipe resources.
        """

        self.face_mesh.close()