import cv2

from ai.metrics.ear import calculate_ear


class EyeDetector:
    """
    Extracts eye landmarks and calculates Eye Aspect Ratio (EAR).
    """

    # Six landmarks for each eye used for EAR calculation
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    def __init__(self, ear_threshold=0.20):
        self.ear_threshold = ear_threshold

    def _landmark_to_pixel(self, landmark, frame_width, frame_height):
        """
        Convert MediaPipe normalized coordinates into pixel coordinates.
        """

        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)

        return x, y

    def classify_eye_state(self, average_ear):
        if average_ear < self.ear_threshold:
            return "CLOSED"
        return "OPEN"

    def get_eye_points(self, face_landmarks, frame_width, frame_height):
        """
        Extract left and right eye points from MediaPipe landmarks.
        """

        landmarks = face_landmarks.landmark

        left_eye = [
            self._landmark_to_pixel(
                landmarks[index],
                frame_width,
                frame_height
            )
            for index in self.LEFT_EYE_INDICES
        ]

        right_eye = [
            self._landmark_to_pixel(
                landmarks[index],
                frame_width,
                frame_height
            )
            for index in self.RIGHT_EYE_INDICES
        ]

        return left_eye, right_eye

    def calculate_eye_ear(
        self,
        face_landmarks,
        frame_width,
        frame_height
    ):
        """
        Calculate EAR for both eyes and their average.
        """

        left_eye, right_eye = self.get_eye_points(
            face_landmarks,
            frame_width,
            frame_height
        )

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        average_ear = (left_ear + right_ear) / 2.0

        return left_ear, right_ear, average_ear

    def draw_eye_points(self, frame, left_eye, right_eye):
        """
        Draw selected EAR landmarks for debugging.
        """

        for point in left_eye + right_eye:
            cv2.circle(
                frame,
                point,
                3,
                (0, 255, 0),
                -1
            )

        return frame