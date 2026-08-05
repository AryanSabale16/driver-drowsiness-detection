import cv2

from ai.metrics.mar import calculate_mar


class MouthDetector:
    """
    Extracts mouth landmarks and calculates
    Mouth Aspect Ratio (MAR).
    """

    # Selected MediaPipe mouth landmarks
    MOUTH_INDICES = [
        61,     # left mouth corner
        13,     # upper inner lip
        291,    # right mouth corner
        14,     # lower inner lip
    ]

    def __init__(self, mar_threshold=0.30):
        self.mar_threshold = mar_threshold

    def _landmark_to_pixel(
        self,
        landmark,
        frame_width,
        frame_height
    ):
        """
        Convert MediaPipe normalized coordinates
        into pixel coordinates.
        """

        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)

        return x, y

    def get_mouth_points(
        self,
        face_landmarks,
        frame_width,
        frame_height
    ):
        """
        Extract selected mouth landmarks.
        """

        landmarks = face_landmarks.landmark

        mouth_points = [
            self._landmark_to_pixel(
                landmarks[index],
                frame_width,
                frame_height
            )
            for index in self.MOUTH_INDICES
        ]

        return mouth_points

    def calculate_mouth_mar(
        self,
        face_landmarks,
        frame_width,
        frame_height
    ):
        """
        Calculate MAR from the selected
        mouth landmarks.
        """

        mouth_points = self.get_mouth_points(
            face_landmarks,
            frame_width,
            frame_height
        )

        mar = calculate_mar(mouth_points)

        return mar

    def classify_mouth_state(self, mar):
        """
        Basic mouth OPEN / CLOSED classification.
        """

        if mar > self.mar_threshold:
            return "OPEN"

        return "CLOSED"

    def draw_mouth_points(
        self,
        frame,
        mouth_points
    ):
        """
        Draw selected mouth landmarks.
        """

        for point in mouth_points:

            cv2.circle(
                frame,
                point,
                3,
                (255, 0, 255),
                -1
            )

        return frame