import cv2
import numpy as np


class HeadPoseDetector:

    def __init__(
        self,
        yaw_threshold=20,
        pitch_threshold=15
    ):

        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold

        # MediaPipe facial landmark indices
        #
        # 1   -> Nose
        # 152 -> Chin
        # 33  -> Left eye
        # 263 -> Right eye
        # 61  -> Left mouth corner
        # 291 -> Right mouth corner

        self.landmark_indices = [
            1,
            152,
            33,
            263,
            61,
            291
        ]

        # Approximate 3D facial model
        self.model_points = np.array(
            [
                (0.0, 0.0, 0.0),          # Nose
                (0.0, -63.6, -12.5),      # Chin
                (-43.3, 32.7, -26.0),     # Left eye
                (43.3, 32.7, -26.0),      # Right eye
                (-28.9, -28.9, -24.1),    # Left mouth
                (28.9, -28.9, -24.1)      # Right mouth
            ],
            dtype=np.float64
        )

    # ---------------------------------------------------------
    # GET FACIAL LANDMARK IMAGE POINTS
    # ---------------------------------------------------------

    def get_image_points(
        self,
        face_landmarks,
        frame_width,
        frame_height
    ):

        image_points = []

        for index in self.landmark_indices:

            landmark = face_landmarks.landmark[index]

            x = landmark.x * frame_width
            y = landmark.y * frame_height

            image_points.append((x, y))

        return np.array(
            image_points,
            dtype=np.float64
        )

    # ---------------------------------------------------------
    # CALCULATE HEAD ROTATION
    # ---------------------------------------------------------

    def calculate_pose(
        self,
        face_landmarks,
        frame_width,
        frame_height
    ):

        image_points = self.get_image_points(
            face_landmarks,
            frame_width,
            frame_height
        )

        # Approximate camera focal length
        focal_length = frame_width

        # Camera center
        center = (
            frame_width / 2,
            frame_height / 2
        )

        # Camera matrix
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ],
            dtype=np.float64
        )

        # Assume no lens distortion
        distortion_coefficients = np.zeros(
            (4, 1),
            dtype=np.float64
        )

        # Estimate head rotation
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            distortion_coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(
            rotation_vector
        )

        # Extract Euler angles
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(
            rotation_matrix
        )

        pitch = angles[0]
        yaw = angles[1]
        roll = angles[2]

        # -----------------------------------------------------
        # NORMALIZE PITCH
        # -----------------------------------------------------
        #
        # RQDecomp3x3 can return pitch values close to
        # +180 or -180 instead of values around 0.
        #
        # Convert those values into a more intuitive range.

        if pitch > 90:
            pitch = pitch - 180

        elif pitch < -90:
            pitch = pitch + 180

        # -----------------------------------------------------
        # NORMALIZE ROLL
        # -----------------------------------------------------

        if roll > 90:
            roll = roll - 180

        elif roll < -90:
            roll = roll + 180

        return {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll
        }

    # ---------------------------------------------------------
    # CLASSIFY HEAD POSITION
    # ---------------------------------------------------------

    def classify_pose(
        self,
        pitch,
        yaw
    ):

        # Webcam is mirrored, so yaw direction appears reversed.
        #
        # Positive yaw  -> physical LEFT
        # Negative yaw  -> physical RIGHT

        if yaw > self.yaw_threshold:
            return "LEFT"

        if yaw < -self.yaw_threshold:
            return "RIGHT"

        # Pitch direction is also reversed in our
        # coordinate representation.
        #
        # Positive pitch -> physical DOWN
        # Negative pitch -> physical UP

        if pitch > self.pitch_threshold:
            return "DOWN"

        if pitch < -self.pitch_threshold:
            return "UP"

        return "FORWARD"