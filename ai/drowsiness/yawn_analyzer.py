import time


class YawnAnalyzer:
    """
    Analyzes mouth opening over time to detect yawns.

    A yawn is detected when MAR remains above
    the threshold for a sustained duration.
    """

    def __init__(
        self,
        mar_threshold=0.55,
        yawn_duration_threshold=1.5
    ):
        self.mar_threshold = mar_threshold
        self.yawn_duration_threshold = yawn_duration_threshold

        # Time when a large mouth opening started
        self.mouth_open_start = None

        # Total number of detected yawns
        self.total_yawns = 0

        # Prevents counting the same yawn repeatedly
        self.yawn_active = False

    def update(self, mar):
        """
        Analyze the current MAR value.

        Returns information about:
        - mouth opening duration
        - whether a yawn was detected
        - whether a yawn is currently active
        - total yawn count
        """

        current_time = time.time()

        mouth_open_duration = 0.0
        yawn_detected = False

        # -----------------------------------------------------
        # MAR IS ABOVE YAWN THRESHOLD
        # -----------------------------------------------------

        if mar >= self.mar_threshold:

            # Start timing the mouth opening
            if self.mouth_open_start is None:
                self.mouth_open_start = current_time

            mouth_open_duration = (
                current_time - self.mouth_open_start
            )

            # Check whether mouth has remained open
            # long enough to qualify as a yawn
            if (
                mouth_open_duration
                >= self.yawn_duration_threshold
            ):

                # Only count once per yawn
                if not self.yawn_active:

                    self.total_yawns += 1
                    yawn_detected = True
                    self.yawn_active = True

        # -----------------------------------------------------
        # MAR DROPPED BELOW THRESHOLD
        # -----------------------------------------------------

        else:

            self.mouth_open_start = None
            self.yawn_active = False

        return {
            "yawn_detected": yawn_detected,
            "yawn_active": self.yawn_active,
            "total_yawns": self.total_yawns,
            "mouth_open_duration": mouth_open_duration
        }