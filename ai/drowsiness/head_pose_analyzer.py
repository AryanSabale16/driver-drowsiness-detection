import time


class HeadPoseAnalyzer:

    def __init__(self, prolonged_threshold=1.5):

        # Time required before a downward pose
        # is considered prolonged.
        self.prolonged_threshold = prolonged_threshold

        self.downward_start_time = None
        self.total_prolonged_events = 0

        self.prolonged_active = False

    def update(self, head_state):

        current_time = time.time()

        # -------------------------------------------------
        # DOWNWARD HEAD POSITION
        # -------------------------------------------------

        if head_state == "DOWN":

            # Start timing when DOWN is detected
            if self.downward_start_time is None:

                self.downward_start_time = current_time

            # Calculate how long the head has been down
            downward_duration = (
                current_time - self.downward_start_time
            )

            # Check whether it has become prolonged
            if (
                downward_duration >= self.prolonged_threshold
                and not self.prolonged_active
            ):

                self.prolonged_active = True
                self.total_prolonged_events += 1

        # -------------------------------------------------
        # HEAD IS NO LONGER DOWN
        # -------------------------------------------------

        else:

            self.downward_start_time = None
            self.prolonged_active = False
            downward_duration = 0.0

        return {
            "downward_duration": downward_duration,
            "prolonged_downward": self.prolonged_active,
            "total_prolonged_events": (
                self.total_prolonged_events
            )
        }