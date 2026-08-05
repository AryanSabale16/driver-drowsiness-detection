import time


class TemporalAnalyzer:
    """
    Analyzes eye state over time.

    Responsibilities:
    - Detect normal blinks
    - Measure eye-closure duration
    - Detect prolonged eye closure
    - Count total blinks
    """

    def __init__(
        self,
        min_blink_duration=0.08,
        max_blink_duration=0.50,
        prolonged_closure_threshold=1.5
    ):
        self.min_blink_duration = min_blink_duration
        self.max_blink_duration = max_blink_duration
        self.prolonged_closure_threshold = prolonged_closure_threshold

        self.eye_closed_start = None

        self.total_blinks = 0

        self.prolonged_closure = False

    def update(self, eye_state):
        """
        Update temporal eye analysis using the current eye state.
        """

        current_time = time.time()

        blink_detected = False
        closure_duration = 0.0

        # Eyes have just become closed
        if eye_state == "CLOSED":

            if self.eye_closed_start is None:
                self.eye_closed_start = current_time

            closure_duration = (
                current_time - self.eye_closed_start
            )

            # Check prolonged closure
            if closure_duration >= self.prolonged_closure_threshold:
                self.prolonged_closure = True

        # Eyes are open
        else:

            if self.eye_closed_start is not None:

                closure_duration = (
                    current_time - self.eye_closed_start
                )

                # Determine whether the closure was a normal blink
                if (
                    self.min_blink_duration
                    <= closure_duration
                    <= self.max_blink_duration
                ):
                    self.total_blinks += 1
                    blink_detected = True

                # Reset closure tracking
                self.eye_closed_start = None

            self.prolonged_closure = False

        return {
            "blink_detected": blink_detected,
            "total_blinks": self.total_blinks,
            "closure_duration": closure_duration,
            "prolonged_closure": self.prolonged_closure
        }