import time
from collections import deque


class PerclosAnalyzer:
    """
    Calculates PERCLOS over a rolling time window.

    PERCLOS represents the percentage of recent
    observed time during which the eyes were closed.
    """

    def __init__(self, window_duration=20.0):

        # Length of rolling observation window
        self.window_duration = window_duration

        # Stores:
        # (timestamp, eye_state)
        self.history = deque()

    def update(self, eye_state):
        """
        Add the current eye state and calculate PERCLOS.
        """

        current_time = time.time()

        # Store current observation
        self.history.append(
            (current_time, eye_state)
        )

        # -----------------------------------------------------
        # REMOVE OLD OBSERVATIONS
        # -----------------------------------------------------

        cutoff_time = (
            current_time - self.window_duration
        )

        while (
            self.history
            and self.history[0][0] < cutoff_time
        ):
            self.history.popleft()

        # -----------------------------------------------------
        # CALCULATE PERCLOS
        # -----------------------------------------------------

        total_samples = len(self.history)

        if total_samples == 0:
            return 0.0

        closed_samples = sum(
            1
            for _, state in self.history
            if state == "CLOSED"
        )

        perclos = (
            closed_samples / total_samples
        ) * 100.0

        return perclos