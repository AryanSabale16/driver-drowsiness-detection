import time
from collections import deque


class BlinkRateAnalyzer:
    """
    Calculates blink frequency using a rolling time window.

    Blink frequency is returned as estimated blinks per minute.
    """

    def __init__(self, window_duration=20.0):

        self.window_duration = window_duration

        # Stores timestamps of detected blinks
        self.blink_times = deque()

        # Keeps track of the previous total blink count
        self.previous_blink_count = 0

        # Used for startup normalization
        self.start_time = time.time()

    def update(self, total_blinks):
        """
        Update blink history and calculate blink frequency.

        Parameters
        ----------
        total_blinks : int
            Total blink count produced by TemporalAnalyzer.

        Returns
        -------
        float
            Estimated blinks per minute.
        """

        current_time = time.time()

        # -----------------------------------------------------
        # DETECT NEW BLINK EVENT
        # -----------------------------------------------------

        if total_blinks > self.previous_blink_count:

            new_blinks = (
                total_blinks - self.previous_blink_count
            )

            # Normally new_blinks will be 1.
            # This also safely handles skipped updates.
            for _ in range(new_blinks):
                self.blink_times.append(current_time)

        self.previous_blink_count = total_blinks

        # -----------------------------------------------------
        # REMOVE BLINKS OUTSIDE ROLLING WINDOW
        # -----------------------------------------------------

        cutoff_time = (
            current_time - self.window_duration
        )

        while (
            self.blink_times
            and self.blink_times[0] < cutoff_time
        ):
            self.blink_times.popleft()

        # -----------------------------------------------------
        # CALCULATE OBSERVATION DURATION
        # -----------------------------------------------------

        elapsed_time = (
            current_time - self.start_time
        )

        observation_duration = min(
            elapsed_time,
            self.window_duration
        )

        if observation_duration <= 0:
            return 0.0

        # -----------------------------------------------------
        # CONVERT TO BLINKS PER MINUTE
        # -----------------------------------------------------

        blink_rate = (
            len(self.blink_times)
            / observation_duration
        ) * 60.0

        return blink_rate