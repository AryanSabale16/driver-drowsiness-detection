import threading
import time


class AlarmManager:

    def __init__(self):

        self.is_active = False
        self.current_level = "NONE"

        self._stop_event = threading.Event()
        self._alarm_thread = None

    def _alarm_loop(self, level):

        while not self._stop_event.wait(1):

            if level == "WARNING":
                print("🔊 WARNING ALARM ACTIVE")

            elif level == "CRITICAL":
                print("🚨 CRITICAL ALARM ACTIVE")

            # Wait instead of using time.sleep().
            # This allows the thread to stop immediately.
            self._stop_event.wait(1)

    def _start_alarm(self, level):

        # If the requested alarm is already active,
        # don't create another thread.
        if (
            self.is_active
            and self.current_level == level
            and self._alarm_thread is not None
            and self._alarm_thread.is_alive()
        ):
            return

        # Stop any existing alarm first.
        self.stop()

        self.is_active = True
        self.current_level = level

        self._stop_event.clear()

        self._alarm_thread = threading.Thread(
            target=self._alarm_loop,
            args=(level,),
            daemon=True
        )

        self._alarm_thread.start()

    def start_warning(self):

        self._start_alarm("WARNING")

    def start_critical(self):

        self._start_alarm("CRITICAL")

    def stop(self):

        self._stop_event.set()

        # Wait for the existing alarm thread to finish.
        if (
            self._alarm_thread is not None
            and self._alarm_thread.is_alive()
            and self._alarm_thread != threading.current_thread()
        ):
            self._alarm_thread.join(timeout=1.5)

        self.is_active = False
        self.current_level = "NONE"
        self._alarm_thread = None