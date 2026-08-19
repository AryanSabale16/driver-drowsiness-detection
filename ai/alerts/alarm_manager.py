import threading
import os
import pygame


class AlarmManager:

    def __init__(self):

        self.is_active = False
        self.current_level = "NONE"

        self._stop_event = threading.Event()
        self._alarm_thread = None

        project_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                ".."
            )
        )

        self.warning_sound = os.path.join(
            project_root,
            "assets",
            "sounds",
            "warning.wav"
        )

        self.critical_sound = os.path.join(
            project_root,
            "assets",
            "sounds",
            "critical.wav"
        )

        # Initialize pygame audio
        pygame.mixer.init()


    def _alarm_loop(self, level):

        if level == "WARNING":
            sound_file = self.warning_sound
            message = "🔊 WARNING ALARM ACTIVE"

        elif level == "CRITICAL":
            sound_file = self.critical_sound
            message = "🚨 CRITICAL ALARM ACTIVE"

        else:
            return

        if not os.path.exists(sound_file):
            print(f"Alarm sound not found: {sound_file}")
            return

        try:

            pygame.mixer.music.load(sound_file)

            pygame.mixer.music.play(-1)

            print(message)

            # Keep the alarm alive until stop() is called
            while not self._stop_event.wait(0.1):
                pass

        except Exception as e:

            print(f"Alarm playback error: {e}")

        finally:

            pygame.mixer.music.stop()


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

        # Stop existing alarm before starting another one.
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

        # Immediately stop currently playing audio.
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

        # Wait for the alarm thread to finish.
        if (
            self._alarm_thread is not None
            and self._alarm_thread.is_alive()
            and self._alarm_thread != threading.current_thread()
        ):
            self._alarm_thread.join(timeout=1.5)

        self.is_active = False
        self.current_level = "NONE"
        self._alarm_thread = None


    def shutdown(self):

        self.stop()

        try:
            pygame.mixer.quit()
        except Exception:
            pass