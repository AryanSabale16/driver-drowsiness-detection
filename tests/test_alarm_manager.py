import time

from ai.alerts.alarm_manager import AlarmManager


def main():

    alarm_manager = AlarmManager()

    print("=" * 60)
    print("ALARM MANAGER TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # TEST 1 - Warning Alarm
    # ---------------------------------------------------------

    print("\nTEST 1 - Starting warning alarm")

    alarm_manager.start_warning()

    time.sleep(3)

    print("Alarm active:", alarm_manager.is_active)
    print("Alarm level :", alarm_manager.current_level)

    # ---------------------------------------------------------
    # TEST 2 - Critical Alarm
    # ---------------------------------------------------------

    print("\nTEST 2 - Switching to critical alarm")

    alarm_manager.start_critical()

    time.sleep(3)

    print("Alarm active:", alarm_manager.is_active)
    print("Alarm level :", alarm_manager.current_level)

    # ---------------------------------------------------------
    # TEST 3 - Stop Alarm
    # ---------------------------------------------------------

    print("\nTEST 3 - Stopping alarm")

    alarm_manager.stop()

    time.sleep(1)

    print("Alarm active:", alarm_manager.is_active)
    print("Alarm level :", alarm_manager.current_level)

    print("\n" + "=" * 60)
    print("ALARM MANAGER TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()