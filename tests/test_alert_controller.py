import time

from ai.alerts.alert_controller import AlertController


def main():

    controller = AlertController()

    tests = [

        {
            "name": "Normal",
            "data": {
                "alert_level": "NONE"
            }
        },

        {
            "name": "Warning",
            "data": {
                "alert_level": "WARNING"
            }
        },

        {
            "name": "Critical",
            "data": {
                "alert_level": "CRITICAL"
            }
        }
    ]

    for test in tests:

        print("=" * 60)
        print(test["name"])
        print("=" * 60)

        result = controller.process(test["data"])

        print("Alert Level  :", result["alert_level"])
        print("Alarm Active :", result["alarm_active"])
        print("Alarm Status :", result["alarm_status"])

        time.sleep(2)

    print("\nStopping controller...")

    controller.stop()

    print("Alarm Active :", controller.alarm_manager.is_active)
    print("Alarm Status :", controller.alarm_manager.current_level)


if __name__ == "__main__":
    main()