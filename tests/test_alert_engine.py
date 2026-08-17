from ai.alerts.alert_engine import AlertEngine


def main():

    engine = AlertEngine()

    test_cases = [

        {
            "name": "Normal Driver",
            "data": {
                "score": 0,
                "status": "ALERT",
                "reasons": []
            }
        },

        {
            "name": "Drowsy Driver",
            "data": {
                "score": 55,
                "status": "DROWSY",
                "reasons": [
                    "Elevated PERCLOS",
                    "Prolonged eye closure detected"
                ]
            }
        },

        {
            "name": "Critical Driver",
            "data": {
                "score": 85,
                "status": "CRITICAL",
                "reasons": [
                    "High PERCLOS",
                    "Prolonged eye closure detected",
                    "Sustained yawn detected"
                ]
            }
        }
    ]

    for test in test_cases:

        result = engine.evaluate(test["data"])

        print("=" * 60)
        print(test["name"])
        print("=" * 60)

        print("Score        :", result["score"])
        print("Status       :", result["status"])
        print("Alert        :", result["alert_required"])
        print("Alert Level  :", result["alert_level"])
        print("Reasons      :", result["reasons"])


if __name__ == "__main__":
    main()