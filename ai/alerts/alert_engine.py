class AlertEngine:

    def __init__(self):
        self.current_status = "ALERT"
        self.last_alert_status = "ALERT"

    def evaluate(self, drowsiness_result):
        """
        Evaluate the output of the Drowsiness Intelligence Engine.

        Expected input:
        {
            "score": int,
            "status": str,
            "reasons": list
        }
        """

        score = drowsiness_result["score"]
        status = drowsiness_result["status"]
        reasons = drowsiness_result["reasons"]

        self.current_status = status

        alert_required = False
        alert_level = "NONE"

        if status == "DROWSY":
            alert_required = True
            alert_level = "WARNING"

        elif status == "CRITICAL":
            alert_required = True
            alert_level = "CRITICAL"

        return {
            "alert_required": alert_required,
            "alert_level": alert_level,
            "status": status,
            "score": score,
            "reasons": reasons
        }