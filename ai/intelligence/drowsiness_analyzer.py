class DrowsinessAnalyzer:
    """
    Rule-based drowsiness intelligence engine.

    Combines behavioural indicators from:
    - PERCLOS
    - Prolonged eye closure
    - Yawning
    - Prolonged downward head position

    Returns:
    - Drowsiness score
    - Severity status
    - Reasons contributing to the score
    """

    def __init__(self):

        # -----------------------------------------------------
        # SCORE CONTRIBUTIONS
        # -----------------------------------------------------

        self.prolonged_closure_score = 35
        self.yawn_score = 20
        self.head_down_score = 20

        # -----------------------------------------------------
        # PERCLOS THRESHOLDS
        # -----------------------------------------------------

        self.perclos_low = 10.0
        self.perclos_medium = 20.0
        self.perclos_high = 30.0
        self.perclos_critical = 40.0

    # =========================================================
    # PERCLOS SCORE
    # =========================================================

    def calculate_perclos_score(self, perclos):

        if perclos >= self.perclos_critical:
            return 40

        elif perclos >= self.perclos_high:
            return 30

        elif perclos >= self.perclos_medium:
            return 20

        elif perclos >= self.perclos_low:
            return 10

        return 0

    # =========================================================
    # MAIN ANALYSIS
    # =========================================================

    def analyze(
        self,
        perclos,
        temporal_data,
        yawn_data,
        head_pose_data
    ):

        score = 0
        reasons = []

        # -----------------------------------------------------
        # PERCLOS
        # -----------------------------------------------------

        perclos_score = self.calculate_perclos_score(
            perclos
        )

        score += perclos_score

        if perclos >= self.perclos_critical:

            reasons.append(
                f"Very high PERCLOS ({perclos:.1f}%)"
            )

        elif perclos >= self.perclos_high:

            reasons.append(
                f"High PERCLOS ({perclos:.1f}%)"
            )

        elif perclos >= self.perclos_medium:

            reasons.append(
                f"Elevated PERCLOS ({perclos:.1f}%)"
            )

        # -----------------------------------------------------
        # PROLONGED EYE CLOSURE
        # -----------------------------------------------------

        if temporal_data["prolonged_closure"]:

            score += self.prolonged_closure_score

            reasons.append(
                "Prolonged eye closure detected"
            )

        # -----------------------------------------------------
        # YAWN
        # -----------------------------------------------------

        if yawn_data["yawn_active"]:

            score += self.yawn_score

            reasons.append(
                "Sustained yawn detected"
            )

        # -----------------------------------------------------
        # PROLONGED HEAD DOWN
        # -----------------------------------------------------

        if head_pose_data["prolonged_downward"]:

            score += self.head_down_score

            reasons.append(
                "Prolonged downward head position"
            )

        # -----------------------------------------------------
        # LIMIT SCORE
        # -----------------------------------------------------

        score = min(score, 100)

        # -----------------------------------------------------
        # DETERMINE STATUS
        # -----------------------------------------------------

        if score >= 75:

            status = "CRITICAL"

        elif score >= 50:

            status = "DROWSY"

        elif score >= 25:

            status = "CAUTION"

        else:

            status = "ALERT"

        # -----------------------------------------------------
        # RETURN RESULT
        # -----------------------------------------------------

        return {
            "score": score,
            "status": status,
            "reasons": reasons
        }