class HybridDrowsinessAnalyzer:
    """
    Hybrid YOLO + MediaPipe drowsiness fusion.

    YOLO:
        Provides learned alert/drowsy prediction
        and confidence.

    MediaPipe + temporal intelligence:
        Provides behavioral drowsiness score.

    Class mapping:
        0 -> alert
        1 -> drowsy
    """

    MAX_BEHAVIOR_SCORE = 115.0

    def __init__(
        self,
        behavior_weight=0.70,
        yolo_weight=0.30
    ):
        self.behavior_weight = behavior_weight
        self.yolo_weight = yolo_weight

        if abs(
            (behavior_weight + yolo_weight) - 1.0
        ) > 1e-6:
            raise ValueError(
                "Fusion weights must add up to 1.0"
            )

    def analyze(
        self,
        behavior_data,
        yolo_class_id,
        yolo_confidence
    ):
        """
        Combine behavioral intelligence with
        YOLO drowsiness prediction.

        Args:
            behavior_data:
                Output from DrowsinessAnalyzer.

            yolo_class_id:
                0 = alert
                1 = drowsy

            yolo_confidence:
                YOLO confidence between 0 and 1.

        Returns:
            Dictionary containing the hybrid result.
        """

        # =====================================================
        # VALIDATE YOLO INPUT
        # =====================================================

        if yolo_class_id not in (0, 1):
            raise ValueError(
                f"Invalid YOLO class ID: {yolo_class_id}"
            )

        yolo_confidence = max(
            0.0,
            min(1.0, float(yolo_confidence))
        )

        # =====================================================
        # BEHAVIORAL SCORE
        # =====================================================

        behavior_score = float(
            behavior_data.get("score", 0.0)
        )

        behavior_normalized = (
            behavior_score
            / self.MAX_BEHAVIOR_SCORE
        ) * 100.0

        behavior_normalized = max(
            0.0,
            min(100.0, behavior_normalized)
        )

        # =====================================================
        # YOLO DROWSINESS SCORE
        # =====================================================

        if yolo_class_id == 1:
            # YOLO predicts DROWSY.
            # Confidence becomes drowsiness evidence.
            yolo_drowsiness_score = (
                yolo_confidence * 100.0
            )
        else:
            # YOLO predicts ALERT.
            # Do NOT convert low alert confidence
            # into drowsiness evidence.
            yolo_drowsiness_score = 0.0

        # =====================================================
        # HYBRID FUSION
        # =====================================================

        hybrid_score = (
            self.behavior_weight
            * behavior_normalized
        ) + (
            self.yolo_weight
            * yolo_drowsiness_score
        )

        hybrid_score = max(
            0.0,
            min(100.0, hybrid_score)
        )

        # =====================================================
        # FINAL STATUS
        # =====================================================

        if hybrid_score >= 75:
            status = "CRITICAL"

        elif hybrid_score >= 50:
            status = "DROWSY"

        elif hybrid_score >= 25:
            status = "CAUTION"

        else:
            status = "ALERT"

        # =====================================================
        # REASONS
        # =====================================================

        reasons = list(
            behavior_data.get("reasons", [])
        )

        if yolo_class_id == 1:
            reasons.append(
                f"YOLO drowsy prediction "
                f"({yolo_confidence:.2f})"
            )
        else:
            reasons.append(
                f"YOLO alert prediction "
                f"({yolo_confidence:.2f})"
            )

        # =====================================================
        # RESULT
        # =====================================================

        return {
            "score": round(hybrid_score, 2),
            "status": status,

            "behavior_score": round(
                behavior_score,
                2
            ),

            "behavior_score_normalized": round(
                behavior_normalized,
                2
            ),

            "yolo_class_id": yolo_class_id,

            "yolo_class": (
                "drowsy"
                if yolo_class_id == 1
                else "alert"
            ),

            "yolo_confidence": round(
                yolo_confidence,
                4
            ),

            "yolo_drowsiness_score": round(
                yolo_drowsiness_score,
                2
            ),

            "behavior_weight": self.behavior_weight,
            "yolo_weight": self.yolo_weight,

            "reasons": reasons
        }