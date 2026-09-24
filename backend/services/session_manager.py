import json
import time
from datetime import datetime

from sqlalchemy.orm import Session

from backend.database.models import (
    DrowsinessEvent,
    DrowsinessSession,
)


class SessionManager:
    """
    Manages drowsiness detection sessions and meaningful
    drowsiness events.

    A short loss of driver detection does not immediately
    terminate the session. This prevents session fragmentation
    caused by temporary YOLO/MediaPipe detection failures.
    """

    DRIVER_LOSS_GRACE_PERIOD = 3.0

    def __init__(self):

        self.active_session_id = None

        self.last_event_signature = None

        self.driver_last_seen = None

    # ========================================================
    # START SESSION
    # ========================================================

    def start_session(
        self,
        db: Session
    ) -> int:

        if self.active_session_id is not None:

            self.driver_last_seen = time.monotonic()

            return self.active_session_id

        session = DrowsinessSession(
            status="ACTIVE"
        )

        db.add(session)
        db.commit()
        db.refresh(session)

        self.active_session_id = session.id

        self.last_event_signature = None

        self.driver_last_seen = time.monotonic()

        return session.id

    # ========================================================
    # DRIVER DETECTED
    # ========================================================

    def driver_detected(
        self,
        db: Session
    ):

        self.driver_last_seen = time.monotonic()

        if self.active_session_id is None:

            return self.start_session(db)

        return self.active_session_id

    # ========================================================
    # DRIVER LOST
    # ========================================================

    def driver_lost(
        self,
        db: Session
    ):

        if self.active_session_id is None:

            return False

        if self.driver_last_seen is None:

            self.driver_last_seen = time.monotonic()

            return False

        elapsed = (
            time.monotonic()
            - self.driver_last_seen
        )

        # Keep session alive during short detection loss.
        if elapsed < self.DRIVER_LOSS_GRACE_PERIOD:

            return False

        # Driver has been absent long enough.
        self.end_session(db)

        return True

    # ========================================================
    # END SESSION
    # ========================================================

    def end_session(
        self,
        db: Session
    ):

        if self.active_session_id is None:

            return

        session = db.get(
            DrowsinessSession,
            self.active_session_id
        )

        if session is not None:

            session.end_time = datetime.utcnow()

            session.status = "COMPLETED"

            db.commit()

        self.active_session_id = None

        self.last_event_signature = None

        self.driver_last_seen = None

    # ========================================================
    # RECORD EVENT
    # ========================================================

    def record_event(
        self,
        db: Session,
        data: dict
    ):

        if self.active_session_id is None:

            return None

        signature = (
            data.get("status"),
            data.get("alert_level"),
            data.get("alarm_active"),
        )

        # Do not repeatedly store the exact same state.
        if signature == self.last_event_signature:

            return None

        self.last_event_signature = signature

        reasons = data.get(
            "reasons",
            []
        )

        event = DrowsinessEvent(

            session_id=self.active_session_id,

            status=data.get(
                "status",
                "ALERT"
            ),

            drowsiness_score=float(
                data.get(
                    "drowsiness_score",
                    0.0
                )
            ),

            behavior_score=float(
                data.get(
                    "behavior_score",
                    0.0
                )
            ),

            yolo_score=float(
                data.get(
                    "yolo_score",
                    0.0
                )
            ),

            yolo_class=data.get(
                "yolo_class"
            ),

            yolo_confidence=float(
                data.get(
                    "yolo_confidence",
                    0.0
                )
            ),

            ear=float(
                data.get(
                    "ear",
                    0.0
                )
            ),

            eye_state=data.get(
                "eye_state",
                "UNKNOWN"
            ),

            perclos=float(
                data.get(
                    "perclos",
                    0.0
                )
            ),

            blink_count=int(
                data.get(
                    "blink_count",
                    0
                )
            ),

            mar=float(
                data.get(
                    "mar",
                    0.0
                )
            ),

            mouth_state=data.get(
                "mouth_state",
                "UNKNOWN"
            ),

            yawn_count=int(
                data.get(
                    "yawn_count",
                    0
                )
            ),

            head_state=data.get(
                "head_state",
                "UNKNOWN"
            ),

            pitch=float(
                data.get(
                    "pitch",
                    0.0
                )
            ),

            yaw=float(
                data.get(
                    "yaw",
                    0.0
                )
            ),

            alert_level=data.get(
                "alert_level",
                "NONE"
            ),

            alarm_active=bool(
                data.get(
                    "alarm_active",
                    False
                )
            ),

            reasons=json.dumps(
                reasons
            ),
        )

        db.add(event)

        db.commit()

        db.refresh(event)

        return event.id