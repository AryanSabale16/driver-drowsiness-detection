from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)

from sqlalchemy.orm import Mapped, mapped_column

from backend.database.database import Base


# ============================================================
# DROWSINESS SESSION
# ============================================================

class DrowsinessSession(Base):

    __tablename__ = "drowsiness_sessions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    start_time: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    end_time: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        default="ACTIVE",
        nullable=False,
    )


# ============================================================
# DROWSINESS EVENT
# ============================================================

class DrowsinessEvent(Base):

    __tablename__ = "drowsiness_events"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "drowsiness_sessions.id"
        ),
        nullable=False,
        index=True,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True,
    )

    # --------------------------------------------------------
    # HYBRID INTELLIGENCE
    # --------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    drowsiness_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    behavior_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    yolo_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    yolo_class: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    yolo_confidence: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    # --------------------------------------------------------
    # EYES
    # --------------------------------------------------------

    ear: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    eye_state: Mapped[str] = mapped_column(
        String(20),
        default="UNKNOWN",
    )

    perclos: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    blink_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    # --------------------------------------------------------
    # MOUTH
    # --------------------------------------------------------

    mar: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    mouth_state: Mapped[str] = mapped_column(
        String(20),
        default="UNKNOWN",
    )

    yawn_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    # --------------------------------------------------------
    # HEAD POSE
    # --------------------------------------------------------

    head_state: Mapped[str] = mapped_column(
        String(20),
        default="UNKNOWN",
    )

    pitch: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    yaw: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    # --------------------------------------------------------
    # ALERT
    # --------------------------------------------------------

    alert_level: Mapped[str] = mapped_column(
        String(20),
        default="NONE",
    )

    alarm_active: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )

    # --------------------------------------------------------
    # REASONS
    # --------------------------------------------------------

    reasons: Mapped[str] = mapped_column(
        Text,
        default="",
    )