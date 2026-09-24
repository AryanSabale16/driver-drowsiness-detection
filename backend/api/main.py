from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database.database import (
    SessionLocal,
    init_database,
)

from backend.database.models import (
    DrowsinessEvent,
    DrowsinessSession,
)

from backend.services.session_manager import (
    SessionManager,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Driver Drowsiness Detection API",
    description=(
        "Backend API for the hybrid YOLO + MediaPipe "
        "driver drowsiness detection system."
    ),
    version="1.0.0",
)


# ============================================================
# CORS CONFIGURATION
# ============================================================
# Allows the React frontend running on Vite to communicate
# with this FastAPI backend.
#
# React:
#   http://localhost:5173
#
# FastAPI:
#   http://127.0.0.1:8000
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

init_database()


# ============================================================
# SESSION MANAGER
# ============================================================

session_manager = SessionManager()


# ============================================================
# CURRENT SYSTEM STATE
# ============================================================
# This contains the latest state received from the AI pipeline.
#
# The YOLO + MediaPipe pipeline sends updates to:
#     POST /status
#
# The React dashboard reads the latest state from:
#     GET /status
# ============================================================

system_state = {
    "status": "ALERT",
    "drowsiness_score": 0.0,

    # Intelligence scores
    "behavior_score": 0.0,
    "yolo_score": 0.0,

    # YOLO
    "yolo_class": None,
    "yolo_confidence": 0.0,

    # Eye / face
    "ear": 0.0,
    "eye_state": "UNKNOWN",
    "face_detected": False,

    # Temporal eye analysis
    "perclos": 0.0,
    "blink_count": 0,

    # Mouth
    "mar": 0.0,
    "mouth_state": "UNKNOWN",
    "yawn_count": 0,

    # Head pose
    "head_state": "UNKNOWN",
    "pitch": 0.0,
    "yaw": 0.0,

    # Alert system
    "alert_level": "NONE",
    "alarm_active": False,

    # Intelligence reasons
    "reasons": [],

    # Driver / session
    "driver_detected": False,
    "session_id": None,

    # Database event
    "last_event_id": None,

    # Timestamp
    "last_updated": None,
}


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    """
    Basic API information endpoint.
    """

    return {
        "message": "Driver Drowsiness Detection API",
        "status": "running",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    """
    Health check endpoint.

    Used to verify that the FastAPI backend is running.
    """

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# GET CURRENT SYSTEM STATUS
# ============================================================

@app.get("/status")
def get_status():
    """
    Return the latest AI system state.

    React dashboard uses this endpoint for live monitoring.
    """

    return system_state


# ============================================================
# UPDATE SYSTEM STATUS
# ============================================================

@app.post("/status")
def update_status(data: dict[str, Any]):
    """
    Receive the latest AI pipeline state.

    The YOLO + MediaPipe pipeline periodically sends data here.

    Responsibilities:
        1. Update current system state.
        2. Manage driver session.
        3. Record meaningful events.
        4. Return the updated state.
    """

    # --------------------------------------------------------
    # Update in-memory system state
    # --------------------------------------------------------

    system_state.update(data)

    system_state["last_updated"] = (
        datetime.now(timezone.utc).isoformat()
    )

    # --------------------------------------------------------
    # Database session
    # --------------------------------------------------------

    db = SessionLocal()

    try:

        driver_detected = bool(
            data.get("driver_detected", False)
        )

        # ====================================================
        # DRIVER DETECTED
        # ====================================================

        if driver_detected:

            # Start or continue the current session.
            session_id = session_manager.driver_detected(db)

            system_state["session_id"] = session_id

            # Record meaningful state changes.
            event_id = session_manager.record_event(
                db,
                data,
            )

        # ====================================================
        # DRIVER NOT DETECTED
        # ====================================================

        else:

            # SessionManager applies the configured grace period
            # before ending a session.
            session_manager.driver_lost(db)

            # If the session is still within the grace period,
            # keep its ID.
            if session_manager.active_session_id is not None:

                system_state["session_id"] = (
                    session_manager.active_session_id
                )

            else:

                system_state["session_id"] = None

            event_id = None

        # ----------------------------------------------------
        # Store latest event ID
        # ----------------------------------------------------

        system_state["last_event_id"] = event_id

    finally:

        db.close()

    return system_state


# ============================================================
# GET ALL SESSIONS
# ============================================================

@app.get("/sessions")
def get_sessions():
    """
    Return all recorded drowsiness sessions.

    Sessions are returned newest first.
    """

    db = SessionLocal()

    try:

        sessions = (
            db.query(DrowsinessSession)
            .order_by(DrowsinessSession.id.desc())
            .all()
        )

        return [
            {
                "id": session.id,

                "start_time": (
                    session.start_time.isoformat()
                    if session.start_time
                    else None
                ),

                "end_time": (
                    session.end_time.isoformat()
                    if session.end_time
                    else None
                ),

                "status": session.status,
            }

            for session in sessions
        ]

    finally:

        db.close()


# ============================================================
# GET EVENTS FOR A SESSION
# ============================================================

@app.get("/sessions/{session_id}/events")
def get_session_events(session_id: int):
    """
    Return all recorded events for a particular session.
    """

    db = SessionLocal()

    try:

        events = (
            db.query(DrowsinessEvent)
            .filter(
                DrowsinessEvent.session_id == session_id
            )
            .order_by(
                DrowsinessEvent.timestamp.asc()
            )
            .all()
        )

        return [
            {
                "id": event.id,
                "session_id": event.session_id,

                "timestamp": (
                    event.timestamp.isoformat()
                ),

                # --------------------------------------------
                # Intelligence
                # --------------------------------------------

                "status": event.status,
                "drowsiness_score": event.drowsiness_score,
                "behavior_score": event.behavior_score,
                "yolo_score": event.yolo_score,

                # --------------------------------------------
                # YOLO
                # --------------------------------------------

                "yolo_class": event.yolo_class,
                "yolo_confidence": event.yolo_confidence,

                # --------------------------------------------
                # Eyes
                # --------------------------------------------

                "ear": event.ear,
                "eye_state": event.eye_state,
                "perclos": event.perclos,
                "blink_count": event.blink_count,

                # --------------------------------------------
                # Mouth
                # --------------------------------------------

                "mar": event.mar,
                "mouth_state": event.mouth_state,
                "yawn_count": event.yawn_count,

                # --------------------------------------------
                # Head pose
                # --------------------------------------------

                "head_state": event.head_state,
                "pitch": event.pitch,
                "yaw": event.yaw,

                # --------------------------------------------
                # Alert
                # --------------------------------------------

                "alert_level": event.alert_level,
                "alarm_active": event.alarm_active,

                # --------------------------------------------
                # Reasons
                # --------------------------------------------

                "reasons": event.reasons,
            }

            for event in events
        ]

    finally:

        db.close()