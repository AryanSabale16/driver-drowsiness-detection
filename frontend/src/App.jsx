import { useEffect, useState } from "react";
import { getStatus } from "./services/api";
import "./App.css";

function App() {
  const [status, setStatus] = useState(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState("");

  const fetchStatus = async () => {
    try {
      const data = await getStatus();

      setStatus(data);
      setConnected(true);
      setError("");
    } catch (err) {
      console.error(err);
      setConnected(false);
      setError("Backend connection unavailable");
    }
  };

  useEffect(() => {
    fetchStatus();

    const interval = setInterval(fetchStatus, 500);

    return () => clearInterval(interval);
  }, []);

  const currentStatus = status?.status || "ALERT";
  const alertLevel = status?.alert_level || "NONE";

  const getStatusClass = (value) => {
    if (value === "CRITICAL") return "critical";
    if (value === "DROWSY") return "drowsy";
    if (value === "CAUTION") return "caution";
    return "normal";
  };

  const score = Number(status?.drowsiness_score || 0);
  const behaviorScore = Number(status?.behavior_score || 0);
  const yoloScore = Number(status?.yolo_score || 0);

  return (
    <div className="app">
      {/* HEADER */}
      <header className="header">
        <div>
          <div className="brand">
            <div className="brand-icon">◉</div>

            <div>
              <h1>Driver Drowsiness Detection</h1>
              <p>Hybrid YOLO + MediaPipe Intelligence System</p>
            </div>
          </div>
        </div>

        <div className="connection">
          <span
            className={`connection-dot ${
              connected ? "connected" : "disconnected"
            }`}
          />

          <span>
            {connected ? "Backend Connected" : "Backend Offline"}
          </span>
        </div>
      </header>

      {/* MAIN */}
      <main className="dashboard">
        {/* SYSTEM OVERVIEW */}
        <section className="hero-grid">
          <div className="score-card">
            <div className="section-label">DROWSINESS SCORE</div>

            <div className="score-value">
              {score.toFixed(1)}
            </div>

            <div className="score-bar">
              <div
                className={`score-fill ${getStatusClass(currentStatus)}`}
                style={{
                  width: `${Math.min(score, 100)}%`,
                }}
              />
            </div>

            <div className="score-scale">
              <span>0</span>
              <span>25</span>
              <span>50</span>
              <span>75</span>
              <span>100</span>
            </div>
          </div>

          <div className={`status-card ${getStatusClass(currentStatus)}`}>
            <div className="section-label">CURRENT STATUS</div>

            <div className="large-status">
              {currentStatus}
            </div>

            <p>
              {currentStatus === "CRITICAL"
                ? "Immediate driver alert required"
                : currentStatus === "DROWSY"
                ? "Drowsiness detected"
                : currentStatus === "CAUTION"
                ? "Early signs of drowsiness"
                : "Driver appears alert"}
            </p>
          </div>

          <div className="alert-card">
            <div className="section-label">ALERT LEVEL</div>

            <div className={`alert-value ${getStatusClass(alertLevel)}`}>
              {alertLevel}
            </div>

            <div className="alarm-row">
              <span>Alarm</span>

              <span
                className={`alarm-status ${
                  status?.alarm_active ? "active" : ""
                }`}
              >
                {status?.alarm_active ? "ACTIVE" : "OFF"}
              </span>
            </div>
          </div>
        </section>

        {/* DETECTION + INTELLIGENCE */}
        <section className="content-grid">
          {/* DETECTION */}
          <div className="panel">
            <div className="panel-header">
              <div>
                <h2>Detection</h2>
                <p>YOLO + MediaPipe observations</p>
              </div>

              <span
                className={`badge ${
                  status?.driver_detected ? "success" : "muted"
                }`}
              >
                {status?.driver_detected
                  ? "DRIVER DETECTED"
                  : "NO DRIVER"}
              </span>
            </div>

            <div className="metrics-grid">
              <Metric
                label="YOLO CLASS"
                value={status?.yolo_class || "—"}
              />

              <Metric
                label="YOLO CONFIDENCE"
                value={`${(
                  Number(status?.yolo_confidence || 0) * 100
                ).toFixed(1)}%`}
              />

              <Metric
                label="FACE"
                value={status?.face_detected ? "DETECTED" : "NOT FOUND"}
              />

              <Metric
                label="EYE STATE"
                value={status?.eye_state || "UNKNOWN"}
              />

              <Metric
                label="EAR"
                value={Number(status?.ear || 0).toFixed(3)}
              />

              <Metric
                label="MOUTH STATE"
                value={status?.mouth_state || "UNKNOWN"}
              />

              <Metric
                label="MAR"
                value={Number(status?.mar || 0).toFixed(3)}
              />

              <Metric
                label="HEAD POSE"
                value={status?.head_state || "UNKNOWN"}
              />
            </div>
          </div>

          {/* INTELLIGENCE */}
          <div className="panel">
            <div className="panel-header">
              <div>
                <h2>Drowsiness Intelligence</h2>
                <p>Hybrid behavioral analysis</p>
              </div>
            </div>

            <div className="intelligence-score">
              <div>
                <span>Hybrid Score</span>
                <strong>{score.toFixed(1)}</strong>
              </div>

              <div className="mini-bar">
                <div
                  style={{
                    width: `${Math.min(score, 100)}%`,
                  }}
                />
              </div>
            </div>

            <div className="comparison">
              <div className="comparison-item">
                <span>Behavior Score</span>
                <strong>{behaviorScore.toFixed(1)}</strong>
              </div>

              <div className="comparison-item">
                <span>YOLO Score</span>
                <strong>{yoloScore.toFixed(1)}</strong>
              </div>
            </div>

            <div className="weights">
              <div>
                <span>Behavior Weight</span>
                <strong>70%</strong>
              </div>

              <div>
                <span>YOLO Weight</span>
                <strong>30%</strong>
              </div>
            </div>
          </div>
        </section>

        {/* TEMPORAL METRICS */}
        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Temporal Analysis</h2>
              <p>Driver behavior over the observation window</p>
            </div>
          </div>

          <div className="temporal-grid">
            <Metric
              label="PERCLOS"
              value={`${Number(status?.perclos || 0).toFixed(1)}%`}
            />

            <Metric
              label="BLINK COUNT"
              value={status?.blink_count ?? 0}
            />

            <Metric
              label="YAWN COUNT"
              value={status?.yawn_count ?? 0}
            />

            <Metric
              label="HEAD PITCH"
              value={`${Number(status?.pitch || 0).toFixed(1)}°`}
            />

            <Metric
              label="HEAD YAW"
              value={`${Number(status?.yaw || 0).toFixed(1)}°`}
            />

            <Metric
              label="SESSION"
              value={
                status?.session_id
                  ? `#${status.session_id}`
                  : "NONE"
              }
            />
          </div>
        </section>

        {/* REASONS */}
        <section className="content-grid">
          <div className="panel">
            <div className="panel-header">
              <div>
                <h2>Detection Reasons</h2>
                <p>Factors contributing to the current state</p>
              </div>
            </div>

            <div className="reasons">
              {status?.reasons?.length > 0 ? (
                status.reasons.map((reason, index) => (
                  <div className="reason" key={index}>
                    <span className="reason-icon">!</span>
                    <span>{reason}</span>
                  </div>
                ))
              ) : (
                <div className="empty-state">
                  No drowsiness factors detected.
                </div>
              )}
            </div>
          </div>

          <div className="panel system-panel">
            <div className="panel-header">
              <div>
                <h2>System Status</h2>
                <p>Current pipeline state</p>
              </div>
            </div>

            <div className="system-list">
              <SystemRow
                label="YOLO Detector"
                value={status?.yolo_class ? "ACTIVE" : "WAITING"}
              />

              <SystemRow
                label="MediaPipe Face Mesh"
                value={status?.face_detected ? "ACTIVE" : "WAITING"}
              />

              <SystemRow
                label="Intelligence Engine"
                value="ACTIVE"
              />

              <SystemRow
                label="Alert Controller"
                value={
                  status?.alarm_active ? "ALARM ACTIVE" : "STANDBY"
                }
              />

              <SystemRow
                label="Session Manager"
                value={
                  status?.session_id
                    ? `SESSION #${status.session_id}`
                    : "NO SESSION"
                }
              />
            </div>
          </div>
        </section>

        {error && (
          <div className="error-banner">
            {error}
          </div>
        )}
      </main>

      <footer>
        Driver Drowsiness Detection System · Hybrid YOLO + MediaPipe
      </footer>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SystemRow({ label, value }) {
  return (
    <div className="system-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;