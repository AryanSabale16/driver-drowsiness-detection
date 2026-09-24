const API_BASE_URL = "http://127.0.0.1:8000";

export async function getStatus() {
  const response = await fetch(`${API_BASE_URL}/status`);

  if (!response.ok) {
    throw new Error("Failed to fetch system status");
  }

  return response.json();
}

export async function getSessions() {
  const response = await fetch(`${API_BASE_URL}/sessions`);

  if (!response.ok) {
    throw new Error("Failed to fetch sessions");
  }

  return response.json();
}

export async function getSessionEvents(sessionId) {
  const response = await fetch(
    `${API_BASE_URL}/sessions/${sessionId}/events`
  );

  if (!response.ok) {
    throw new Error("Failed to fetch session events");
  }

  return response.json();
}