import time
from typing import Optional

import requests



class BackendStatusClient:
    """
    Sends the latest AI pipeline status to the FastAPI backend.

    The backend connection must never stop the AI detection pipeline.
    If FastAPI is unavailable, the AI system continues running normally.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        update_interval: float = 0.2,
    ):
        self.base_url = base_url.rstrip("/")
        self.status_url = f"{self.base_url}/status"

        self.update_interval = update_interval
        self.last_update_time = 0.0

        self.enabled = True

    def update(self, status_data: dict) -> Optional[dict]:
        """
        Send the latest AI status to FastAPI.

        Updates are throttled using update_interval so that
        an HTTP request is not made for every video frame.

        Returns:
            Backend response dictionary if successful.
            None if throttled or backend unavailable.
        """

        if not self.enabled:
            return None

        current_time = time.monotonic()

        # Prevent an HTTP request on every video frame.
        if (
            current_time - self.last_update_time
            < self.update_interval
        ):
            return None

        self.last_update_time = current_time

        try:
            response = requests.post(
                self.status_url,
                json=status_data,
                timeout=0.15,
            )

            response.raise_for_status()

            return response.json()

        except requests.RequestException:
            # Backend failure must never stop the AI pipeline.
            return None

    def stop(self):
        """
        Disable future backend updates.
        """
        self.enabled = False