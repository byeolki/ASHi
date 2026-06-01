import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from ..event import COLORS, Event
from .base import Backend


def _build_embed(event: Event) -> dict:
    embed: dict = {
        "title": event.title,
        "color": COLORS.get(event.color, COLORS["blue"]),
        "fields": [{"name": k, "value": str(v), "inline": True} for k, v in event.fields.items()],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if event.description:
        embed["description"] = event.description
    return embed


class DiscordBackend(Backend):
    """Discord Incoming Webhook backend.

    Create a webhook at Server Settings → Integrations → Webhooks.
    """

    def __init__(self, webhook_url: str, username: str = "Train Logger", timeout: int = 10):
        self._url = webhook_url
        self._username = username
        self._timeout = timeout

    def emit(self, event: Event) -> bool:
        payload = {"username": self._username, "embeds": [_build_embed(event)]}
        if event.file_path:
            return self._send_with_file(payload, event.file_path, event.filename)
        return self._send(payload)

    def _send(self, payload: dict) -> bool:
        try:
            resp = requests.post(self._url, json=payload, timeout=self._timeout)
            return resp.status_code in (200, 204)
        except Exception:
            return False

    def _send_with_file(
        self, payload: dict, file_path: str, filename: Optional[str] = None
    ) -> bool:
        path = Path(file_path)
        if not path.exists():
            return False
        name = filename or path.name
        try:
            with open(path, "rb") as f:
                resp = requests.post(
                    self._url,
                    data={"payload_json": json.dumps(payload)},
                    files={"file": (name, f)},
                    timeout=self._timeout,
                )
            return resp.status_code in (200, 204)
        except Exception:
            return False
