from datetime import datetime, timezone

import requests

from ..event import SLACK_COLORS, Event
from .base import Backend


class SlackBackend(Backend):
    """Slack Incoming Webhook backend.

    Create a webhook at https://api.slack.com/messaging/webhooks
    File uploads are not supported via incoming webhooks (requires a bot token).
    """

    def __init__(self, webhook_url: str, timeout: int = 10):
        self._url = webhook_url
        self._timeout = timeout

    def emit(self, event: Event) -> bool:
        attachment: dict = {
            "color": SLACK_COLORS.get(event.color, SLACK_COLORS["blue"]),
            "title": event.title,
            "fields": [
                {"title": k, "value": str(v), "short": True} for k, v in event.fields.items()
            ],
            "footer": "train-logger",
            "ts": int(datetime.now(timezone.utc).timestamp()),
        }
        if event.description:
            attachment["text"] = event.description

        try:
            resp = requests.post(
                self._url,
                json={"attachments": [attachment]},
                timeout=self._timeout,
            )
            return resp.status_code == 200
        except Exception:
            return False
