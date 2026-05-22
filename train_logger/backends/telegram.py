from pathlib import Path

import requests

from ..event import TELEGRAM_EMOJI, Event
from .base import Backend


def _format_message(event: Event) -> str:
    emoji = TELEGRAM_EMOJI.get(event.color, "\U0001f535")
    lines = [f"{emoji} *{event.title}*"]
    if event.description:
        lines.append(event.description)
    for k, v in event.fields.items():
        lines.append(f"• *{k}:* {v}")
    return "\n".join(lines)


class TelegramBackend(Backend):
    """Telegram Bot API backend.

    Create a bot via @BotFather to get a token.
    Get your chat_id by messaging @userinfobot or via the getUpdates API.
    """

    def __init__(self, token: str, chat_id: str, timeout: int = 10):
        self._base = f"https://api.telegram.org/bot{token}"
        self._chat_id = chat_id
        self._timeout = timeout

    def emit(self, event: Event) -> bool:
        if event.file_path:
            return self._send_document(event)
        return self._send_message(event)

    def _send_message(self, event: Event) -> bool:
        try:
            resp = requests.post(
                f"{self._base}/sendMessage",
                json={
                    "chat_id": self._chat_id,
                    "text": _format_message(event),
                    "parse_mode": "Markdown",
                },
                timeout=self._timeout,
            )
            return resp.ok
        except Exception:
            return False

    def _send_document(self, event: Event) -> bool:
        path = Path(event.file_path)
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                resp = requests.post(
                    f"{self._base}/sendDocument",
                    data={
                        "chat_id": self._chat_id,
                        "caption": _format_message(event)[:1024],
                        "parse_mode": "Markdown",
                    },
                    files={"document": (event.filename or path.name, f)},
                    timeout=self._timeout,
                )
            return resp.ok
        except Exception:
            return False
