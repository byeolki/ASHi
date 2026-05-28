from datetime import datetime, timezone

from ..event import ANSI_COLORS, Event
from .base import Backend


class ConsoleBackend(Backend):
    """Stdout backend — useful for local development and testing without a webhook."""

    def __init__(self, use_color: bool = True):
        self._use_color = use_color

    def emit(self, event: Event) -> bool:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        c = ANSI_COLORS.get(event.color, "") if self._use_color else ""
        reset = ANSI_COLORS["reset"] if self._use_color else ""

        lines = [f"{c}[{ts}] {event.title}{reset}"]
        if event.description:
            for line in event.description.splitlines():
                lines.append(f"  {line}")
        for k, v in event.fields.items():
            lines.append(f"  {k}: {v}")
        if event.file_path:
            lines.append(f"  file: {event.file_path}")
        print("\n".join(lines))
        return True
