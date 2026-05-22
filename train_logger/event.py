from dataclasses import dataclass, field
from typing import Optional

COLORS: dict[str, int] = {
    "green": 0x57F287,
    "red": 0xED4245,
    "yellow": 0xFEE75C,
    "blue": 0x5865F2,
    "gray": 0x95A5A6,
    "orange": 0xE67E22,
}

ANSI_COLORS: dict[str, str] = {
    "green": "\033[92m",
    "red": "\033[91m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "gray": "\033[90m",
    "orange": "\033[33m",
    "reset": "\033[0m",
}

SLACK_COLORS: dict[str, str] = {
    "green": "#57F287",
    "red": "#ED4245",
    "yellow": "#FEE75C",
    "blue": "#5865F2",
    "gray": "#95A5A6",
    "orange": "#E67E22",
}

TELEGRAM_EMOJI: dict[str, str] = {
    "green": "✅",
    "red": "❌",
    "yellow": "⭐",
    "blue": "\U0001f535",
    "gray": "⚪",
    "orange": "\U0001f7e0",
}


def make_progress_bar(current: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return ""
    ratio = min(current / total, 1.0)
    filled = int(width * ratio)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {ratio * 100:.1f}%"


def format_metrics(metrics: dict) -> dict:
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.4f}"
        else:
            formatted[key] = str(value)
    return formatted


@dataclass
class Event:
    kind: str
    title: str
    fields: dict = field(default_factory=dict)
    description: str = ""
    color: str = "blue"
    file_path: Optional[str] = None
    filename: Optional[str] = None
    step: Optional[int] = None
