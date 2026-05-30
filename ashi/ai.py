from __future__ import annotations

import requests

from .event import Event

_SYSTEM_PROMPT = (
    "You are a machine learning training monitor. "
    "Analyze the training metrics below and summarize in 1-2 sentences "
    "whether training is progressing well. "
    "Be specific: mention trends (converging, diverging, overfitting, plateau, etc.) "
    "and the most relevant metric values. Do not use bullet points."
)


def _call_openai(api_key: str, model: str, messages: list, timeout: int) -> str:
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "max_completion_tokens": 150, "temperature": 0.3},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _build_context(experiment: str, config: dict, log: list[dict]) -> str:
    lines = []
    if experiment:
        lines.append(f"Experiment: {experiment}")
    if config:
        lines.append("Config: " + ", ".join(f"{k}={v}" for k, v in config.items()))

    lines.append("\nTraining log (chronological):")

    # First 3 + last 10 entries to keep context concise
    entries = log[:3] + (log[3:-10] and ["  ... ({} entries omitted) ...".format(len(log) - 13)]) + log[-10:]
    for entry in entries:
        if isinstance(entry, str):
            lines.append(entry)
            continue
        fields_str = ", ".join(f"{k}={v}" for k, v in entry["fields"].items())
        line = f"  {entry['title']}"
        if fields_str:
            line += f" | {fields_str}"
        lines.append(line)

    return "\n".join(lines)


class AISummarizer:
    """Accumulates training events and summarizes them with an OpenAI model.

    Usage::

        from ashi import TrainLogger, DiscordBackend
        from ashi.ai import AISummarizer

        logger = TrainLogger(
            DiscordBackend(webhook_url="..."),
            summarizer=AISummarizer(api_key="sk-..."),
        )

        # ... train ...

        logger.summarize()  # posts a 1-2 line AI summary to all backends
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: int = 30,
    ):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._experiment: str = ""
        self._config: dict = {}
        self._log: list[dict] = []

    def record(self, event: Event) -> None:
        if event.kind == "train_start":
            self._experiment = event.fields.get("Experiment", "")
            self._config = {k: v for k, v in event.fields.items() if k != "Experiment"}
        elif event.kind in ("epoch_end", "step_end", "best_metric", "checkpoint", "train_end", "error"):
            self._log.append({"title": event.title, "fields": dict(event.fields)})

    def summarize(self) -> str:
        """Call OpenAI and return a 1-2 sentence training summary.

        Raises requests.HTTPError if the API call fails.
        """
        if not self._log:
            return "No training history recorded yet."

        context = _build_context(self._experiment, self._config, self._log)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]
        return _call_openai(self._api_key, self._model, messages, self._timeout)
