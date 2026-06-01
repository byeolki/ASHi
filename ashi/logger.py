from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .backends.base import Backend
from .event import Event, format_metrics, make_progress_bar

if TYPE_CHECKING:
    from .ai import AISummarizer


class _ErrorCatcher:
    def __init__(self, logger: "TrainLogger", context: str):
        self._logger = logger
        self._context = context

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self._logger.on_error(exc_val, context=self._context)
        return False


class TrainLogger:
    def __init__(self, *backends: Backend, summarizer: Optional["AISummarizer"] = None):
        self._backends = list(backends)
        self._summarizer = summarizer

    def add_backend(self, backend: Backend) -> "TrainLogger":
        self._backends.append(backend)
        return self

    def _emit(self, event: Event) -> bool:
        if self._summarizer:
            self._summarizer.record(event)
        if not self._backends:
            return False
        return all(b.emit(event) for b in self._backends)

    def on_train_start(self, experiment: str, config: Optional[dict] = None) -> bool:
        fields = {"Experiment": experiment}
        if config:
            fields.update(config)
        return self._emit(
            Event(kind="train_start", title="Training Started", fields=fields, color="green")
        )

    def on_epoch_end(
        self,
        epoch: int,
        metrics: dict,
        total_epochs: Optional[int] = None,
        step: Optional[int] = None,
    ) -> bool:
        title = f"Epoch {epoch}" + (f" / {total_epochs}" if total_epochs else "")
        parts = []
        if total_epochs:
            parts.append(make_progress_bar(epoch, total_epochs))
        if step is not None:
            parts.append(f"Step: {step}")
        return self._emit(
            Event(
                kind="epoch_end",
                title=title,
                fields=format_metrics(metrics),
                description="\n".join(parts),
                color="blue",
                step=step,
            )
        )

    def on_step_end(
        self,
        step: int,
        metrics: dict,
        total_steps: Optional[int] = None,
    ) -> bool:
        title = f"Step {step}" + (f" / {total_steps}" if total_steps else "")
        description = make_progress_bar(step, total_steps) if total_steps else ""
        return self._emit(
            Event(
                kind="step_end",
                title=title,
                fields=format_metrics(metrics),
                description=description,
                color="blue",
                step=step,
            )
        )

    def on_best_metric(
        self,
        metric_name: str,
        value: float,
        step: Optional[int] = None,
    ) -> bool:
        fields: dict = {metric_name: f"{value:.4f}"}
        if step is not None:
            fields["Step"] = str(step)
        return self._emit(
            Event(
                kind="best_metric",
                title=f"New Best: {metric_name}",
                fields=fields,
                color="yellow",
                step=step,
            )
        )

    def on_checkpoint_saved(self, step: int, path: str) -> bool:
        return self._emit(
            Event(
                kind="checkpoint",
                title="Checkpoint Saved",
                fields={"Step": str(step), "Path": path},
                color="gray",
                step=step,
            )
        )

    def on_train_end(self, total_steps: int, best_val_loss: Optional[float] = None) -> bool:
        fields: dict = {"Total Steps": str(total_steps)}
        if best_val_loss is not None:
            fields["Best Val Loss"] = f"{best_val_loss:.4f}"
        return self._emit(
            Event(kind="train_end", title="Training Complete", fields=fields, color="green")
        )

    def on_error(self, error: Exception, context: str = "") -> bool:
        description = f"{type(error).__name__}: {error}"
        if context:
            description = f"Context: {context}\n{description}"
        return self._emit(
            Event(kind="error", title="Training Error", description=description, color="red")
        )

    def send(self, message: str, color: str = "blue") -> bool:
        return self._emit(Event(kind="message", title=message, color=color))

    def send_file(self, file_path: str, message: str = "", filename: Optional[str] = None) -> bool:
        if not Path(file_path).exists():
            return False
        return self._emit(
            Event(kind="file", title=message, color="blue", file_path=file_path, filename=filename)
        )

    def catch_errors(self, context: str = "") -> _ErrorCatcher:
        return _ErrorCatcher(self, context)

    def summarize(self) -> str:
        """Call the AI summarizer and post the result to all backends.

        Returns the summary string.
        Raises RuntimeError if no summarizer is configured.
        Raises requests.HTTPError if the OpenAI API call fails.
        """
        if self._summarizer is None:
            raise RuntimeError(
                "No summarizer configured. Pass summarizer=AISummarizer(api_key='sk-...') to TrainLogger."
            )
        summary = self._summarizer.summarize()
        self._emit(
            Event(kind="ai_summary", title="AI Summary", description=summary, color="orange")
        )
        return summary
