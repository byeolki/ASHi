from __future__ import annotations

from typing import Any, Optional

from ..event import Event
from .base import Backend


class NeptuneBackend(Backend):
    """Neptune.ai backend.

    Logs metrics and metadata to a Neptune run using the ``neptune`` client library.
    A new run is initialised automatically on the first ``on_train_start`` event.

    Args:
        run: existing ``neptune.Run`` to reuse.
        project: Neptune project in the form ``"workspace/project"``.
        api_token: Neptune API token (falls back to ``NEPTUNE_API_TOKEN`` env var).
        finish_on_end: call ``run.stop()`` on ``train_end``.
        **init_kwargs: forwarded to ``neptune.init_run()``.
    """

    def __init__(
        self,
        run=None,
        project: Optional[str] = None,
        api_token: Optional[str] = None,
        finish_on_end: bool = True,
        **init_kwargs: Any,
    ):
        self._run = run
        self._project = project
        self._api_token = api_token
        self._finish_on_end = finish_on_end
        self._init_kwargs = init_kwargs

    @staticmethod
    def _neptune():
        try:
            import neptune
            return neptune
        except ImportError as exc:
            raise ImportError("neptune is required: pip install neptune") from exc

    def _init_run(self, experiment: str, config: dict):
        neptune = self._neptune()
        kwargs = dict(self._init_kwargs)
        if self._project:
            kwargs["project"] = self._project
        if self._api_token:
            kwargs["api_token"] = self._api_token
        self._run = neptune.init_run(name=experiment, **kwargs)
        if config:
            self._run["config"] = config
        return self._run

    def emit(self, event: Event) -> bool:
        try:
            self._neptune()
        except ImportError:
            return False

        try:
            kind = event.kind

            if kind == "train_start":
                experiment = event.fields.get("Experiment", "")
                config = {k: v for k, v in event.fields.items() if k != "Experiment"}
                if self._run is None:
                    self._init_run(experiment, config)
                elif config:
                    self._run["config"].update(config)
                return True

            run = self._run
            if run is None:
                return False

            if kind in ("epoch_end", "step_end"):
                for k, v in event.fields.items():
                    try:
                        run[f"metrics/{k}"].append(float(v), step=event.step)
                    except (ValueError, TypeError):
                        pass

            elif kind == "best_metric":
                for k, v in event.fields.items():
                    if k == "Step":
                        continue
                    try:
                        run[f"best/{k}"].append(float(v), step=event.step)
                    except (ValueError, TypeError):
                        pass

            elif kind == "checkpoint":
                run["checkpoint/step"] = event.step
                run["checkpoint/path"] = event.fields.get("Path", "")

            elif kind == "train_end":
                total_str = event.fields.get("Total Steps")
                if total_str:
                    try:
                        run["summary/total_steps"] = float(total_str)
                    except ValueError:
                        pass
                best_str = event.fields.get("Best Val Loss")
                if best_str:
                    try:
                        run["summary/best_val_loss"] = float(best_str)
                    except ValueError:
                        pass
                if self._finish_on_end:
                    run.stop()
                    self._run = None

            elif kind == "error":
                run["error"] = event.description

            elif kind in ("message", "ai_summary"):
                run[event.kind] = event.description or event.title

            return True

        except Exception:
            return False
