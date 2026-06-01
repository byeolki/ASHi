from __future__ import annotations

from typing import Any, Optional

from ..event import Event
from .base import Backend


class CometBackend(Backend):
    """Comet ML backend.

    Logs metrics, parameters, and metadata to a Comet ML experiment.
    A new experiment is created automatically on the first ``on_train_start`` event.

    Args:
        experiment: existing ``comet_ml.Experiment`` to reuse.
        api_key: Comet API key (falls back to ``COMET_API_KEY`` env var).
        project_name: Comet project name.
        workspace: Comet workspace name.
        finish_on_end: call ``experiment.end()`` on ``train_end``.
        **experiment_kwargs: forwarded to ``comet_ml.Experiment()``.
    """

    def __init__(
        self,
        experiment=None,
        api_key: Optional[str] = None,
        project_name: Optional[str] = None,
        workspace: Optional[str] = None,
        finish_on_end: bool = True,
        **experiment_kwargs: Any,
    ):
        self._experiment = experiment
        self._api_key = api_key
        self._project_name = project_name
        self._workspace = workspace
        self._finish_on_end = finish_on_end
        self._experiment_kwargs = experiment_kwargs

    @staticmethod
    def _comet_ml():
        try:
            import comet_ml

            return comet_ml
        except ImportError as exc:
            raise ImportError("comet-ml is required: pip install comet-ml") from exc

    def _init_experiment(self, experiment_name: str, config: dict):
        comet_ml = self._comet_ml()
        kwargs = dict(self._experiment_kwargs)
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._project_name:
            kwargs["project_name"] = self._project_name
        if self._workspace:
            kwargs["workspace"] = self._workspace
        self._experiment = comet_ml.Experiment(**kwargs)
        self._experiment.set_name(experiment_name)
        if config:
            self._experiment.log_parameters(config)
        return self._experiment

    def emit(self, event: Event) -> bool:
        try:
            self._comet_ml()
        except ImportError:
            return False

        try:
            kind = event.kind

            if kind == "train_start":
                experiment = event.fields.get("Experiment", "")
                config = {k: v for k, v in event.fields.items() if k != "Experiment"}
                if self._experiment is None:
                    self._init_experiment(experiment, config)
                elif config:
                    self._experiment.log_parameters(config)
                return True

            exp = self._experiment
            if exp is None:
                return False

            if kind in ("epoch_end", "step_end"):
                metrics = {}
                for k, v in event.fields.items():
                    try:
                        metrics[k] = float(v)
                    except (ValueError, TypeError):
                        pass
                if metrics:
                    exp.log_metrics(metrics, step=event.step)

            elif kind == "best_metric":
                for k, v in event.fields.items():
                    if k == "Step":
                        continue
                    try:
                        exp.log_metric(f"best_{k}", float(v), step=event.step)
                    except (ValueError, TypeError):
                        pass

            elif kind == "checkpoint":
                path = event.fields.get("Path", "")
                if path:
                    try:
                        exp.log_model(name=f"checkpoint-{event.step}", file_or_folder=path)
                    except Exception:
                        pass

            elif kind == "train_end":
                total_str = event.fields.get("Total Steps")
                if total_str:
                    try:
                        exp.log_other("total_steps", float(total_str))
                    except ValueError:
                        pass
                best_str = event.fields.get("Best Val Loss")
                if best_str:
                    try:
                        exp.log_metric("best_val_loss", float(best_str))
                    except ValueError:
                        pass
                if self._finish_on_end:
                    exp.end()
                    self._experiment = None

            elif kind == "error":
                exp.log_other("error", event.description)
                exp.add_tag("error")

            elif kind in ("message", "ai_summary"):
                exp.log_other(event.kind, event.description or event.title)

            return True

        except Exception:
            return False
