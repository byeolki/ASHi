from __future__ import annotations

from typing import Any, Optional

from ..event import Event
from .base import Backend


class MLflowBackend(Backend):
    """MLflow Tracking backend.

    Logs metrics, params, and tags to an MLflow tracking server.
    A new run is started automatically on the first ``on_train_start`` event,
    or you can point to an existing run via ``run_id``.

    Args:
        experiment_name: MLflow experiment name (created if it doesn't exist).
        tracking_uri: MLflow tracking server URI. Defaults to mlflow's default (./mlruns).
        run_id: existing run ID to resume. If given, a new run is not started.
        finish_on_end: call ``mlflow.end_run()`` when a ``train_end`` event is received.
        **run_kwargs: forwarded to ``mlflow.start_run()``.
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        run_id: Optional[str] = None,
        finish_on_end: bool = True,
        **run_kwargs: Any,
    ):
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self._run_id = run_id
        self._finish_on_end = finish_on_end
        self._run_kwargs = run_kwargs
        self._started = False

    @staticmethod
    def _mlflow():
        try:
            import mlflow
            return mlflow
        except ImportError as exc:
            raise ImportError("mlflow is required: pip install mlflow") from exc

    def _ensure_run(self, run_name: str = "", config: Optional[dict] = None):
        mlflow = self._mlflow()
        if mlflow.active_run() is not None:
            self._started = True
            return
        if self._tracking_uri:
            mlflow.set_tracking_uri(self._tracking_uri)
        if self._experiment_name:
            mlflow.set_experiment(self._experiment_name)
        mlflow.start_run(run_id=self._run_id, run_name=run_name or None, **self._run_kwargs)
        self._started = True
        if config:
            mlflow.log_params({k: str(v) for k, v in config.items()})

    def emit(self, event: Event) -> bool:
        try:
            mlflow = self._mlflow()
        except ImportError:
            return False

        try:
            kind = event.kind

            if kind == "train_start":
                experiment = event.fields.get("Experiment", "")
                config = {k: v for k, v in event.fields.items() if k != "Experiment"}
                self._ensure_run(run_name=experiment, config=config)
                return True

            if not self._started:
                self._ensure_run()

            if kind in ("epoch_end", "step_end"):
                metrics = {}
                for k, v in event.fields.items():
                    try:
                        metrics[k] = float(v)
                    except (ValueError, TypeError):
                        pass
                if metrics:
                    mlflow.log_metrics(metrics, step=event.step)

            elif kind == "best_metric":
                metrics = {}
                for k, v in event.fields.items():
                    if k == "Step":
                        continue
                    try:
                        metrics[f"best.{k}"] = float(v)
                    except (ValueError, TypeError):
                        pass
                if metrics:
                    mlflow.log_metrics(metrics, step=event.step)

            elif kind == "checkpoint":
                path = event.fields.get("Path", "")
                if path:
                    try:
                        mlflow.log_artifact(path)
                    except Exception:
                        pass
                if event.step is not None:
                    mlflow.log_metric("checkpoint_step", event.step)

            elif kind == "train_end":
                final: dict = {}
                total_str = event.fields.get("Total Steps")
                if total_str:
                    try:
                        final["total_steps"] = float(total_str)
                    except ValueError:
                        pass
                best_str = event.fields.get("Best Val Loss")
                if best_str:
                    try:
                        final["best_val_loss"] = float(best_str)
                    except ValueError:
                        pass
                if final:
                    mlflow.log_metrics(final)
                if self._finish_on_end:
                    mlflow.end_run()
                    self._started = False

            elif kind == "error":
                mlflow.set_tag("error", event.description[:500])

            elif kind in ("message", "ai_summary"):
                mlflow.set_tag(event.kind, (event.description or event.title)[:500])

            return True

        except Exception:
            return False
