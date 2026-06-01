from __future__ import annotations

from typing import Any, Optional

from ..event import Event
from .base import Backend


class AimBackend(Backend):
    """Aim experiment tracking backend.

    Tracks metrics and metadata to a local (or remote) Aim repository.
    A new run is created automatically on the first ``on_train_start`` event.

    Args:
        run: existing ``aim.Run`` to reuse.
        experiment: Aim experiment name (groups runs in the UI).
        repo: path to the Aim repository. Defaults to current directory.
        finish_on_end: call ``run.close()`` on ``train_end``.
        **run_kwargs: forwarded to ``aim.Run()``.
    """

    def __init__(
        self,
        run=None,
        experiment: Optional[str] = None,
        repo: Optional[str] = None,
        finish_on_end: bool = True,
        **run_kwargs: Any,
    ):
        self._run = run
        self._experiment = experiment
        self._repo = repo
        self._finish_on_end = finish_on_end
        self._run_kwargs = run_kwargs

    @staticmethod
    def _aim():
        try:
            import aim

            return aim
        except ImportError as exc:
            raise ImportError("aim is required: pip install aim") from exc

    def _init_run(self, experiment_name: str, config: dict):
        aim = self._aim()
        kwargs = dict(self._run_kwargs)
        if self._repo:
            kwargs["repo"] = self._repo
        self._run = aim.Run(experiment=self._experiment or experiment_name, **kwargs)
        for k, v in config.items():
            self._run[k] = v
        return self._run

    def emit(self, event: Event) -> bool:
        try:
            self._aim()
        except ImportError:
            return False

        try:
            kind = event.kind

            if kind == "train_start":
                experiment = event.fields.get("Experiment", "")
                config = {k: v for k, v in event.fields.items() if k != "Experiment"}
                if self._run is None:
                    self._init_run(experiment, config)
                else:
                    for k, v in config.items():
                        self._run[k] = v
                return True

            run = self._run
            if run is None:
                return False

            if kind in ("epoch_end", "step_end"):
                for k, v in event.fields.items():
                    try:
                        run.track(float(v), name=k, step=event.step)
                    except (ValueError, TypeError):
                        pass

            elif kind == "best_metric":
                for k, v in event.fields.items():
                    if k == "Step":
                        continue
                    try:
                        run.track(float(v), name=f"best/{k}", step=event.step)
                    except (ValueError, TypeError):
                        pass

            elif kind == "checkpoint":
                run["checkpoint_step"] = event.step
                run["checkpoint_path"] = event.fields.get("Path", "")

            elif kind == "train_end":
                total_str = event.fields.get("Total Steps")
                if total_str:
                    try:
                        run["total_steps"] = float(total_str)
                    except ValueError:
                        pass
                best_str = event.fields.get("Best Val Loss")
                if best_str:
                    try:
                        run["best_val_loss"] = float(best_str)
                    except ValueError:
                        pass
                if self._finish_on_end:
                    run.close()
                    self._run = None

            elif kind == "error":
                run["error"] = event.description

            return True

        except Exception:
            return False
