from __future__ import annotations

from typing import Any, Optional

from ..event import Event
from .base import Backend


def _to_float(v: Any) -> Any:
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


class WandbBackend(Backend):
    """Weights & Biases backend.

    Logs metrics via ``wandb.log()``, updates run config on train start,
    writes summary values, and sends alerts on errors.

    Pass an existing run via ``run=wandb.run`` to reuse it, or let the backend
    call ``wandb.init()`` automatically on the first ``on_train_start`` event.

    Args:
        project: wandb project name used when auto-initialising a run.
        run: existing ``wandb.Run`` — if given, ``project`` and ``init_kwargs`` are ignored.
        finish_on_end: call ``wandb.finish()`` when a ``train_end`` event is received.
        **init_kwargs: forwarded verbatim to ``wandb.init()``.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        run=None,
        finish_on_end: bool = True,
        **init_kwargs: Any,
    ):
        self._project = project
        self._run = run
        self._finish_on_end = finish_on_end
        self._init_kwargs = init_kwargs

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wandb():
        try:
            import wandb

            return wandb
        except ImportError as exc:
            raise ImportError("wandb is required for WandbBackend: pip install wandb") from exc

    def _active_run(self):
        wandb = self._wandb()
        if self._run is not None:
            return self._run
        if wandb.run is not None:
            self._run = wandb.run
        return self._run

    def _init_run(self, experiment: str, config: dict):
        wandb = self._wandb()
        self._run = wandb.init(
            project=self._project or experiment,
            name=experiment,
            config=config or None,
            **self._init_kwargs,
        )
        return self._run

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    def emit(self, event: Event) -> bool:
        try:
            wandb = self._wandb()
        except ImportError:
            return False

        try:
            kind = event.kind

            # ---- train start ----------------------------------------
            if kind == "train_start":
                experiment = event.fields.get("Experiment", "")
                config = {k: v for k, v in event.fields.items() if k != "Experiment"}
                if self._active_run() is None:
                    self._init_run(experiment, config)
                elif config:
                    self._run.config.update(config)
                return True

            run = self._active_run()
            if run is None:
                return False

            # ---- epoch / step metrics --------------------------------
            if kind in ("epoch_end", "step_end"):
                metrics = {k: _to_float(v) for k, v in event.fields.items()}
                wandb.log(metrics, step=event.step)

            # ---- best metric ----------------------------------------
            elif kind == "best_metric":
                metrics = {
                    f"best/{k}": _to_float(v) for k, v in event.fields.items() if k != "Step"
                }
                wandb.log(metrics, step=event.step)
                run.summary.update(metrics)

            # ---- checkpoint -----------------------------------------
            elif kind == "checkpoint":
                run.summary["last_checkpoint_step"] = event.step
                run.summary["last_checkpoint_path"] = event.fields.get("Path", "")

            # ---- train end ------------------------------------------
            elif kind == "train_end":
                total_str = event.fields.get("Total Steps")
                if total_str is not None:
                    run.summary["total_steps"] = _to_float(total_str)
                best_str = event.fields.get("Best Val Loss")
                if best_str is not None:
                    run.summary["best_val_loss"] = _to_float(best_str)
                if self._finish_on_end:
                    wandb.finish()
                    self._run = None

            # ---- error ----------------------------------------------
            elif kind == "error":
                wandb.alert(
                    title=event.title,
                    text=event.description,
                    level=wandb.AlertLevel.ERROR,
                )

            # ---- custom message / ai summary ------------------------
            elif kind in ("message", "ai_summary"):
                wandb.alert(
                    title=event.title,
                    text=event.description or event.title,
                    level=wandb.AlertLevel.INFO,
                )

            return True

        except Exception:
            return False
