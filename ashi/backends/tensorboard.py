from __future__ import annotations

from typing import Any, Optional

from ..event import Event
from .base import Backend


class TensorBoardBackend(Backend):
    """TensorBoard SummaryWriter backend.

    Writes scalars and text to TensorBoard event files.
    Requires ``torch`` (for ``torch.utils.tensorboard``) or ``tensorboardX``.

    Args:
        log_dir: directory for TensorBoard event files. Defaults to ``"runs"``.
        writer: existing ``SummaryWriter`` instance to reuse.
        close_on_end: call ``writer.close()`` on ``train_end``.
        **writer_kwargs: forwarded to ``SummaryWriter()``.
    """

    def __init__(
        self,
        log_dir: str = "runs",
        writer=None,
        close_on_end: bool = True,
        **writer_kwargs: Any,
    ):
        self._log_dir = log_dir
        self._writer = writer
        self._close_on_end = close_on_end
        self._writer_kwargs = writer_kwargs

    @staticmethod
    def _writer_class():
        try:
            from torch.utils.tensorboard import SummaryWriter
            return SummaryWriter
        except ImportError:
            pass
        try:
            from tensorboardX import SummaryWriter
            return SummaryWriter
        except ImportError:
            pass
        raise ImportError(
            "TensorBoard requires torch or tensorboardX: "
            "pip install torch  (or pip install tensorboardX)"
        )

    def _get_writer(self):
        if self._writer is None:
            cls = self._writer_class()
            self._writer = cls(log_dir=self._log_dir, **self._writer_kwargs)
        return self._writer

    def emit(self, event: Event) -> bool:
        try:
            writer = self._get_writer()
        except ImportError:
            return False

        try:
            kind = event.kind

            if kind == "train_start":
                config = {k: v for k, v in event.fields.items() if k != "Experiment"}
                if config:
                    hparam_str = "  \n".join(f"**{k}**: {v}" for k, v in config.items())
                    writer.add_text("hparams", hparam_str)
                return True

            if kind in ("epoch_end", "step_end"):
                for k, v in event.fields.items():
                    try:
                        writer.add_scalar(k, float(v), global_step=event.step)
                    except (ValueError, TypeError):
                        pass

            elif kind == "best_metric":
                for k, v in event.fields.items():
                    if k == "Step":
                        continue
                    try:
                        writer.add_scalar(f"best/{k}", float(v), global_step=event.step)
                    except (ValueError, TypeError):
                        pass

            elif kind == "train_end":
                total_str = event.fields.get("Total Steps")
                if total_str:
                    try:
                        writer.add_scalar("total_steps", float(total_str))
                    except ValueError:
                        pass
                if self._close_on_end:
                    writer.close()
                    self._writer = None
                return True

            elif kind == "error":
                writer.add_text("error", event.description)

            elif kind in ("message", "ai_summary"):
                writer.add_text(event.kind, event.description or event.title)

            writer.flush()
            return True

        except Exception:
            return False
