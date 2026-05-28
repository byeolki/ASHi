"""Framework-specific callbacks for HuggingFace Transformers and PyTorch Lightning.

These classes implement the callback interface via duck-typing — no framework dependency
is required to import them.

HuggingFace example::

    from ashi import TrainLogger, DiscordBackend, HuggingFaceCallback
    logger = TrainLogger(DiscordBackend(webhook_url="..."))
    trainer = Trainer(..., callbacks=[HuggingFaceCallback(logger, experiment="my-run")])

PyTorch Lightning example::

    from ashi import TrainLogger, DiscordBackend, LightningCallback
    logger = TrainLogger(DiscordBackend(webhook_url="..."))
    trainer = pl.Trainer(..., callbacks=[LightningCallback(logger, experiment="my-run")])
"""
from .logger import TrainLogger


class HuggingFaceCallback:
    """transformers.TrainerCallback compatible callback (duck-typed)."""

    def __init__(self, logger: TrainLogger, experiment: str = ""):
        self._logger = logger
        self._experiment = experiment

    def on_train_begin(self, args, state, control, **kwargs):
        config = {
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "num_train_epochs": int(args.num_train_epochs),
            "output_dir": args.output_dir,
        }
        self._logger.on_train_start(
            experiment=self._experiment or args.output_dir,
            config=config,
        )
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        if "eval_loss" in logs or "loss" in logs:
            metrics = {k: v for k, v in logs.items() if k != "epoch"}
            epoch = int(state.epoch) if state.epoch else 0
            self._logger.on_epoch_end(
                epoch=epoch,
                metrics=metrics,
                total_epochs=int(args.num_train_epochs),
                step=state.global_step,
            )
        return control

    def on_save(self, args, state, control, **kwargs):
        self._logger.on_checkpoint_saved(step=state.global_step, path=args.output_dir)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        best_metric = None
        if hasattr(state, "best_metric") and state.best_metric is not None:
            best_metric = float(state.best_metric)
        self._logger.on_train_end(total_steps=state.global_step, best_val_loss=best_metric)
        return control


class LightningCallback:
    """pytorch_lightning.Callback compatible callback (duck-typed)."""

    def __init__(self, logger: TrainLogger, experiment: str = ""):
        self._logger = logger
        self._experiment = experiment

    def on_train_start(self, trainer, pl_module):
        self._logger.on_train_start(
            experiment=self._experiment or type(pl_module).__name__,
            config={"max_epochs": trainer.max_epochs},
        )

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {}
        for k, v in trainer.callback_metrics.items():
            try:
                metrics[k] = float(v)
            except (TypeError, ValueError):
                metrics[k] = str(v)
        self._logger.on_epoch_end(
            epoch=trainer.current_epoch,
            metrics=metrics,
            total_epochs=trainer.max_epochs,
            step=trainer.global_step,
        )

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        path = (ckpt_cb.dirpath or "") if ckpt_cb and hasattr(ckpt_cb, "dirpath") else ""
        self._logger.on_checkpoint_saved(step=trainer.global_step, path=path)

    def on_train_end(self, trainer, pl_module):
        self._logger.on_train_end(total_steps=trainer.global_step)

    def on_exception(self, trainer, pl_module, exception):
        self._logger.on_error(exception, context="Lightning training")
