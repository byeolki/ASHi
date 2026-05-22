import sys
from unittest.mock import MagicMock, patch

import pytest

from train_logger.backends.aim import AimBackend
from train_logger.backends.comet import CometBackend
from train_logger.backends.console import ConsoleBackend
from train_logger.backends.discord import DiscordBackend
from train_logger.backends.mlflow import MLflowBackend
from train_logger.backends.neptune import NeptuneBackend
from train_logger.backends.slack import SlackBackend
from train_logger.backends.telegram import TelegramBackend
from train_logger.backends.tensorboard import TensorBoardBackend
from train_logger.backends.wandb import WandbBackend
from train_logger.event import Event


def _event(**kwargs) -> Event:
    kwargs.setdefault("kind", "test")
    kwargs.setdefault("title", "Test")
    return Event(**kwargs)


def _mock_resp(status: int = 204, ok: bool = True) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.ok = ok
    return r


# ── Discord ──────────────────────────────────────────────────────────────────

class TestDiscordBackend:
    def test_emit_success(self):
        with patch("train_logger.backends.discord.requests.post", return_value=_mock_resp(204)):
            assert DiscordBackend("https://example.com").emit(_event()) is True

    def test_emit_http_error(self):
        with patch("train_logger.backends.discord.requests.post", return_value=_mock_resp(400)):
            assert DiscordBackend("https://example.com").emit(_event()) is False

    def test_emit_network_error(self):
        with patch("train_logger.backends.discord.requests.post", side_effect=Exception("timeout")):
            assert DiscordBackend("https://example.com").emit(_event()) is False

    def test_emit_with_fields_and_description(self):
        with patch("train_logger.backends.discord.requests.post", return_value=_mock_resp(200)) as mock:
            DiscordBackend("https://example.com").emit(
                _event(fields={"loss": "0.42"}, description="progress bar")
            )
            payload = mock.call_args.kwargs["json"]
            embed = payload["embeds"][0]
            assert embed["fields"][0]["name"] == "loss"
            assert embed["description"] == "progress bar"

    def test_emit_file_missing(self):
        assert DiscordBackend("https://example.com").emit(
            _event(file_path="/nonexistent/file.pt")
        ) is False


# ── Slack ─────────────────────────────────────────────────────────────────────

class TestSlackBackend:
    def test_emit_success(self):
        with patch("train_logger.backends.slack.requests.post", return_value=_mock_resp(200)):
            assert SlackBackend("https://hooks.slack.com/...").emit(_event()) is True

    def test_emit_http_error(self):
        with patch("train_logger.backends.slack.requests.post", return_value=_mock_resp(400)):
            assert SlackBackend("https://hooks.slack.com/...").emit(_event()) is False

    def test_emit_network_error(self):
        with patch("train_logger.backends.slack.requests.post", side_effect=Exception):
            assert SlackBackend("https://hooks.slack.com/...").emit(_event()) is False

    def test_payload_shape(self):
        with patch("train_logger.backends.slack.requests.post", return_value=_mock_resp(200)) as mock:
            SlackBackend("https://hooks.slack.com/...").emit(
                _event(fields={"loss": "0.5"}, description="desc", color="red")
            )
            payload = mock.call_args.kwargs["json"]
            att = payload["attachments"][0]
            assert att["color"] == "#ED4245"
            assert att["title"] == "Test"
            assert att["text"] == "desc"
            assert att["fields"][0]["title"] == "loss"


# ── Telegram ──────────────────────────────────────────────────────────────────

class TestTelegramBackend:
    def test_emit_success(self):
        with patch("train_logger.backends.telegram.requests.post", return_value=_mock_resp(ok=True)):
            assert TelegramBackend("TOKEN", "CHAT_ID").emit(_event()) is True

    def test_emit_failure(self):
        with patch("train_logger.backends.telegram.requests.post", return_value=_mock_resp(ok=False)):
            assert TelegramBackend("TOKEN", "CHAT_ID").emit(_event()) is False

    def test_emit_network_error(self):
        with patch("train_logger.backends.telegram.requests.post", side_effect=Exception):
            assert TelegramBackend("TOKEN", "CHAT_ID").emit(_event()) is False

    def test_emit_file_missing(self):
        assert TelegramBackend("TOKEN", "CHAT_ID").emit(
            _event(file_path="/nonexistent/file.pt")
        ) is False

    def test_message_format(self):
        with patch("train_logger.backends.telegram.requests.post", return_value=_mock_resp(ok=True)) as mock:
            TelegramBackend("TOKEN", "CHAT_ID").emit(
                _event(fields={"loss": "0.42"}, color="green")
            )
            body = mock.call_args.kwargs["json"]
            assert "Test" in body["text"]
            assert "loss" in body["text"]


# ── Wandb ─────────────────────────────────────────────────────────────────────

class TestWandbBackend:
    def _mock_wandb(self):
        wandb = MagicMock()
        wandb.run = None
        wandb.AlertLevel.ERROR = "ERROR"
        wandb.AlertLevel.INFO = "INFO"
        return wandb

    def test_train_start_inits_run(self):
        wandb = self._mock_wandb()
        with patch.dict(sys.modules, {"wandb": wandb}):
            backend = WandbBackend(project="test-project")
            result = backend.emit(_event(kind="train_start", title="Training Started",
                                         fields={"Experiment": "my-exp", "lr": "3e-4"}))
        assert result is True
        wandb.init.assert_called_once()
        call_kwargs = wandb.init.call_args.kwargs
        assert call_kwargs["project"] == "test-project"
        assert call_kwargs["name"] == "my-exp"

    def test_train_start_reuses_existing_run(self):
        wandb = self._mock_wandb()
        existing_run = MagicMock()
        wandb.run = existing_run
        with patch.dict(sys.modules, {"wandb": wandb}):
            backend = WandbBackend()
            backend.emit(_event(kind="train_start", title="Training Started",
                                fields={"Experiment": "exp", "lr": "1e-3"}))
        wandb.init.assert_not_called()
        existing_run.config.update.assert_called_once_with({"lr": "1e-3"})

    def test_epoch_end_logs_metrics(self):
        wandb = self._mock_wandb()
        run = MagicMock()
        with patch.dict(sys.modules, {"wandb": wandb}):
            backend = WandbBackend(run=run)
            backend.emit(_event(kind="epoch_end", title="Epoch 1 / 10",
                                fields={"loss": "0.4200", "val_loss": "0.3800"}, step=500))
        wandb.log.assert_called_once_with({"loss": 0.42, "val_loss": 0.38}, step=500)

    def test_best_metric_logs_and_updates_summary(self):
        wandb = self._mock_wandb()
        run = MagicMock()
        with patch.dict(sys.modules, {"wandb": wandb}):
            backend = WandbBackend(run=run)
            backend.emit(_event(kind="best_metric", title="New Best: val_loss",
                                fields={"val_loss": "0.3100", "Step": "1000"}, step=1000))
        wandb.log.assert_called_once_with({"best/val_loss": 0.31}, step=1000)
        run.summary.update.assert_called_once_with({"best/val_loss": 0.31})

    def test_train_end_updates_summary_and_finishes(self):
        wandb = self._mock_wandb()
        run = MagicMock()
        run.summary = {}
        with patch.dict(sys.modules, {"wandb": wandb}):
            backend = WandbBackend(run=run)
            backend.emit(_event(kind="train_end", title="Training Complete",
                                fields={"Total Steps": "50000", "Best Val Loss": "0.3100"}))
        assert run.summary["total_steps"] == 50000.0
        assert run.summary["best_val_loss"] == 0.31
        wandb.finish.assert_called_once()

    def test_train_end_no_finish_when_disabled(self):
        wandb = self._mock_wandb()
        run = MagicMock()
        with patch.dict(sys.modules, {"wandb": wandb}):
            backend = WandbBackend(run=run, finish_on_end=False)
            backend.emit(_event(kind="train_end", title="Training Complete",
                                fields={"Total Steps": "100"}))
        wandb.finish.assert_not_called()

    def test_error_sends_alert(self):
        wandb = self._mock_wandb()
        run = MagicMock()
        with patch.dict(sys.modules, {"wandb": wandb}):
            backend = WandbBackend(run=run)
            backend.emit(_event(kind="error", title="Training Error", description="OOM"))
        wandb.alert.assert_called_once_with(title="Training Error", text="OOM", level="ERROR")

    def test_no_wandb_installed_returns_false(self):
        with patch.dict(sys.modules, {"wandb": None}):
            backend = WandbBackend(project="test")
            result = backend.emit(_event(kind="train_start", fields={"Experiment": "e"}))
        assert result is False

    def test_no_active_run_skips_non_start_events(self):
        wandb = self._mock_wandb()
        with patch.dict(sys.modules, {"wandb": wandb}):
            backend = WandbBackend()
            result = backend.emit(_event(kind="epoch_end", fields={"loss": "0.5"}))
        assert result is False
        wandb.log.assert_not_called()


# ── MLflow ────────────────────────────────────────────────────────────────────

class TestMLflowBackend:
    def _mock_mlflow(self, active_run=None):
        mlflow = MagicMock()
        mlflow.active_run.return_value = active_run
        return mlflow

    def test_train_start_starts_run(self):
        mlflow = self._mock_mlflow()
        with patch.dict(sys.modules, {"mlflow": mlflow}):
            backend = MLflowBackend(experiment_name="my-exp")
            result = backend.emit(_event(kind="train_start", fields={"Experiment": "run-1", "lr": "3e-4"}))
        assert result is True
        mlflow.start_run.assert_called_once_with(run_id=None, run_name="run-1")
        mlflow.log_params.assert_called_once_with({"lr": "3e-4"})

    def test_train_start_reuses_active_run(self):
        mlflow = self._mock_mlflow(active_run=MagicMock())
        with patch.dict(sys.modules, {"mlflow": mlflow}):
            backend = MLflowBackend()
            backend.emit(_event(kind="train_start", fields={"Experiment": "run-1"}))
        mlflow.start_run.assert_not_called()

    def test_epoch_end_logs_metrics(self):
        mlflow = self._mock_mlflow(active_run=MagicMock())
        with patch.dict(sys.modules, {"mlflow": mlflow}):
            backend = MLflowBackend()
            backend._started = True
            backend.emit(_event(kind="epoch_end", fields={"loss": "0.42", "val_loss": "0.38"}, step=100))
        mlflow.log_metrics.assert_called_once_with({"loss": 0.42, "val_loss": 0.38}, step=100)

    def test_train_end_logs_summary_and_ends(self):
        mlflow = self._mock_mlflow(active_run=MagicMock())
        with patch.dict(sys.modules, {"mlflow": mlflow}):
            backend = MLflowBackend()
            backend._started = True
            backend.emit(_event(kind="train_end", fields={"Total Steps": "50000", "Best Val Loss": "0.31"}))
        mlflow.log_metrics.assert_called_once_with({"total_steps": 50000.0, "best_val_loss": 0.31})
        mlflow.end_run.assert_called_once()

    def test_error_sets_tag(self):
        mlflow = self._mock_mlflow(active_run=MagicMock())
        with patch.dict(sys.modules, {"mlflow": mlflow}):
            backend = MLflowBackend()
            backend._started = True
            backend.emit(_event(kind="error", description="OOM error"))
        mlflow.set_tag.assert_called_once_with("error", "OOM error")

    def test_no_mlflow_installed_returns_false(self):
        with patch.dict(sys.modules, {"mlflow": None}):
            backend = MLflowBackend()
            assert backend.emit(_event(kind="train_start", fields={"Experiment": "e"})) is False


# ── Comet ML ──────────────────────────────────────────────────────────────────

class TestCometBackend:
    def _mock_comet_ml(self):
        comet_ml = MagicMock()
        return comet_ml

    def test_train_start_creates_experiment(self):
        comet_ml = self._mock_comet_ml()
        with patch.dict(sys.modules, {"comet_ml": comet_ml}):
            backend = CometBackend(project_name="my-project")
            result = backend.emit(_event(kind="train_start", fields={"Experiment": "run-1", "lr": "1e-3"}))
        assert result is True
        comet_ml.Experiment.assert_called_once()
        comet_ml.Experiment.return_value.set_name.assert_called_once_with("run-1")
        comet_ml.Experiment.return_value.log_parameters.assert_called_once_with({"lr": "1e-3"})

    def test_train_start_reuses_existing_experiment(self):
        comet_ml = self._mock_comet_ml()
        existing = MagicMock()
        with patch.dict(sys.modules, {"comet_ml": comet_ml}):
            backend = CometBackend(experiment=existing)
            backend.emit(_event(kind="train_start", fields={"Experiment": "run-1", "lr": "1e-3"}))
        comet_ml.Experiment.assert_not_called()
        existing.log_parameters.assert_called_once_with({"lr": "1e-3"})

    def test_epoch_end_logs_metrics(self):
        comet_ml = self._mock_comet_ml()
        exp = MagicMock()
        with patch.dict(sys.modules, {"comet_ml": comet_ml}):
            backend = CometBackend(experiment=exp)
            backend.emit(_event(kind="epoch_end", fields={"loss": "0.42"}, step=50))
        exp.log_metrics.assert_called_once_with({"loss": 0.42}, step=50)

    def test_train_end_ends_experiment(self):
        comet_ml = self._mock_comet_ml()
        exp = MagicMock()
        with patch.dict(sys.modules, {"comet_ml": comet_ml}):
            backend = CometBackend(experiment=exp)
            backend.emit(_event(kind="train_end", fields={"Total Steps": "1000"}))
        exp.end.assert_called_once()

    def test_no_comet_installed_returns_false(self):
        with patch.dict(sys.modules, {"comet_ml": None}):
            backend = CometBackend()
            assert backend.emit(_event(kind="train_start", fields={"Experiment": "e"})) is False


# ── Neptune ───────────────────────────────────────────────────────────────────

class TestNeptuneBackend:
    def _mock_neptune(self):
        neptune = MagicMock()
        return neptune

    def test_train_start_inits_run(self):
        neptune = self._mock_neptune()
        with patch.dict(sys.modules, {"neptune": neptune}):
            backend = NeptuneBackend(project="org/project")
            result = backend.emit(_event(kind="train_start", fields={"Experiment": "run-1", "lr": "3e-4"}))
        assert result is True
        neptune.init_run.assert_called_once_with(name="run-1", project="org/project")

    def test_train_start_reuses_existing_run(self):
        neptune = self._mock_neptune()
        run = MagicMock()
        with patch.dict(sys.modules, {"neptune": neptune}):
            backend = NeptuneBackend(run=run)
            backend.emit(_event(kind="train_start", fields={"Experiment": "run-1", "lr": "1e-3"}))
        neptune.init_run.assert_not_called()

    def test_epoch_end_appends_metrics(self):
        neptune = self._mock_neptune()
        run = MagicMock()
        with patch.dict(sys.modules, {"neptune": neptune}):
            backend = NeptuneBackend(run=run)
            backend.emit(_event(kind="epoch_end", fields={"loss": "0.42"}, step=100))
        run.__getitem__.return_value.append.assert_called()

    def test_train_end_stops_run(self):
        neptune = self._mock_neptune()
        run = MagicMock()
        with patch.dict(sys.modules, {"neptune": neptune}):
            backend = NeptuneBackend(run=run)
            backend.emit(_event(kind="train_end", fields={"Total Steps": "5000"}))
        run.stop.assert_called_once()

    def test_no_neptune_installed_returns_false(self):
        with patch.dict(sys.modules, {"neptune": None}):
            backend = NeptuneBackend()
            assert backend.emit(_event(kind="train_start", fields={"Experiment": "e"})) is False


# ── TensorBoard ───────────────────────────────────────────────────────────────

class TestTensorBoardBackend:
    def test_epoch_end_adds_scalars(self):
        writer = MagicMock()
        backend = TensorBoardBackend(writer=writer)
        backend.emit(_event(kind="epoch_end", fields={"loss": "0.42", "val_loss": "0.38"}, step=10))
        calls = {c.args[0]: c.args[1] for c in writer.add_scalar.call_args_list}
        assert calls["loss"] == 0.42
        assert calls["val_loss"] == 0.38

    def test_best_metric_adds_scalar(self):
        writer = MagicMock()
        backend = TensorBoardBackend(writer=writer)
        backend.emit(_event(kind="best_metric", fields={"val_loss": "0.31", "Step": "100"}, step=100))
        writer.add_scalar.assert_called_once_with("best/val_loss", 0.31, global_step=100)

    def test_train_end_closes_writer(self):
        writer = MagicMock()
        backend = TensorBoardBackend(writer=writer)
        backend.emit(_event(kind="train_end", fields={"Total Steps": "5000"}))
        writer.close.assert_called_once()

    def test_error_adds_text(self):
        writer = MagicMock()
        backend = TensorBoardBackend(writer=writer)
        backend.emit(_event(kind="error", description="OOM"))
        writer.add_text.assert_called_with("error", "OOM")

    def test_no_tensorboard_installed_returns_false(self):
        with patch.dict(sys.modules, {"torch": None, "torch.utils": None,
                                       "torch.utils.tensorboard": None, "tensorboardX": None}):
            backend = TensorBoardBackend()
            assert backend.emit(_event(kind="epoch_end", fields={"loss": "0.5"})) is False


# ── Aim ───────────────────────────────────────────────────────────────────────

class TestAimBackend:
    def _mock_aim(self):
        aim = MagicMock()
        return aim

    def test_train_start_inits_run(self):
        aim = self._mock_aim()
        with patch.dict(sys.modules, {"aim": aim}):
            backend = AimBackend(experiment="my-exp")
            result = backend.emit(_event(kind="train_start", fields={"Experiment": "run-1", "lr": "3e-4"}))
        assert result is True
        aim.Run.assert_called_once_with(experiment="my-exp")

    def test_train_start_reuses_existing_run(self):
        aim = self._mock_aim()
        run = MagicMock()
        with patch.dict(sys.modules, {"aim": aim}):
            backend = AimBackend(run=run)
            backend.emit(_event(kind="train_start", fields={"Experiment": "run-1", "lr": "1e-3"}))
        aim.Run.assert_not_called()

    def test_epoch_end_tracks_metrics(self):
        aim = self._mock_aim()
        run = MagicMock()
        with patch.dict(sys.modules, {"aim": aim}):
            backend = AimBackend(run=run)
            backend.emit(_event(kind="epoch_end", fields={"loss": "0.42"}, step=5))
        run.track.assert_called_once_with(0.42, name="loss", step=5)

    def test_train_end_closes_run(self):
        aim = self._mock_aim()
        run = MagicMock()
        with patch.dict(sys.modules, {"aim": aim}):
            backend = AimBackend(run=run)
            backend.emit(_event(kind="train_end", fields={"Total Steps": "1000"}))
        run.close.assert_called_once()

    def test_no_aim_installed_returns_false(self):
        with patch.dict(sys.modules, {"aim": None}):
            backend = AimBackend()
            assert backend.emit(_event(kind="train_start", fields={"Experiment": "e"})) is False


# ── Console ───────────────────────────────────────────────────────────────────

class TestConsoleBackend:
    def test_emit_always_true(self, capsys):
        assert ConsoleBackend().emit(_event()) is True

    def test_output_contains_title(self, capsys):
        ConsoleBackend(use_color=False).emit(_event(title="My Title"))
        assert "My Title" in capsys.readouterr().out

    def test_output_contains_fields(self, capsys):
        ConsoleBackend(use_color=False).emit(_event(fields={"loss": "0.5"}))
        out = capsys.readouterr().out
        assert "loss" in out
        assert "0.5" in out

    def test_output_contains_description(self, capsys):
        ConsoleBackend(use_color=False).emit(_event(description="some progress"))
        assert "some progress" in capsys.readouterr().out
