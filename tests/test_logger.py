from unittest.mock import MagicMock, patch

import pytest

from train_logger import TrainLogger
from train_logger.backends.base import Backend
from train_logger.event import Event


class _FakeBackend(Backend):
    def __init__(self, return_value: bool = True):
        self.events: list[Event] = []
        self._return_value = return_value

    def emit(self, event: Event) -> bool:
        self.events.append(event)
        return self._return_value


@pytest.fixture
def backend():
    return _FakeBackend()


@pytest.fixture
def logger(backend):
    return TrainLogger(backend)


def test_on_train_start(logger, backend):
    assert logger.on_train_start("my-exp", config={"lr": 3e-4}) is True
    assert backend.events[0].kind == "train_start"
    assert backend.events[0].color == "green"
    assert "Experiment" in backend.events[0].fields


def test_on_train_start_no_config(logger, backend):
    assert logger.on_train_start("exp") is True


def test_on_epoch_end(logger, backend):
    assert logger.on_epoch_end(5, {"loss": 0.4}, total_epochs=10, step=500) is True
    e = backend.events[0]
    assert e.kind == "epoch_end"
    assert "50.0%" in e.description
    assert "Step: 500" in e.description


def test_on_epoch_end_no_total(logger, backend):
    assert logger.on_epoch_end(1, {"loss": 0.5}) is True


def test_on_step_end(logger, backend):
    assert logger.on_step_end(100, {"loss": 0.5}, total_steps=1000) is True
    e = backend.events[0]
    assert e.kind == "step_end"
    assert "10.0%" in e.description


def test_on_best_metric(logger, backend):
    assert logger.on_best_metric("val_loss", 0.312, step=500) is True
    e = backend.events[0]
    assert e.kind == "best_metric"
    assert e.color == "yellow"


def test_on_checkpoint_saved(logger, backend):
    assert logger.on_checkpoint_saved(step=1000, path="/ckpt/step_1000.pt") is True
    assert backend.events[0].kind == "checkpoint"


def test_on_train_end(logger, backend):
    assert logger.on_train_end(total_steps=5000, best_val_loss=0.31) is True
    e = backend.events[0]
    assert e.kind == "train_end"
    assert e.color == "green"


def test_on_error(logger, backend):
    assert logger.on_error(ValueError("bad"), context="train_step") is True
    e = backend.events[0]
    assert e.kind == "error"
    assert e.color == "red"
    assert "Context: train_step" in e.description


def test_send(logger, backend):
    assert logger.send("hello", color="orange") is True
    assert backend.events[0].color == "orange"


def test_send_file_missing(logger, backend):
    assert logger.send_file("/nonexistent/file.pt") is False


def test_catch_errors_no_exception(logger, backend):
    with logger.catch_errors("ctx"):
        pass
    assert len(backend.events) == 0


def test_catch_errors_with_exception(logger, backend):
    with pytest.raises(ValueError):
        with logger.catch_errors("ctx"):
            raise ValueError("boom")
    assert backend.events[0].kind == "error"


def test_add_backend(backend):
    b2 = _FakeBackend()
    logger = TrainLogger(backend)
    logger.add_backend(b2)
    logger.send("hi")
    assert len(backend.events) == 1
    assert len(b2.events) == 1


def test_fan_out_all_return_true():
    b1, b2 = _FakeBackend(True), _FakeBackend(True)
    assert TrainLogger(b1, b2).send("hi") is True


def test_fan_out_one_fails():
    b1, b2 = _FakeBackend(True), _FakeBackend(False)
    assert TrainLogger(b1, b2).send("hi") is False


def test_no_backends():
    assert TrainLogger().send("hi") is False
