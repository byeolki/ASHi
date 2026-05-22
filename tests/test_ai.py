from unittest.mock import MagicMock, patch

import pytest

from train_logger import AISummarizer, TrainLogger
from train_logger.ai import _build_context
from train_logger.backends.base import Backend
from train_logger.event import Event


class _FakeBackend(Backend):
    def __init__(self):
        self.events: list[Event] = []

    def emit(self, event: Event) -> bool:
        self.events.append(event)
        return True


def _mock_openai(text: str = "Training looks good."):
    resp = MagicMock()
    resp.ok = True
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": text}}]}
    return resp


# ── AISummarizer unit tests ───────────────────────────────────────────────────

class TestAISummarizer:
    def test_summarize_no_history(self):
        s = AISummarizer(api_key="sk-test")
        assert s.summarize() == "No training history recorded yet."

    def test_record_train_start(self):
        s = AISummarizer(api_key="sk-test")
        s.record(Event(kind="train_start", title="Training Started", fields={"Experiment": "exp", "lr": "3e-4"}))
        assert s._experiment == "exp"
        assert s._config["lr"] == "3e-4"

    def test_record_epoch_end(self):
        s = AISummarizer(api_key="sk-test")
        s.record(Event(kind="epoch_end", title="Epoch 1 / 10", fields={"loss": "0.42"}))
        assert len(s._log) == 1

    def test_record_irrelevant_event_ignored(self):
        s = AISummarizer(api_key="sk-test")
        s.record(Event(kind="message", title="custom"))
        assert len(s._log) == 0

    def test_summarize_calls_openai(self):
        s = AISummarizer(api_key="sk-test")
        s.record(Event(kind="epoch_end", title="Epoch 1", fields={"loss": "0.5"}))

        with patch("train_logger.ai.requests.post", return_value=_mock_openai("Loss is converging.")) as mock:
            result = s.summarize()

        assert result == "Loss is converging."
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-test"
        payload = call_kwargs["json"]
        assert payload["model"] == "gpt-4o-mini"
        assert any("training" in m["content"].lower() for m in payload["messages"])

    def test_summarize_http_error_propagates(self):
        import requests as req
        s = AISummarizer(api_key="sk-bad")
        s.record(Event(kind="epoch_end", title="Epoch 1", fields={}))
        err_resp = MagicMock()
        err_resp.raise_for_status.side_effect = req.HTTPError("401")
        with patch("train_logger.ai.requests.post", return_value=err_resp):
            with pytest.raises(req.HTTPError):
                s.summarize()

    def test_custom_model(self):
        s = AISummarizer(api_key="sk-test", model="gpt-4o")
        s.record(Event(kind="epoch_end", title="Epoch 1", fields={}))
        with patch("train_logger.ai.requests.post", return_value=_mock_openai()) as mock:
            s.summarize()
        assert mock.call_args.kwargs["json"]["model"] == "gpt-4o"


# ── TrainLogger.summarize integration ────────────────────────────────────────

class TestTrainLoggerSummarize:
    def test_summarize_no_summarizer_raises(self):
        logger = TrainLogger()
        with pytest.raises(RuntimeError, match="No summarizer configured"):
            logger.summarize()

    def test_summarize_posts_to_backends(self):
        backend = _FakeBackend()
        s = AISummarizer(api_key="sk-test")

        logger = TrainLogger(backend, summarizer=s)
        logger.on_train_start("exp", config={"lr": 3e-4})
        logger.on_epoch_end(1, {"loss": 0.5}, total_epochs=10)

        with patch("train_logger.ai.requests.post", return_value=_mock_openai("Looks great!")):
            result = logger.summarize()

        assert result == "Looks great!"
        summary_event = next(e for e in backend.events if e.kind == "ai_summary")
        assert summary_event.description == "Looks great!"
        assert summary_event.color == "orange"

    def test_summarize_records_events_automatically(self):
        backend = _FakeBackend()
        s = AISummarizer(api_key="sk-test")
        logger = TrainLogger(backend, summarizer=s)

        logger.on_epoch_end(1, {"loss": 0.8})
        logger.on_epoch_end(2, {"loss": 0.6})
        logger.on_epoch_end(3, {"loss": 0.4})

        assert len(s._log) == 3


# ── _build_context ─────────────────────────────────────────────────────────────

def test_build_context_includes_experiment():
    ctx = _build_context("my-exp", {"lr": "3e-4"}, [{"title": "Epoch 1", "fields": {"loss": "0.5"}}])
    assert "my-exp" in ctx
    assert "lr=3e-4" in ctx
    assert "loss=0.5" in ctx


def test_build_context_truncates_middle():
    log = [{"title": f"Epoch {i}", "fields": {"loss": str(0.9 - i * 0.01)}} for i in range(30)]
    ctx = _build_context("exp", {}, log)
    assert "omitted" in ctx
