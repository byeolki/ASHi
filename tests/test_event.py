from ashi.event import Event, format_metrics, make_progress_bar


def test_make_progress_bar_half():
    bar = make_progress_bar(10, 20)
    assert "50.0%" in bar
    assert "█" in bar
    assert "░" in bar


def test_make_progress_bar_complete():
    assert "100.0%" in make_progress_bar(20, 20)


def test_make_progress_bar_overflow_clamps():
    assert "100.0%" in make_progress_bar(25, 20)


def test_make_progress_bar_zero_total():
    assert make_progress_bar(5, 0) == ""


def test_format_metrics_float():
    assert format_metrics({"loss": 0.12345})["loss"] == "0.1235"


def test_format_metrics_int():
    assert format_metrics({"step": 100})["step"] == "100"


def test_event_defaults():
    e = Event(kind="test", title="Hello")
    assert e.fields == {}
    assert e.description == ""
    assert e.color == "blue"
    assert e.file_path is None
