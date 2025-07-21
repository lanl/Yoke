"""Tests for logging utilities."""

import logging
import re
import importlib

import pytest

from yoke.helpers import logger as logmod


def reload_logger_module() -> None:
    """Reload the logger module to reset its global state."""
    importlib.reload(logmod)


def test_get_logger_without_init_returns_none() -> None:
    """Ensure get_logger returns None before any configure_logger call."""
    reload_logger_module()
    assert logmod.get_logger() is None


def test_configure_logger_sets_global_logger() -> None:
    """Test that configure_logger sets and returns the module-level logger."""
    reload_logger_module()
    lg = logmod.configure_logger("test_cfg1", logging.DEBUG, log_time=False)
    assert isinstance(lg, logging.Logger)
    assert logmod.logger is lg


def test_log_time_false_formats_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Check that formatting without timestamp prints level and message only."""
    reload_logger_module()
    lg = logmod.configure_logger("test_fmt1", logging.INFO, log_time=False)

    lg.debug("dbg")  # should be filtered out
    lg.info("info_msg")
    captured = capsys.readouterr()
    out = captured.out

    # Should include level and message
    assert out.startswith("INFO:")
    assert "info_msg" in out

    # Should include the ".py:" pattern indicating filename:lineno
    assert ".py:" in out


def test_log_time_true_includes_timestamp(capsys: pytest.CaptureFixture[str]) -> None:
    """Check formatting with timestamp when log_time=True."""
    reload_logger_module()
    lg = logmod.configure_logger("test_fmt2", logging.INFO, log_time=True)

    lg.info("hello_time")
    captured = capsys.readouterr()
    out = captured.out

    # Should include ISO date pattern
    assert re.search(r"\d{4}-\d{2}-\d{2}", out)
    assert "hello_time" in out


def test_level_filtering_warn_and_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify that messages below the set level are filtered out."""
    reload_logger_module()
    lg = logmod.configure_logger("test_lvl", logging.ERROR, log_time=False)

    lg.warning("warn_msg")
    lg.error("err_msg")
    captured = capsys.readouterr()
    out = captured.out

    assert "warn_msg" not in out
    assert "err_msg" in out


def test_get_logger_returns_existing_instance() -> None:
    """Ensure get_logger returns the same logger after initialization."""
    reload_logger_module()
    lg1 = logmod.configure_logger("test_get", logging.DEBUG, log_time=False)
    lg2 = logmod.get_logger()
    assert lg1 is lg2
