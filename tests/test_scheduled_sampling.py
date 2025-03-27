"""Tests for scheduled sampling utilities."""

from typing import Callable

import numpy as np

from yoke.scheduled_sampling import exponential, inverse_sigmoid, linear


def assert_valid_schedule(
    scheduler: Callable,
    initial_schedule_prob: float,
    decay_param: float,
    minimum_schedule_prob: float,
    n_epochs: int = 1000,
) -> None:
    """Assert that a scheduled sampling scheduler falls within specified range."""
    epochs = np.arange(n_epochs)
    schedule_fxn = scheduler(
        initial_schedule_prob=initial_schedule_prob,
        decay_param=decay_param,
        minimum_schedule_prob=minimum_schedule_prob,
    )
    scheduled_prob = schedule_fxn(epochs)
    assert np.all(scheduled_prob <= 1.0), (
        "`scheduled_prob` should be less than or equal to 1.0!"
    )
    assert np.all(scheduled_prob >= minimum_schedule_prob), (
        "`scheduled_prob` should be greater than or equal to `minimum_schedule_prob`!"
    )


def test_exponential() -> None:
    """Test exponential scheduled sampling function."""
    assert_valid_schedule(
        scheduler=exponential,
        initial_schedule_prob=0.9,
        decay_param=0.99,
        minimum_schedule_prob=0.1,
        n_epochs=1000,
    )


def test_inverse_sigmoid() -> None:
    """Test inverse_sigmoid scheduled sampling function."""
    assert_valid_schedule(
        scheduler=inverse_sigmoid,
        initial_schedule_prob=0.9,
        decay_param=10.0,
        minimum_schedule_prob=0.1,
        n_epochs=1000,
    )


def test_linear() -> None:
    """Test linear scheduled sampling function."""
    assert_valid_schedule(
        scheduler=linear,
        initial_schedule_prob=0.9,
        decay_param=1.0e-2,
        minimum_schedule_prob=0.1,
        n_epochs=1000,
    )
