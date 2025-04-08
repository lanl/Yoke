"""Scheduled sampling functions."""

import numpy as np


# Exponential, inverse sigmoid, and linear from Bengio et al. 2015
# "Scheduled Sampling for Sequence Prediction with
# Recurrent Neural Networks"
# https://proceedings.neurips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html
# with slight modifications.


def exponential(
    initial_schedule_prob: float = 1.0,
    decay_param: float = 1.0,
    minimum_schedule_prob: float = 0.0,
) -> float:
    """Exponential decay sampling function."""
    return lambda x: np.clip(
        initial_schedule_prob * (decay_param**x),
        a_min=minimum_schedule_prob,
        a_max=1.0,
    )


def inverse_sigmoid(
    initial_schedule_prob: float = 1.0,
    decay_param: float = 1.0,
    minimum_schedule_prob: float = 0.0,
) -> float:
    """Inverse sigmoid scheduled sampling function."""
    return lambda x: np.clip(
        initial_schedule_prob
        * decay_param
        / (decay_param + np.exp(x / decay_param) - 1.0),
        a_min=minimum_schedule_prob,
        a_max=1.0,
    )


def linear(
    initial_schedule_prob: float = 1.0,
    decay_param: float = 0.0,
    minimum_schedule_prob: float = 0.0,
) -> float:
    """Linear scheduled sampling function."""
    return lambda x: np.clip(
        initial_schedule_prob * (1.0 - decay_param * x),
        a_min=minimum_schedule_prob,
        a_max=1.0,
    )
