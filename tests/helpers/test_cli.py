"""Test cli module."""

import argparse

from yoke.helpers import cli


def test_add_default_args() -> None:
    """Ensure default argparser runs without crashing."""
    # Test default use case.
    cli.add_default_args()

    # Test use case of adding to existing parser.
    cli.add_default_args(argparse.ArgumentParser())


def test_add_filepath_args() -> None:
    """Ensure filepath argparser runs without crashing."""
    cli.add_filepath_args(argparse.ArgumentParser())


def test_add_computing_args() -> None:
    """Ensure computing argparser runs without crashing."""
    cli.add_computing_args(argparse.ArgumentParser())


def test_add_model_args() -> None:
    """Ensure model argparser runs without crashing."""
    cli.add_model_args(argparse.ArgumentParser())


def test_add_training_args() -> None:
    """Ensure training argparser runs without crashing."""
    cli.add_training_args(argparse.ArgumentParser())


def test_add_step_lr_scheduler_args() -> None:
    """Ensure step lr scheduler argparser runs without crashing."""
    cli.add_step_lr_scheduler_args(argparse.ArgumentParser())


def test_add_cosine_lr_scheduler_args() -> None:
    """Ensure cosine lr scheduler argparser runs without crashing."""
    cli.add_cosine_lr_scheduler_args(argparse.ArgumentParser())


def test_add_scheduled_sampling_args() -> None:
    """Ensure scheduled sampling argparser runs without crashing."""
    cli.add_scheduled_sampling_args(argparse.ArgumentParser())
