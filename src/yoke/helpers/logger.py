"""
logger.py

This module provides logging configuration for Python applications. It sets up a
logger with a custom formatter that includes log level, timestamp, filename, line
number, and the log message. The logger is configured to output logs to the console.

Usage:
    Import this module and use the logger instance to log messages.

Example Usage:

.. code-block:: python

    import yoke.utils.logger as yl

    yl.init()

    # Or, initialize with one of the logging levels below.
    yl.init(logging.DEBUG)
    yl.init(logging.INFO)
    yl.init(logging.WARNING)
    yl.init(logging.ERROR)
    yl.init(logging.CRITICAL)

    # Alternately, after yl.init(), logging level can be set as below.
    yl.logger.setLevel(logging.DEBUG)
    yl.logger.setLevel(logging.INFO)
    yl.logger.setLevel(logging.WARNING)
    yl.logger.setLevel(logging.ERROR)
    yl.logger.setLevel(logging.CRITICAL)

    # Use logging.
    logger.debug("DEBUG message.")       # For debug level messages
    logger.info("INFO message.")         # For info level messages
    logger.warning("WARNING message.")   # For warning level messages
    logger.error("ERROR message.")       # For error messages
    logger.critical("CRITICAL message.") # For critical messages

Note about logging levels:
    Logging level hierarchy: DEBUG < INFO < WARNING < ERROR < CRITICAL
    Messages for levels including and above the set log level will be printed.
    For example:
    - Setting log level to DEBUG will cause all log level messages to print
    - Setting log level to ERROR will cause ERROR and CRITICAL log messages to print
    - Setting log level to CRITICAL will only print CRITICAL messages
"""

import sys
import logging

logger = None


def configure_logger(
    name: str, level: int = logging.CRITICAL, log_time: bool = False
) -> logging.Logger:
    """
    Configures and returns a logger instance.

    Args:
        name (str): The name of the logger, typically set to `__name__` for the calling module.
        level (int, optional): The logging level. Defaults to `logging.DEBUG`.

    Returns:
        logging.Logger: A configured logger instance.
    """

    global logger

    # Create a logger with the specified name and level
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)

    # Define a custom formatter
    if log_time:
        formatter = logging.Formatter(
            "%(levelname)s: %(asctime)s: " + "%(filename)s:%(lineno)4d - %(message)s"
        )
    else:
        formatter = logging.Formatter(
            "%(levelname)s: " + "%(filename)s:%(lineno)4d - %(message)s"
        )

    # Attach the formatter to the handler
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    return logger

def get_logger() -> logging.Logger:
    """"
    Returns the logger instance.

    Returns:
        logging.Logger: The logger instance.
    """
    return logger