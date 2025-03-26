"""
logger.py

This module provides logging configuration for Python applications. It sets up a
logger with a custom formatter that includes log level, timestamp, filename, line
number, and the log message. The logger is configured to output logs to the console.

Usage:
    Import this module and use the logger instance to log messages.

EXAMPLE USAGE:

    STEP 1:
    import yoke.helpers.logger as yl

    STEP 2:
    yl.init()

    # Or, initialize with one of the logging levels below.
    yl.init(<logging.DEBUG | loggin.INFO | loggin.WARNING | logging.ERROR | loggin.CRITIICAL>)

    # Alternately, after yl.init(), loggin level can be set as below.
    yl.logger.selLevel(<logging.DEBUG | loggin.INFO | loggin.WARNING | logging.ERROR | loggin.CRITIICAL>)

    STEP 3:
    # Use logging.

    logger.debug("DEBUG message.")       # For debug level messages
    logger.info("INFO message.")         # For info level messages
    logger.warning("WARNING message.")   # For warning level messages
    logger.error("ERROR message.")       # For error messages
    logger.critical("CRITICAL message.") # For critical messages

    NOTE About logging levels:
    Logging level hirearchy: DEBUG < INFO < WARNING < ERROR < CRITIICAL

    Messages for levels including and above the set log level will be printed.
    For ex.
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
    return logger

# Example usage:
if __name__ == "__main__":
    """
    Example usage of the logger module.
    """
    # Get a logger instance
    logger = configure_logger("yoke_logger", level=logging.DEBUG)

    for log_level in (
        logging.CRITICAL,
        logging.ERROR,
        logging.WARNING,
        logging.INFO,
        logging.DEBUG,
    ):
        logger.setLevel(log_level)
        # Log some example messages
        print("")
        logger.debug("This is a debug message")
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        logger.critical("This is a critical message")
