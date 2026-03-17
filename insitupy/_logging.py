"""
Logging utilities for InSituPy.

This module provides custom logging handlers and context managers for managing
log output during batch operations with progress bars.
"""

import io
import logging
import sys
import warnings
from collections import Counter
from contextlib import contextmanager


def setup_logging():
    """Configure the insitupy package logger with a TqdmLoggingHandler."""
    logger = logging.getLogger('insitupy')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Only add handler if one doesn't exist yet
    # Use TqdmLoggingHandler to prevent progress bar disruption from log messages
    if not logger.handlers:
        handler = TqdmLoggingHandler()
        handler.setFormatter(
            logging.Formatter('%(asctime)s | [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        )
        logger.addHandler(handler)


class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler that uses tqdm.write() to output log messages.

    This ensures that log messages don't interfere with tqdm progress bars,
    as tqdm.write() properly handles output interleaving with progress bars.
    """

    def emit(self, record):
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


class WarningCollector:
    """
    A class to collect warnings and log messages during batch operations.

    Warnings are deduplicated and counted, then can be printed as a summary.
    """

    def __init__(self):
        self.warnings = []
        self._log_records = []
        self._stdout_messages = []

    def add_warning(self, message: str, category: str = "Warning"):
        """Add a warning message to the collection."""
        self.warnings.append((category, message))

    def add_log_record(self, record: logging.LogRecord):
        """Add a log record to the collection."""
        self._log_records.append(record)

    def add_stdout_message(self, message: str):
        """Add a captured stdout message to the collection."""
        # Clean up the message
        message = message.strip()
        if message:
            self._stdout_messages.append(message)

    def get_summary(self) -> str:
        """Get a summary of all collected warnings."""
        if not self.warnings and not self._log_records and not self._stdout_messages:
            return ""

        # Count unique warnings
        warning_counts = Counter(self.warnings)
        log_counts = Counter((r.levelname, r.getMessage()) for r in self._log_records)
        stdout_counts = Counter(self._stdout_messages)

        lines = []

        if warning_counts:
            for (category, message), count in warning_counts.items():
                if count > 1:
                    lines.append(f"  [{category}] {message} (x{count})")
                else:
                    lines.append(f"  [{category}] {message}")

        if log_counts:
            for (level, message), count in log_counts.items():
                if count > 1:
                    lines.append(f"  [{level}] {message} (x{count})")
                else:
                    lines.append(f"  [{level}] {message}")

        if stdout_counts:
            for message, count in stdout_counts.items():
                if count > 1:
                    lines.append(f"  [INFO] {message} (x{count})")
                else:
                    lines.append(f"  [INFO] {message}")

        if lines:
            return "Collected warnings:\n" + "\n".join(lines)
        return ""

    def print_summary(self):
        """Print the summary of collected warnings."""
        summary = self.get_summary()
        if summary:
            from tqdm import tqdm
            tqdm.write(summary)

    def __len__(self):
        return len(self.warnings) + len(self._log_records) + len(self._stdout_messages)


class CollectingHandler(logging.Handler):
    """A logging handler that collects records into a WarningCollector."""

    def __init__(self, collector: WarningCollector, level=logging.WARNING):
        super().__init__(level)
        self.collector = collector

    def emit(self, record):
        self.collector.add_log_record(record)


class StdoutCapture(io.StringIO):
    """
    A StringIO wrapper that captures stdout and stores messages in a WarningCollector.

    This allows us to intercept print statements (like those from anndata/scanpy)
    that would otherwise disrupt progress bars.
    """

    def __init__(self, collector: WarningCollector, original_stdout):
        super().__init__()
        self.collector = collector
        self.original_stdout = original_stdout

    def write(self, s):
        # Capture the output
        if s.strip():  # Only capture non-empty, non-whitespace strings
            self.collector.add_stdout_message(s)
        return len(s)

    def flush(self):
        pass  # No-op, we don't need to flush


@contextmanager
def collect_warnings(collector: WarningCollector = None, capture_stdout: bool = True):
    """
    Context manager that collects warnings, log messages, and optionally stdout into a WarningCollector.

    This allows batch operations to run without interruption from warnings or print statements,
    while still capturing them for later review.

    Args:
        collector: An optional WarningCollector instance. If not provided, a new one is created.
        capture_stdout: If True, also capture stdout (print statements from libraries like anndata).

    Yields:
        WarningCollector: The collector containing all captured warnings.

    Example:
        >>> with collect_warnings() as collector:
        ...     # operations that might produce warnings
        ...     pass
        >>> collector.print_summary()
    """
    if collector is None:
        collector = WarningCollector()

    # Set up stdout capture if requested
    original_stdout = None
    if capture_stdout:
        original_stdout = sys.stdout
        sys.stdout = StdoutCapture(collector, original_stdout)

    # Set up warning capture
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        # Set up log capture for insitupy and tifffile loggers
        loggers_to_capture = ['insitupy', 'tifffile']
        collecting_handlers = {}
        old_levels = {}
        disabled_handlers = {}

        for logger_name in loggers_to_capture:
            logger = logging.getLogger(logger_name)
            old_levels[logger_name] = logger.level

            # Temporarily disable existing handlers to prevent duplicate output
            disabled_handlers[logger_name] = logger.handlers.copy()
            for h in disabled_handlers[logger_name]:
                h.setLevel(logging.CRITICAL + 1)  # Effectively disable by setting very high level

            # Add a collecting handler
            handler = CollectingHandler(collector)
            collecting_handlers[logger_name] = handler
            logger.addHandler(handler)

            # Prevent the log messages from going to other handlers
            logger.setLevel(logging.WARNING)
            logger.propagate = False

        try:
            yield collector
        finally:
            # Restore stdout first
            if original_stdout is not None:
                sys.stdout = original_stdout

            # Process captured warnings
            for w in caught_warnings:
                collector.add_warning(str(w.message), w.category.__name__)

            # Remove collecting handlers and restore logger settings
            for logger_name, handler in collecting_handlers.items():
                logger = logging.getLogger(logger_name)
                logger.removeHandler(handler)
                logger.setLevel(old_levels[logger_name])

                # Restore original handler levels
                if logger_name in disabled_handlers:
                    for h in disabled_handlers[logger_name]:
                        h.setLevel(logging.NOTSET)  # Reset to default (use logger's level)

                if logger_name == 'insitupy':
                    logger.propagate = False
                else:
                    logger.propagate = True
