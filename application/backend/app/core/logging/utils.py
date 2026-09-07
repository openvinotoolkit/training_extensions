# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager

from loguru import logger

from .config import LogConfig
from .handlers import InterceptHandler


@contextmanager
def job_log_sink(config: LogConfig) -> Generator[str]:
    """Temporarily attach *only* a file sink for a job log file.

    Unlike :func:`logging_ctx` this does not touch the stdlib ``logging`` root
    handlers and does not import heavy third-party modules, which makes it safe
    to use from the parent process (e.g. the job control plane) to append a few
    records to a job log file that is otherwise written by the child process.

    Args:
        config: LogConfig instance specifying the log file path and sink parameters.

    Yields:
        str: Full path to the log file.
    """
    log_path = os.path.join(config.log_folder, config.log_file)
    sink_id = logger.add(
        log_path,
        rotation=config.rotation,
        retention=config.retention,
        level=config.level,
        serialize=config.serialize,
        enqueue=True,
    )
    try:
        yield log_path
    finally:
        logger.remove(sink_id)


@contextmanager
def logging_ctx(config: LogConfig) -> Generator[str]:
    """Create a temporary logging context with an additional file sink.

    Adds a context-specific log file sink that captures all logs emitted within
    the context. The sink is automatically removed on exit, but the log file
    persists. Logs continue to go to all other configured sinks (stdout, main
    log file, etc.).

    While active, this also installs a fresh `InterceptHandler` on the root
    stdlib logger and on `nncf`'s logger (which doesn't propagate to root),
    so log records from third-party libraries using stdlib `logging` are
    forwarded into loguru and captured by the job-specific sink too. Any
    handlers/levels set on these loggers prior to entering the context are
    saved and restored on exit, so this is safe to use regardless of whether
    root logging was already configured (e.g. by `app.main`) in the current
    process.

    Useful for capturing logs from specific operations (e.g., training jobs) into
    separate files while maintaining application-wide logging.

    Args:
        config: LogConfig instance specifying the log file path, rotation,
                retention, and other sink parameters.

    Yields:
        str: Full path to the created log file.

    Raises:
        RuntimeError: If the log sink cannot be added (e.g., due to permission
                      issues or invalid configuration).

    Example:
        >>> log_config = LogConfig(
        ...     log_folder="logs/jobs",
        ...     log_file="train-8f3e22f2.log"
        ... )
        >>> with logging_ctx(log_config) as logging_path:
        ...     logger.info("Training started")  # Logged to both main and job-specific file
        ...     # ... training code ...
        >>> # Sink removed, but logs/jobs/train-8f3e22f2.log persists
    """
    from nncf.common.logging.logger import nncf_logger

    log_path = os.path.join(config.log_folder, config.log_file)

    root_logger = logging.getLogger()
    prev_root_handlers = list(root_logger.handlers)
    prev_root_level = root_logger.level

    prev_nncf_handlers = list(nncf_logger.handlers)
    prev_nncf_level = nncf_logger.level

    try:
        sink_id = logger.add(
            log_path,
            rotation=config.rotation,
            retention=config.retention,
            level=config.level,
            serialize=config.serialize,
            enqueue=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to add log sink for {log_path}: {e}") from e

    level = getattr(logging, config.level.upper(), logging.INFO)

    try:
        # Replace handlers so there's always exactly one InterceptHandler on root,
        # regardless of whether basicConfig already ran in the process utilizing this
        # function (e.g. via the multiprocessing "spawn" main-script reimport) or not.
        root_logger.handlers = [InterceptHandler()]
        root_logger.setLevel(level)

        # nncf_logger doesn't propagate, so it needs its own handler
        nncf_logger.handlers = [InterceptHandler()]
        nncf_logger.setLevel(level)

        logger.debug("Started logging to {}", log_path)
        yield log_path
    finally:
        root_logger.handlers = prev_root_handlers
        root_logger.setLevel(prev_root_level)

        nncf_logger.handlers = prev_nncf_handlers
        nncf_logger.setLevel(prev_nncf_level)

        logger.debug("Stopped logging to {}", log_path)
        logger.remove(sink_id)
