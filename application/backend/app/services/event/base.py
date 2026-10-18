# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from collections.abc import Callable, Sequence
from threading import RLock
from typing import TypeVar

from loguru import logger

E = TypeVar("E")


class BaseEventBus[E]:
    def __init__(self) -> None:
        self._event_handlers: dict[E, list[Callable]] = defaultdict(list)
        self._lock = RLock()

    def subscribe(self, event_types: Sequence[E], handler: Callable) -> None:
        with self._lock:
            for event_type in event_types:
                self._event_handlers[event_type].append(handler)
                logger.debug("registered event handler for event '{}'", event_type)

    def emit_event(self, event_type: E) -> None:
        with self._lock:
            handlers = list(self._event_handlers[event_type])
        logger.debug("Emitting event '{}' to {} handlers", event_type, len(handlers))
        for handler in handlers:
            try:
                handler()
            except Exception as e:
                # A misbehaving subscriber must not prevent the remaining subscribers from being
                # notified, nor propagate back to the caller that emitted the event (which could be
                # a running pipeline worker or an API request). Log and continue.
                logger.exception("Event handler for event '{}' raised an exception; continuing {}", event_type, e)
