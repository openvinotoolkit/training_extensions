# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import multiprocessing as mp
from collections.abc import Callable, Generator
from multiprocessing.synchronize import Condition, Event
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.services.event.event_bus import EventBus, EventType

type EventBusFactory = Callable[[Condition | None, Condition | None, Event | None], EventBus]


@pytest.fixture
def fxt_db_session() -> Generator[Session]:
    """A real (in-memory) SQLAlchemy session to exercise the commit-deferred emission."""
    engine = create_engine("sqlite://", poolclass=StaticPool)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def fxt_event_bus() -> EventBusFactory:
    def _create_bus(
        source_changed_condition: Condition | None,
        sink_changed_condition: Condition | None,
        model_reload_event: Event | None,
    ) -> EventBus:
        return EventBus(
            source_changed_condition=source_changed_condition,
            sink_changed_condition=sink_changed_condition,
            model_reload_event=model_reload_event,
        )

    return _create_bus


class TestEventBus:
    """Unit tests for TestEventBus."""

    @pytest.mark.parametrize("event_type", EventType)
    def test_subscribe(self, event_type: EventType, fxt_event_bus: EventBusFactory) -> None:
        """Test subscription"""
        handler = MagicMock(spec=Callable)
        event_bus = fxt_event_bus(None, None, None)

        event_bus.subscribe(event_types=[event_type], handler=handler)

        assert event_bus._event_handlers == {event_type: [handler]}

    def test_source_changed(self, fxt_event_bus: EventBusFactory) -> None:
        """Test source changed"""
        handler = MagicMock(spec=Callable)
        source_changed_condition = MagicMock(spec=Condition)
        event_bus = fxt_event_bus(source_changed_condition, None, None)
        event_bus.subscribe(event_types=[EventType.SOURCE_CHANGED], handler=handler)

        event_bus.emit_event(EventType.SOURCE_CHANGED)

        handler.assert_called_once_with()
        source_changed_condition.notify_all.assert_called_once_with()

    def test_sink_changed(self, fxt_event_bus: EventBusFactory) -> None:
        """Test sink changed"""
        handler = MagicMock(spec=Callable)
        sink_changed_condition = MagicMock(spec=Condition)
        event_bus = fxt_event_bus(None, sink_changed_condition, None)
        event_bus.subscribe(event_types=[EventType.SINK_CHANGED], handler=handler)

        event_bus.emit_event(EventType.SINK_CHANGED)

        handler.assert_called_once_with()
        sink_changed_condition.notify_all.assert_called_once_with()

    def test_model_changed(self, fxt_event_bus: EventBusFactory) -> None:
        """Test model changed"""
        handler = MagicMock(spec=Callable)
        model_reload_event = mp.Event()
        event_bus = fxt_event_bus(None, None, model_reload_event)
        event_bus.subscribe(event_types=[EventType.MODEL_CHANGED], handler=handler)

        event_bus.emit_event(EventType.MODEL_CHANGED)

        handler.assert_called_once_with()
        assert model_reload_event.is_set()

    def test_pipeline_dataset_collection_policies_changed(self, fxt_event_bus: EventBusFactory) -> None:
        """Test pipeline dataset collection policies changed"""
        handler = MagicMock(spec=Callable)
        event_bus = fxt_event_bus(None, None, None)
        event_bus.subscribe(event_types=[EventType.PIPELINE_DATASET_COLLECTION_POLICIES_CHANGED], handler=handler)

        event_bus.emit_event(EventType.PIPELINE_DATASET_COLLECTION_POLICIES_CHANGED)

        handler.assert_called_once_with()

    def test_pipeline_status_changed(self, fxt_event_bus: EventBusFactory) -> None:
        """Test pipeline status changed"""
        handler = MagicMock(spec=Callable)
        source_changed_condition = MagicMock(spec=Condition)
        sink_changed_condition = MagicMock(spec=Condition)
        model_reload_event = mp.Event()
        event_bus = fxt_event_bus(source_changed_condition, sink_changed_condition, model_reload_event)
        event_bus.subscribe(event_types=[EventType.PIPELINE_STATUS_CHANGED], handler=handler)

        event_bus.emit_event(EventType.PIPELINE_STATUS_CHANGED)

        handler.assert_called_once_with()
        source_changed_condition.notify_all.assert_called_once_with()
        sink_changed_condition.notify_all.assert_called_once_with()
        assert model_reload_event.is_set()

    def test_inference_device_changed(self, fxt_event_bus: EventBusFactory) -> None:
        """Test inference device changed triggers model reload"""
        handler = MagicMock(spec=Callable)
        model_reload_event = mp.Event()
        event_bus = fxt_event_bus(None, None, model_reload_event)
        event_bus.subscribe(event_types=[EventType.INFERENCE_DEVICE_CHANGED], handler=handler)

        event_bus.emit_event(EventType.INFERENCE_DEVICE_CHANGED)

        handler.assert_called_once_with()
        assert model_reload_event.is_set()

    def test_inference_params_refresh_without_model_reload(self) -> None:
        """Inference params (including label colors) refresh the worker without reloading the model."""
        handler = MagicMock(spec=Callable)
        model_reload_event = mp.Event()
        inference_params_event = mp.Event()
        event_bus = EventBus(model_reload_event=model_reload_event, inference_params_event=inference_params_event)
        event_bus.subscribe(event_types=[EventType.INFERENCE_PARAMS_CHANGED], handler=handler)

        event_bus.emit_event(EventType.INFERENCE_PARAMS_CHANGED)

        handler.assert_called_once_with()
        assert inference_params_event.is_set()
        # Refreshing the cached inference params (incl. label colors) must not reload the model.
        assert not model_reload_event.is_set()

    def test_emit_event_survives_failing_handler(self, fxt_event_bus: EventBusFactory) -> None:
        """A failing subscriber must not stop other subscribers or the condition notification."""
        failing_handler = MagicMock(spec=Callable, side_effect=RuntimeError("boom"))
        ok_handler = MagicMock(spec=Callable)
        source_changed_condition = MagicMock(spec=Condition)
        model_reload_event = mp.Event()
        event_bus = fxt_event_bus(source_changed_condition, None, model_reload_event)
        # Subscribe the failing handler first so it would otherwise short-circuit the rest.
        event_bus.subscribe(event_types=[EventType.PIPELINE_STATUS_CHANGED], handler=failing_handler)
        event_bus.subscribe(event_types=[EventType.PIPELINE_STATUS_CHANGED], handler=ok_handler)

        # Must not raise despite the first handler failing.
        event_bus.emit_event(EventType.PIPELINE_STATUS_CHANGED)

        failing_handler.assert_called_once_with()
        # The remaining handler is still invoked ...
        ok_handler.assert_called_once_with()
        # ... and the side-effect notifications still happen (condition notified, model reload set).
        source_changed_condition.notify_all.assert_called_once_with()
        assert model_reload_event.is_set()

    def test_emit_event_after_commit_defers_until_commit(
        self, fxt_event_bus: EventBusFactory, fxt_db_session: Session
    ) -> None:
        """`emit_event_after_commit` must not notify anyone until the session actually commits."""
        handler = MagicMock(spec=Callable)
        source_changed_condition = MagicMock(spec=Condition)
        event_bus = fxt_event_bus(source_changed_condition, None, None)
        event_bus.subscribe(event_types=[EventType.SOURCE_CHANGED], handler=handler)
        fxt_db_session.begin()

        event_bus.emit_event_after_commit(fxt_db_session, EventType.SOURCE_CHANGED)

        # Nothing must happen before the commit: consumers would otherwise read stale data.
        handler.assert_not_called()
        source_changed_condition.notify_all.assert_not_called()

        fxt_db_session.commit()

        handler.assert_called_once_with()
        source_changed_condition.notify_all.assert_called_once_with()

    def test_emit_event_after_commit_skipped_on_rollback(
        self, fxt_event_bus: EventBusFactory, fxt_db_session: Session
    ) -> None:
        """If the transaction is rolled back, the event is never emitted."""
        handler = MagicMock(spec=Callable)
        event_bus = fxt_event_bus(None, None, None)
        event_bus.subscribe(event_types=[EventType.SOURCE_CHANGED], handler=handler)
        fxt_db_session.begin()

        event_bus.emit_event_after_commit(fxt_db_session, EventType.SOURCE_CHANGED)
        fxt_db_session.rollback()

        handler.assert_not_called()
