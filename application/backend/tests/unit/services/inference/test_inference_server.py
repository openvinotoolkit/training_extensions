# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import ANY, Mock, patch
from uuid import UUID, uuid4

import numpy as np
import pytest
from model_api.models import Model

from app.models import BatchInferenceInput, DatasetItemAnnotation, Label
from app.models.model_revision import ModelFormat, ModelPrecision, ModelVariant
from app.models.system import DeviceInfo, DeviceType
from app.services import ModelService, ResourceNotFoundError
from app.services.inference import InferenceModel, InferenceServer, InferenceState, InferenceStatus
from app.services.inference.inference_server import InferenceBusyError
from app.services.inference.model_loader import LoadedModelHandle
from app.services.inference.model_registry import EntryState, ModelCacheEntry

CPU = DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)
XPU = DeviceInfo(type=DeviceType.XPU, name="XPU", memory=1024, index=0)


def _cached(server: InferenceServer) -> list[ModelCacheEntry]:
    """The cached entries of the server, most-recently-used first."""
    return list(server._entries.values())


def _handle(model_id: UUID, variant_id: UUID, device: DeviceInfo, model: Mock | None = None) -> LoadedModelHandle:
    """A loaded model handle wrapping a mocked Model API model."""
    return LoadedModelHandle(
        model_id=model_id,
        variant_id=variant_id,
        model=model if model is not None else Mock(spec=Model),
        device=device,
        loaded_at=datetime.now(),
    )


def _seed_entry(
    server: InferenceServer,
    *,
    model_id: UUID | None = None,
    variant_id: UUID | None = None,
    device: DeviceInfo = CPU,
    model: Mock | None = None,
    state: EntryState = EntryState.READY,
    refcount: int = 0,
    last_used: float | None = None,
) -> ModelCacheEntry:
    """Insert a ready-made cache entry at the MRU front of the server cache, bypassing any loading."""
    entry = ModelCacheEntry(
        project_id=uuid4(),
        model_id=model_id if model_id is not None else uuid4(),
        variant_id=variant_id if variant_id is not None else uuid4(),
        device=device,
        xml_path=Path("model.xml"),
        state=state,
        refcount=refcount,
    )
    if state is EntryState.READY:
        entry.handle = _handle(entry.model_id, entry.variant_id, device, model)
        entry.ready.set()
    if last_used is not None:
        entry.last_used = last_used
    server._entries[entry.key] = entry
    server._entries.move_to_end(entry.key, last=False)
    return entry


@contextmanager
def _patched_model_service(
    tmp_path, weights_size: int = 0, variant_id: UUID | None = None
) -> Iterator[dict[UUID, UUID]]:
    """Patch the database lookups performed while resolving a model variant.
    Args:
        tmp_path: Directory holding the fake model files.
        weights_size: On-disk size reported for the resolved variant weights.
        variant_id: Variant id resolved for every model; a fresh one per model when omitted.
    Yields a mapping of model id to the variant id resolved for it.
    """
    variant_ids: dict[UUID, UUID] = {}

    def _get_model_variants(*, project_id: UUID, model_id: UUID) -> list[Mock]:
        resolved_id = variant_ids.setdefault(model_id, variant_id if variant_id is not None else uuid4())
        return [
            Mock(
                spec=ModelVariant,
                id=resolved_id,
                format=ModelFormat.OPENVINO,
                precision=ModelPrecision.FP16,
                weights_size=weights_size,
            )
        ]

    with (
        patch.object(target=ModelService, attribute="get_model_variants", side_effect=_get_model_variants),
        patch.object(
            target=ModelService,
            attribute="get_model_binary_files",
            return_value=(True, (tmp_path / "model.xml", tmp_path / "model.bin")),
        ),
    ):
        yield variant_ids


@contextmanager
def _patched_loader() -> Iterator[tuple[Mock, Mock, list[LoadedModelHandle]]]:
    """Patch the model loader, handing out a fresh handle per load.
    Yields the load and unload mocks, together with the handles returned by each load, in order.
    """
    handles: list[LoadedModelHandle] = []

    def _load(*, model_id: UUID, variant_id: UUID, model_xml_path: Path, device: DeviceInfo) -> LoadedModelHandle:
        handle = _handle(model_id, variant_id, device)
        handles.append(handle)
        return handle

    with (
        patch("app.services.inference.model_loader.ModelLoader.load", side_effect=_load) as mock_load,
        patch("app.services.inference.model_loader.ModelLoader.unload") as mock_unload,
    ):
        yield mock_load, mock_unload, handles


class TestInferenceServer:
    def test_confidence_threshold_param_exists_in_mapi(self) -> None:
        """Ensure that the confidence threshold parameter is defined in the Model API."""
        from model_api.models.parameters import ParameterRegistry

        from app.services.inference.inference_server import CONFIDENCE_THRESHOLD_PARAM

        assert CONFIDENCE_THRESHOLD_PARAM in ParameterRegistry.CONFIDENCE_THRESHOLD

    def test_cache_hit_does_not_reload(self, tmp_path) -> None:
        """A second request for an already cached model on the same device does not load it again."""
        server = InferenceServer(data_dir=Path(tmp_path), max_models=2)
        entry = _seed_entry(server)
        with _patched_loader() as (mock_load, mock_unload, _handles):
            for _ in range(2):
                server.infer_batch(
                    project_id=entry.project_id,
                    model_id=entry.model_id,
                    model_variant_id=entry.variant_id,
                    device=CPU,
                    labels=[],
                    inputs=[],
                )
        mock_load.assert_not_called()
        mock_unload.assert_not_called()
        assert _cached(server) == [entry]

    def test_release_promotes_entry_to_mru_front(self, tmp_path) -> None:
        """Releasing a cached entry makes the access order match the last use time."""
        server = InferenceServer(data_dir=Path(tmp_path), max_models=2)
        newer = _seed_entry(server)
        older = _seed_entry(server)

        assert _cached(server) == [older, newer]

        server._release(newer)

        assert _cached(server) == [newer, older]

    def test_omitted_variant_id_resolves_default_instead_of_reusing_mru_non_default(self, tmp_path) -> None:
        """Omitting the variant id resolves the default FP16 variant even when another cached variant is MRU."""
        server = InferenceServer(data_dir=Path(tmp_path), max_models=2)
        project_id = uuid4()
        model_id = uuid4()
        int8_variant_id = uuid4()
        fp16_variant_id = uuid4()
        int8_entry = _seed_entry(server, model_id=model_id, variant_id=int8_variant_id)
        model_variants = [
            Mock(
                spec=ModelVariant,
                id=int8_variant_id,
                format=ModelFormat.OPENVINO,
                precision=ModelPrecision.INT8,
                weights_size=10,
            ),
            Mock(
                spec=ModelVariant,
                id=fp16_variant_id,
                format=ModelFormat.OPENVINO,
                precision=ModelPrecision.FP16,
                weights_size=20,
            ),
        ]

        def _get_model_binary_files(*, project_id: UUID, model_id: UUID, model_variant_id: UUID):
            return True, (tmp_path / f"{model_variant_id}.xml", tmp_path / f"{model_variant_id}.bin")

        with (
            patch.object(ModelService, "get_model_variants", return_value=model_variants) as mock_get_model_variants,
            patch.object(ModelService, "get_model_binary_files", side_effect=_get_model_binary_files),
            _patched_loader() as (mock_load, mock_unload, _handles),
        ):
            server.infer_batch(project_id=project_id, model_id=model_id, device=CPU, labels=[], inputs=[])
        mock_get_model_variants.assert_called_once_with(project_id=project_id, model_id=model_id)
        mock_load.assert_called_once_with(
            model_id=model_id,
            variant_id=fp16_variant_id,
            model_xml_path=tmp_path / f"{fp16_variant_id}.xml",
            device=CPU,
        )
        mock_unload.assert_not_called()
        assert [entry.variant_id for entry in _cached(server)] == [fp16_variant_id, int8_variant_id]
        assert int8_entry.handle is not None

    def test_device_switch_unloads_then_reloads(self, tmp_path) -> None:
        """Requesting a cached model on another device unloads it and loads it again on the new device."""
        server = InferenceServer(data_dir=Path(tmp_path), max_models=2)
        entry = _seed_entry(server, device=CPU)
        old_handle = entry.handle
        with (
            _patched_model_service(tmp_path, variant_id=entry.variant_id),
            _patched_loader() as (mock_load, mock_unload, _handles),
        ):
            server.infer_batch(
                project_id=entry.project_id,
                model_id=entry.model_id,
                model_variant_id=entry.variant_id,
                device=XPU,
                labels=[],
                inputs=[],
            )
        mock_unload.assert_called_once_with(old_handle)
        mock_load.assert_called_once_with(
            model_id=entry.model_id,
            variant_id=entry.variant_id,
            model_xml_path=tmp_path / "model.xml",
            device=XPU,
        )
        assert [e.device for e in _cached(server)] == [XPU]

    def test_evicts_lru_when_max_models_reached(self, tmp_path) -> None:
        """With a full cache, loading another model evicts the least recently used one."""
        server = InferenceServer(data_dir=Path(tmp_path), max_models=2)
        project_id = uuid4()
        model_a, model_b, model_c = uuid4(), uuid4(), uuid4()
        with _patched_model_service(tmp_path), _patched_loader() as (mock_load, mock_unload, handles):
            for model_id in (model_a, model_b, model_c):
                server.infer_batch(project_id=project_id, model_id=model_id, device=CPU, labels=[], inputs=[])
            evicted_handle = handles[0]
        assert mock_load.call_count == 3
        mock_unload.assert_called_once_with(evicted_handle)
        assert {e.model_id for e in _cached(server)} == {model_b, model_c}

    def test_in_use_model_is_not_evicted(self, tmp_path, monkeypatch) -> None:
        """An entry that is leased is skipped both when making room and when evicting expired models."""
        monkeypatch.setattr("app.services.inference.inference_server.LOCK_ACQUIRE_TIMEOUT", 0.2)
        server = InferenceServer(data_dir=Path(tmp_path), max_models=1, model_ttl=60)
        in_use = _seed_entry(server, refcount=1, last_used=0.0)
        with _patched_model_service(tmp_path), _patched_loader() as (mock_load, mock_unload, _handles):
            with pytest.raises(InferenceBusyError):
                server.infer_batch(project_id=uuid4(), model_id=uuid4(), device=CPU, labels=[], inputs=[])
            server.evict_expired()
        mock_load.assert_not_called()
        mock_unload.assert_not_called()
        assert _cached(server) == [in_use]

    def test_concurrent_requests_for_same_model_load_once(self, tmp_path) -> None:
        """Two threads asking for the same model share a single load and the resulting handle."""
        server = InferenceServer(data_dir=Path(tmp_path), max_models=2)
        project_id, model_id = uuid4(), uuid4()
        load_started = threading.Event()
        finish_load = threading.Event()
        handles: list[LoadedModelHandle | None] = []

        def _blocking_load(*, model_id: UUID, variant_id: UUID, model_xml_path: Path, device: DeviceInfo):
            load_started.set()
            finish_load.wait(timeout=5)
            return _handle(model_id, variant_id, device)

        def _acquire_and_release() -> None:
            entry = server._acquire(project_id=project_id, model_id=model_id, device=CPU, variant_id=None)
            handles.append(entry.handle)
            server._release(entry)

        with (
            _patched_model_service(tmp_path),
            patch("app.services.inference.model_loader.ModelLoader.load", side_effect=_blocking_load) as mock_load,
        ):
            first = threading.Thread(target=_acquire_and_release)
            first.start()
            assert load_started.wait(timeout=5)
            second = threading.Thread(target=_acquire_and_release)
            second.start()
            finish_load.set()
            first.join(timeout=5)
            second.join(timeout=5)
        assert mock_load.call_count == 1
        assert len(handles) == 2
        assert handles[0] is handles[1] is not None
        assert len(_cached(server)) == 1

    def test_load_failure_removes_entry_and_propagates(self, tmp_path) -> None:
        """A failed load leaves no entry behind, propagates the error, and does not poison later requests."""
        server = InferenceServer(data_dir=Path(tmp_path), max_models=2)
        project_id, model_id = uuid4(), uuid4()
        handle = _handle(model_id, uuid4(), CPU)
        with (
            _patched_model_service(tmp_path),
            patch(
                "app.services.inference.model_loader.ModelLoader.load",
                side_effect=[RuntimeError("load failed"), handle],
            ) as mock_load,
        ):
            with pytest.raises(RuntimeError, match="load failed"):
                server.infer_batch(project_id=project_id, model_id=model_id, device=CPU, labels=[], inputs=[])
            assert _cached(server) == []
            server.infer_batch(project_id=project_id, model_id=model_id, device=CPU, labels=[], inputs=[])
        assert mock_load.call_count == 2
        assert [e.handle for e in _cached(server)] == [handle]

    def test_invalidate_model_during_variant_resolution_blocks_insert(self, tmp_path) -> None:
        """A model deleted while its variant is still being resolved must never enter the cache."""
        server = InferenceServer(data_dir=Path(tmp_path), max_models=2)
        project_id, model_id = uuid4(), uuid4()
        resolve_started = threading.Event()
        allow_resolve = threading.Event()
        errors: list[BaseException] = []
        original_resolve = server._resolve_variant

        def _blocking_resolve(project_id: UUID, model_id: UUID, variant_id: UUID | None):
            resolve_started.set()
            allow_resolve.wait(timeout=5)
            return original_resolve(project_id, model_id, variant_id)

        def _infer() -> None:
            try:
                server.infer_batch(project_id=project_id, model_id=model_id, device=CPU, labels=[], inputs=[])
            except BaseException as exc:
                errors.append(exc)

        with (
            _patched_model_service(tmp_path),
            patch.object(server, "_resolve_variant", side_effect=_blocking_resolve),
            patch("app.services.inference.model_loader.ModelLoader.load") as mock_load,
        ):
            worker = threading.Thread(target=_infer)
            worker.start()
            assert resolve_started.wait(timeout=5)

            server.invalidate_model(project_id=project_id, model_id=model_id)
            allow_resolve.set()
            worker.join(timeout=5)

        assert len(errors) == 1
        assert isinstance(errors[0], ResourceNotFoundError)
        mock_load.assert_not_called()
        assert _cached(server) == []

    def test_invalidate_model_during_load_allows_current_load_to_finish_but_blocks_future_requests(
        self, tmp_path
    ) -> None:
        """A model deleted while loading may finish the in-flight request, but remains tombstoned afterward."""
        server = InferenceServer(data_dir=Path(tmp_path), max_models=2)
        project_id, model_id = uuid4(), uuid4()
        load_started = threading.Event()
        allow_finish = threading.Event()
        loaded_handles: list[LoadedModelHandle] = []
        errors: list[BaseException] = []

        def _blocking_load(*, model_id: UUID, variant_id: UUID, model_xml_path: Path, device: DeviceInfo):
            handle = _handle(model_id, variant_id, device)
            loaded_handles.append(handle)
            load_started.set()
            allow_finish.wait(timeout=5)
            return handle

        def _infer() -> None:
            try:
                server.infer_batch(project_id=project_id, model_id=model_id, device=CPU, labels=[], inputs=[])
            except BaseException as exc:
                errors.append(exc)

        with (
            _patched_model_service(tmp_path),
            patch("app.services.inference.model_loader.ModelLoader.load", side_effect=_blocking_load),
            patch("app.services.inference.model_loader.ModelLoader.unload") as mock_unload,
        ):
            worker = threading.Thread(target=_infer)
            worker.start()
            assert load_started.wait(timeout=5)

            server.invalidate_model(project_id=project_id, model_id=model_id)
            allow_finish.set()
            worker.join(timeout=5)

        assert errors == []
        assert len(loaded_handles) == 1
        mock_unload.assert_not_called()
        assert _cached(server) == [_cached(server)[0]]
        assert _cached(server)[0].handle is loaded_handles[0]
        assert _cached(server)[0].state is EntryState.READY
        assert (project_id, model_id) in server._deleted_models

        with pytest.raises(ResourceNotFoundError):
            server.infer_batch(
                project_id=project_id,
                model_id=model_id,
                model_variant_id=_cached(server)[0].variant_id,
                device=CPU,
                labels=[],
                inputs=[],
            )

    def test_stop_during_load_aborts_publish_and_unloads_handle(self, tmp_path) -> None:
        """A load finishing after stop() must not publish READY or proceed with inference."""
        server = InferenceServer(data_dir=Path(tmp_path), max_models=2)
        project_id, model_id = uuid4(), uuid4()
        load_started = threading.Event()
        allow_finish = threading.Event()
        loaded_handles: list[LoadedModelHandle] = []
        errors: list[BaseException] = []

        def _blocking_load(*, model_id: UUID, variant_id: UUID, model_xml_path: Path, device: DeviceInfo):
            handle = _handle(model_id, variant_id, device)
            loaded_handles.append(handle)
            load_started.set()
            allow_finish.wait(timeout=5)
            return handle

        def _infer() -> None:
            try:
                server.infer_batch(project_id=project_id, model_id=model_id, device=CPU, labels=[], inputs=[])
            except BaseException as exc:
                errors.append(exc)

        with (
            _patched_model_service(tmp_path),
            patch("app.services.inference.model_loader.ModelLoader.load", side_effect=_blocking_load),
            patch("app.services.inference.model_loader.ModelLoader.unload") as mock_unload,
        ):
            worker = threading.Thread(target=_infer)
            worker.start()
            assert load_started.wait(timeout=5)
            server.stop()
            allow_finish.set()
            worker.join(timeout=5)
        assert len(errors) == 1
        assert isinstance(errors[0], InferenceBusyError)
        assert len(loaded_handles) == 1
        mock_unload.assert_called_once_with(loaded_handles[0])
        assert _cached(server) == []

    # --- eviction and teardown ---
    def test_evict_expired_unloads_only_expired_entries(self, tmp_path) -> None:
        """Only the models idle for longer than the TTL are unloaded."""
        server = InferenceServer(data_dir=Path(tmp_path), max_models=2, model_ttl=60)
        expired = _seed_entry(server, last_used=0.0)
        fresh = _seed_entry(server)
        expired_handle = expired.handle
        with patch("app.services.inference.model_loader.ModelLoader.unload") as mock_unload:
            server.evict_expired()
        mock_unload.assert_called_once_with(expired_handle)
        assert _cached(server) == [fresh]
        assert expired.state is EntryState.EVICTED

    def test_invalidate_model_marks_deleted_without_unloading_cached_entries(self, tmp_path) -> None:
        """Deleting a model marks it as deleted while keeping cached entries resident."""
        server = InferenceServer(data_dir=Path(tmp_path), max_models=2)
        target = _seed_entry(server)
        other = _seed_entry(server)
        with patch("app.services.inference.model_loader.ModelLoader.unload") as mock_unload:
            server.invalidate_model(project_id=target.project_id, model_id=target.model_id)
        mock_unload.assert_not_called()
        assert _cached(server) == [other, target]
        assert (target.project_id, target.model_id) in server._deleted_models

        with pytest.raises(ResourceNotFoundError):
            server.infer_batch(
                project_id=target.project_id,
                model_id=target.model_id,
                model_variant_id=target.variant_id,
                device=target.device,
                labels=[],
                inputs=[],
            )

    def test_invalidate_model_marks_all_matching_variants_deleted_without_unloading(self, tmp_path) -> None:
        """All cached variants of a deleted model remain loaded but cannot be used for new inference."""
        server = InferenceServer(data_dir=Path(tmp_path), max_models=3)
        project_id = uuid4()
        model_id = uuid4()
        first = _seed_entry(server, model_id=model_id)
        second = _seed_entry(server, model_id=model_id)
        other = _seed_entry(server)
        first.project_id = project_id
        second.project_id = project_id
        matching_entries = [
            entry for entry in _cached(server) if entry.project_id == project_id and entry.model_id == model_id
        ]
        with patch("app.services.inference.model_loader.ModelLoader.unload") as mock_unload:
            server.invalidate_model(project_id=project_id, model_id=model_id)

        mock_unload.assert_not_called()
        assert _cached(server) == [other, second, first]
        assert (project_id, model_id) in server._deleted_models
        assert server._pending_unloads == []
        for entry in matching_entries:
            with pytest.raises(ResourceNotFoundError):
                server.infer_batch(
                    project_id=project_id,
                    model_id=model_id,
                    model_variant_id=entry.variant_id,
                    device=entry.device,
                    labels=[],
                    inputs=[],
                )

    def test_evict_expired_retries_failed_unloads_from_pending_queue(self, tmp_path) -> None:
        """A dropped entry whose unload fails is preserved and retried on the next eviction pass."""
        server = InferenceServer(data_dir=Path(tmp_path), max_models=2, model_ttl=60)
        expired = _seed_entry(server, last_used=0.0)
        expired_handle = expired.handle
        with patch(
            "app.services.inference.model_loader.ModelLoader.unload",
            side_effect=[RuntimeError("unload failed"), None],
        ) as mock_unload:
            with pytest.raises(RuntimeError, match="unload failed"):
                server.evict_expired()
            assert _cached(server) == []
            assert server._pending_unloads == [expired]
            assert expired.handle is expired_handle
            server.evict_expired()
        assert mock_unload.call_count == 2
        assert [call.args[0] for call in mock_unload.call_args_list] == [expired_handle, expired_handle]
        assert server._pending_unloads == []
        assert expired.handle is None

    def test_evict_expired_attempts_all_expired_entries_before_raising(self, tmp_path) -> None:
        """A failed unload of one expired entry does not stop the remaining expired entries from unloading."""
        server = InferenceServer(data_dir=Path(tmp_path), max_models=2, model_ttl=60)
        _seed_entry(server, last_used=0.0)
        _seed_entry(server, last_used=0.0)
        expired_entries = list(_cached(server))
        expected_handles = [entry.handle for entry in expired_entries]

        with (
            patch(
                "app.services.inference.model_loader.ModelLoader.unload",
                side_effect=[RuntimeError("unload failed"), None],
            ) as mock_unload,
            pytest.raises(RuntimeError, match="unload failed"),
        ):
            server.evict_expired()

        assert [call.args[0] for call in mock_unload.call_args_list] == expected_handles
        assert server._pending_unloads == [expired_entries[0]]
        assert _cached(server) == []
        assert expired_entries[1].handle is None

    def test_stop_unloads_all_models(self, tmp_path) -> None:
        """Stopping the server drains the cache and unloads every model."""
        server = InferenceServer(data_dir=Path(tmp_path), max_models=2)
        first = _seed_entry(server)
        second = _seed_entry(server)
        unloaded_handles = [entry.handle for entry in _cached(server)]
        assert unloaded_handles == [second.handle, first.handle]
        with patch("app.services.inference.model_loader.ModelLoader.unload") as mock_unload:
            server.stop()
        assert [call.args[0] for call in mock_unload.call_args_list] == unloaded_handles
        assert _cached(server) == []

    def test_stop_attempts_all_entries_before_raising_unload_error(self, tmp_path) -> None:
        """A failed unload during stop does not prevent later entries from being unloaded."""
        server = InferenceServer(data_dir=Path(tmp_path), max_models=2)
        _seed_entry(server)
        _seed_entry(server)
        stopped_entries = list(_cached(server))
        expected_handles = [entry.handle for entry in stopped_entries]

        with (
            patch(
                "app.services.inference.model_loader.ModelLoader.unload",
                side_effect=[RuntimeError("unload failed"), None],
            ) as mock_unload,
            pytest.raises(RuntimeError, match="unload failed"),
        ):
            server.stop()

        assert [call.args[0] for call in mock_unload.call_args_list] == expected_handles
        assert server._pending_unloads == [stopped_entries[0]]
        assert _cached(server) == []
        assert stopped_entries[1].handle is None

    def test_stop_defers_unload_until_inference_lock_is_acquired(self, tmp_path, monkeypatch) -> None:
        """Shutdown must not unload a model while another thread still holds its inference lock."""
        monkeypatch.setattr("app.services.inference.inference_server.LOCK_ACQUIRE_TIMEOUT", 0.1)
        server = InferenceServer(data_dir=Path(tmp_path), max_models=2)
        first = _seed_entry(server)
        second = _seed_entry(server)
        first_handle = first.handle
        second_handle = second.handle
        assert first.infer_lock.acquire()
        try:
            with patch("app.services.inference.model_loader.ModelLoader.unload") as mock_unload:
                server.stop()

                assert [call.args[0] for call in mock_unload.call_args_list] == [second_handle]
                assert server._pending_unloads == [first]
                assert _cached(server) == []
                assert first.handle is not None
                assert second.handle is None

                first.infer_lock.release()
                server.stop()

            assert [call.args[0] for call in mock_unload.call_args_list] == [second_handle, first_handle]
            assert server._pending_unloads == []
            assert _cached(server) == []
            assert first.handle is None
        finally:
            if first.infer_lock.locked():
                first.infer_lock.release()

    def test_evict_expired_keeps_stop_deferred_entries_pending_until_inference_lock_is_acquired(
        self, tmp_path, monkeypatch
    ) -> None:
        """Eviction must not unload a stop-deferred model while another thread still holds its inference lock."""
        monkeypatch.setattr("app.services.inference.inference_server.LOCK_ACQUIRE_TIMEOUT", 0.1)
        server = InferenceServer(data_dir=Path(tmp_path), max_models=2)
        first = _seed_entry(server)
        second = _seed_entry(server)
        first_handle = first.handle
        second_handle = second.handle
        assert first.infer_lock.acquire()
        try:
            with patch("app.services.inference.model_loader.ModelLoader.unload") as mock_unload:
                server.stop()

                assert [call.args[0] for call in mock_unload.call_args_list] == [second_handle]
                assert server._pending_unloads == [first]
                assert first.handle is first_handle

                server.evict_expired()

                assert [call.args[0] for call in mock_unload.call_args_list] == [second_handle]
                assert server._pending_unloads == [first]
                assert first.handle is first_handle

                first.infer_lock.release()
                server.evict_expired()

            assert [call.args[0] for call in mock_unload.call_args_list] == [second_handle, first_handle]
            assert server._pending_unloads == []
            assert first.handle is None
        finally:
            if first.infer_lock.locked():
                first.infer_lock.release()

    # --- status ---
    @pytest.mark.parametrize(
        "states, expected_status, expected_models",
        [
            pytest.param([], InferenceStatus.IDLE, 0, id="idle"),
            pytest.param([EntryState.LOADING], InferenceStatus.LOADING, 0, id="loading"),
            pytest.param([EntryState.READY, EntryState.READY], InferenceStatus.ACTIVE, 2, id="active"),
        ],
    )
    def test_get_status_idle_loading_active(self, states, expected_status, expected_models, tmp_path) -> None:
        server = InferenceServer(data_dir=Path(tmp_path), max_models=2)
        for state in states:
            _seed_entry(server, state=state)
        status = server.get_status()
        assert status.status == expected_status
        assert len(status.models) == expected_models
        ready = [entry for entry in _cached(server) if entry.state is EntryState.READY]
        assert list(status.models) == [
            InferenceModel(model_id=entry.model_id, device=entry.device, load_timestamp=ANY) for entry in ready
        ]

    def test_get_status_reports_loaded_model(self, tmp_path) -> None:
        server = InferenceServer(data_dir=Path(tmp_path))
        entry = _seed_entry(server)
        assert server.get_status() == InferenceState(
            status=InferenceStatus.ACTIVE,
            models=(InferenceModel(model_id=entry.model_id, device=CPU, load_timestamp=ANY),),
        )

    # --- inference ---
    @staticmethod
    def _model_with_threshold(current_threshold: float | None) -> Mock:
        """A mocked model that either exposes the given confidence threshold, or none at all."""
        model = Mock(spec=Model)
        model.infer_batch.return_value = []
        model.parameters.return_value = {} if current_threshold is None else {"confidence_threshold": Mock()}
        model.get_param.return_value = current_threshold
        return model

    def _infer(self, server: InferenceServer, entry: ModelCacheEntry, **kwargs) -> dict:
        """Run a batch inference against an already cached entry."""
        return server.infer_batch(
            project_id=entry.project_id,
            model_id=entry.model_id,
            model_variant_id=entry.variant_id,
            device=entry.device,
            labels=kwargs.pop("labels", []),
            inputs=kwargs.pop("inputs", []),
            **kwargs,
        )

    def test_infer_batch_applies_confidence_threshold(self, tmp_path) -> None:
        """A threshold that differs from the one in use is pushed onto the model before inferring."""
        server = InferenceServer(data_dir=Path(tmp_path))
        model = self._model_with_threshold(0.5)
        entry = _seed_entry(server, model=model)
        self._infer(server, entry, confidence_threshold=0.8)
        model.set_param.assert_called_once_with("confidence_threshold", 0.8)
        model.infer_batch.assert_called_once()

    @pytest.mark.parametrize(
        "current_threshold, requested_threshold",
        [
            pytest.param(0.5, 0.5, id="already_applied"),
            pytest.param(0.5, None, id="not_requested"),
            pytest.param(None, 0.8, id="unsupported_by_model"),
        ],
    )
    def test_infer_batch_keeps_confidence_threshold(self, current_threshold, requested_threshold, tmp_path) -> None:
        """The model is left untouched when the threshold is already in use, not requested, or unsupported."""
        server = InferenceServer(data_dir=Path(tmp_path))
        model = self._model_with_threshold(current_threshold)
        entry = _seed_entry(server, model=model)
        self._infer(server, entry, confidence_threshold=requested_threshold)
        model.set_param.assert_not_called()
        model.infer_batch.assert_called_once()

    def test_infer_batch(self, tmp_path) -> None:
        media_id = uuid4()
        model = Mock(spec=Model)
        inference_result = Mock()
        model.infer_batch.return_value = [inference_result]
        label = Mock(spec=Label)
        raw_uint8 = np.full((20, 20, 3), 128, dtype=np.uint8)
        input = BatchInferenceInput(media_id=media_id, frame_index=15, data=raw_uint8)
        annotation = Mock(spec=DatasetItemAnnotation)
        server = InferenceServer(data_dir=Path(tmp_path))
        entry = _seed_entry(server, model=model)
        with patch("app.services.inference.inference_server.convert_prediction") as mock_convert_prediction:
            mock_convert_prediction.return_value = [annotation]
            result = self._infer(server, entry, labels=[label], inputs=[input])
        (passed_batch,) = model.infer_batch.call_args.args
        assert len(passed_batch) == 1
        np.testing.assert_array_equal(passed_batch[0], raw_uint8)
        assert passed_batch[0].dtype == np.uint8
        assert result == {(media_id, 15): [annotation]}

    def test_infer_batch_not_loaded(self, tmp_path) -> None:
        """Inferring with an empty cache loads the requested model instead of failing."""
        server = InferenceServer(data_dir=Path(tmp_path))
        project_id, model_id = uuid4(), uuid4()
        label = Mock(spec=Label)
        with _patched_model_service(tmp_path) as variant_ids, _patched_loader() as (mock_load, _, _handles):
            server.infer_batch(project_id=project_id, model_id=model_id, device=CPU, labels=[label], inputs=[])
        mock_load.assert_called_once_with(
            model_id=model_id,
            variant_id=variant_ids[model_id],
            model_xml_path=tmp_path / "model.xml",
            device=CPU,
        )
        assert [e.model_id for e in _cached(server)] == [model_id]

    def test_infer_batch_busy_raises(self, tmp_path, monkeypatch) -> None:
        """A model already busy with another inference makes the request time out."""
        monkeypatch.setattr("app.services.inference.inference_server.LOCK_ACQUIRE_TIMEOUT", 0.1)
        server = InferenceServer(data_dir=Path(tmp_path))
        entry = _seed_entry(server)
        entry.infer_lock.acquire()
        try:
            with pytest.raises(InferenceBusyError):
                self._infer(server, entry)
        finally:
            entry.infer_lock.release()
        assert entry.refcount == 0
