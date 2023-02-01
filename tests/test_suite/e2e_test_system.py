# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


"""
The functions in the file generate pytest decorators
for integrating with e2e test system and the class DataCollector
that allows pushing information to the dashboard of e2e test system.

If e2e test system is not installed, the generated pytest decorators do nothing,
and the DataCollector class is replaced with a stub that does nothing too.
"""

import functools
import traceback

import pytest


def _generate_e2e_pytest_decorators():
    try:
        from e2e.markers.mark_meta import MarkMeta
    except ImportError:

        def _e2e_pytest_api(func):
            return func

        def _e2e_pytest_performance(func):
            return func

        def _e2e_pytest_component(func):
            return func

        def _e2e_pytest_unit(func):
            return func

        return (
            _e2e_pytest_api,
            _e2e_pytest_performance,
            _e2e_pytest_component,
            _e2e_pytest_unit,
        )

    class Requirements:
        # Dummy requirement
        REQ_DUMMY = "Dummy requirement"

    class OTXComponent(MarkMeta):
        OTX = "otx"

    def _e2e_pytest_api(func):
        @pytest.mark.components(OTXComponent.OTX)
        @pytest.mark.priority_medium
        @pytest.mark.reqids(Requirements.REQ_DUMMY)
        @pytest.mark.api_other
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def _e2e_pytest_performance(func):
        @pytest.mark.components(OTXComponent.OTX)
        @pytest.mark.priority_medium
        @pytest.mark.reqids(Requirements.REQ_DUMMY)
        @pytest.mark.api_performance
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def _e2e_pytest_component(func):
        @pytest.mark.components(OTXComponent.OTX)
        @pytest.mark.priority_medium
        @pytest.mark.reqids(Requirements.REQ_DUMMY)
        @pytest.mark.component
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def _e2e_pytest_unit(func):
        @pytest.mark.components(OTXComponent.OTX)
        @pytest.mark.priority_medium
        @pytest.mark.reqids(Requirements.REQ_DUMMY)
        @pytest.mark.unit
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return (
        _e2e_pytest_api,
        _e2e_pytest_performance,
        _e2e_pytest_component,
        _e2e_pytest_unit,
    )


def _create_class_DataCollector():
    try:
        from e2e.collection_system.systems import TinySystem

        return TinySystem
    except ImportError:

        class _dummy_DataCollector:  # should have the same interface as TinySystem
            def __init__(self, *args, **kwargs):
                pass

            def flush(self):
                pass

            def register_collector(self, *args, **kwargs):
                pass

            def register_exporter(self, *args, **kwargs):
                pass

            def log_final_metric(self, *args, **kwargs):
                pass

            def log_internal_metric(self, *args, **kwargs):
                pass

            def update_metadata(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, typerr, value, tback):
                if typerr is not None:
                    traceback.format_tb(tback)
                    raise typerr(value)
                return True

        return _dummy_DataCollector


(
    e2e_pytest_api,
    e2e_pytest_performance,
    e2e_pytest_component,
    e2e_pytest_unit,
) = _generate_e2e_pytest_decorators()
DataCollector = _create_class_DataCollector()
