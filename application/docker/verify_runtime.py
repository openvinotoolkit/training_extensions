# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Build-time smoke test for the native runtime combination used by training jobs.

A training job is a single process that first trains with PyTorch (on CUDA/XPU) and then
evaluates the exported model with OpenVINO on the CPU. Both stacks ship their own copies of
several native libraries; on the XPU image in particular, ``torch==2.14.0+xpu`` pulls the
whole oneAPI runtime into ``<venv>/lib``, which contains a ``libtbb.so.12`` with the *same
SONAME* as the ``libtbb.so.12`` bundled inside the ``openvino`` wheel. Only one library per
SONAME is ever loaded into a process, so whichever stack is imported first decides which TBB
the OpenVINO CPU plugin - which is entirely TBB-scheduled - ends up running on.

This script reproduces that exact import order and exercises the asynchronous CPU inference
path, so an incompatible combination fails the image build instead of sporadically killing a
customer's training job. It also prints where the ambiguous libraries were actually resolved
from, which is the single most useful piece of information when triaging such a failure.
"""

from __future__ import annotations

import sys

# isort: off
# The order of this import block is load-bearing, so it is fenced off from auto-fixers.
# ``torch`` has to enter the process before OpenVINO, exactly as in a training job, because
# that is what decides which ``libtbb.so.12`` the OpenVINO CPU plugin runs on. Alphabetising
# these lines would leave a script that still passes while testing the wrong thing, so
# ``_assert_import_order()`` re-checks the real load order at runtime as a backstop.
import torch
import openvino as ov
import openvino.opset13 as ops
import numpy as np

# isort: on

NUM_REQUESTS = 4
NUM_INFERS = 32
AMBIGUOUS_SONAMES = ("libtbb.so.12", "libtbbbind", "libiomp5.so", "libgomp.so.1")


def _assert_import_order() -> None:
    """Fail loudly if OpenVINO was pulled into the process before torch.

    ``sys.modules`` is insertion ordered, so it doubles as a record of which stack was
    imported first. This guards against an auto-formatter silently reordering the import
    block above and turning this smoke test into a no-op.
    """
    loaded = list(sys.modules)
    if loaded.index("torch") > loaded.index("openvino"):
        msg = (
            "verify_runtime.py imported OpenVINO before torch, so it no longer reproduces "
            "the native library load order of a training job. Restore the import order in "
            "the '# isort: off' block at the top of this file."
        )
        raise RuntimeError(msg)


def _mapped_libraries() -> dict[str, list[str]]:
    """Return, per ambiguous SONAME, the file(s) actually mapped into this process."""
    found: dict[str, set[str]] = {soname: set() for soname in AMBIGUOUS_SONAMES}
    with open("/proc/self/maps") as maps:
        for line in maps:
            path = line.split()[-1]
            if not path.startswith("/"):
                continue
            for soname in AMBIGUOUS_SONAMES:
                if soname in path:
                    found[soname].add(path)
    return {soname: sorted(paths) for soname, paths in found.items()}


def main() -> int:
    """Run the smoke test and report where the ambiguous native libraries resolved from."""
    _assert_import_order()

    param = ops.parameter([1, 3, 16, 16], ov.Type.f32, name="input")
    model = ov.Model([ops.relu(param)], [param], "smoke")

    # THROUGHPUT + an async queue mirrors OVModel/ModelAPI, which is what the evaluation
    # step uses; it is also the configuration that actually stresses the TBB scheduler.
    compiled = ov.Core().compile_model(model, "CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
    queue = ov.AsyncInferQueue(compiled, NUM_REQUESTS)

    results: dict[int, np.ndarray] = {}
    errors: list[BaseException] = []

    def callback(request: ov.InferRequest, idx: int) -> None:
        try:
            results[idx] = request.get_output_tensor(0).data.copy()
        except BaseException as exc:
            errors.append(exc)

    queue.set_callback(callback)
    for i in range(NUM_INFERS):
        queue.start_async({0: np.random.rand(1, 3, 16, 16).astype(np.float32)}, i)
    queue.wait_all()

    if errors:
        raise errors[0]
    if len(results) != NUM_INFERS:
        msg = (
            f"OpenVINO async inference returned {len(results)} of {NUM_INFERS} results"
        )
        raise RuntimeError(msg)

    print(
        f"OpenVINO/torch CPU smoke test OK (torch {torch.__version__}, openvino {ov.get_version()})"
    )
    for soname, paths in _mapped_libraries().items():
        print(f"  {soname}: {', '.join(paths) if paths else '<not loaded>'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
