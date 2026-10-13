# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from loguru import logger

from app.models.media import Image, MediaType, Video
from app.models.model_revision import ModelFormat

if TYPE_CHECKING:
    from app.models.model_manifest import License
    from app.services import MediaService
    from app.services.label_service import LabelService
    from app.supported_models.attributions import ModelAttribution


# Filename of the model weights inside the archive, depending on the format.
_MODEL_FILENAME_BY_FORMAT: dict[ModelFormat, str] = {
    ModelFormat.OPENVINO: "model.xml",
    ModelFormat.ONNX: "model.onnx",
}

_DEFAULT_IMAGE_FILENAME = "image.jpg"

# Files that must be redistributed together with the model weights to comply with the model license,
# keyed by the license name declared in the model manifests or in the timm catalog (`license.name`).
# Each entry maps the filename inside the archive to a resource under `app/supported_models/licenses/`.
# Licenses that are not listed here (CC BY-NC 4.0, CC BY-NC-SA 4.0) explicitly allow to satisfy their
# terms with a link to the license text, which is always included in the README.
_LICENSE_FILES_BY_NAME: dict[str, dict[str, str]] = {
    "AGPL-3.0": {"LICENSE": "AGPL-3.0.txt"},
    "Apache 2.0": {"LICENSE": "Apache-2.0.txt"},
    "Apple ML MobileOne License": {"LICENSE": "Apple-MobileOne.txt"},
    "Apple-AMLR": {"LICENSE": "Apple-AMLR.txt"},
    "Apple-ASCL": {"LICENSE": "Apple-ASCL.txt"},
    "BSD-3-Clause": {"LICENSE": "BSD-3-Clause.txt"},
    "DEIMv2 License": {"LICENSE": "DEIMv2.txt"},
    "DINOv3 License": {"LICENSE": "DINOv3.txt"},
    "EdgeCrafter": {"LICENSE": "EdgeCrafter.txt"},
    "Fair Noncommercial Research License": {"LICENSE": "FAIR-Noncommercial-Research.txt"},
    # Gemma Terms of Use, Section 3.1: distributions must also be accompanied by a "Notice" text file.
    "GEMMA": {"LICENSE": "Gemma.txt", "NOTICE": "Gemma-NOTICE.txt"},
    "MIT": {"LICENSE": "MIT.txt"},
}

# Licenses whose terms require the copyright notice and/or the attribution of the original work to be retained in
# redistributions. An attribution must be registered for every model under one of these licenses, see
# `app/supported_models/licenses/attributions.yaml`.
LICENSES_REQUIRING_ATTRIBUTION: frozenset[str] = frozenset({"MIT", "BSD-3-Clause", "CC BY-NC 4.0", "CC BY-NC-SA 4.0"})

# Placeholder, in the license file templates (e.g. MIT, BSD-3-Clause), for the copyright notice of the original work.
_COPYRIGHT_PLACEHOLDER = "<copyright notice>"

# Fallback copyright notice, used only if no attribution is registered for a model (should not happen).
_GENERIC_COPYRIGHT_NOTICE = "Copyright (c) the authors of the original model and its pretrained weights"

_FORMAT_DISPLAY_NAMES: dict[ModelFormat, str] = {
    ModelFormat.OPENVINO: "OpenVINO IR",
    ModelFormat.ONNX: "ONNX",
}


def _copyright_notice(attribution: ModelAttribution | None) -> str:
    """Copyright notice of the original work, to be retained in the license file."""
    if attribution is None:
        return _GENERIC_COPYRIGHT_NOTICE
    if attribution.copyright is not None:
        return attribution.copyright
    return f"Copyright (c) {attribution.creators}"


def load_license_files(license_name: str, attribution: ModelAttribution | None = None) -> dict[str, bytes]:
    """Load the license files that must be redistributed together with the model.

    License files that contain a copyright notice placeholder (e.g. MIT, BSD-3-Clause) are completed with
    the copyright notice of the original work, taken from the model attribution.

    Args:
        license_name: Name of the license, as declared in the model manifest (`license.name`).
        attribution: Attribution of the original work the model is derived from, if any.

    Returns:
        A mapping from filename inside the archive to file content. Empty if the license does
        not require any file to be bundled (a link to the license text is sufficient).
    """
    licenses_dir = resources.files("app.supported_models").joinpath("licenses")
    files: dict[str, bytes] = {}
    for filename, resource_name in _LICENSE_FILES_BY_NAME.get(license_name, {}).items():
        content = licenses_dir.joinpath(resource_name).read_text(encoding="utf-8")
        content = content.replace(_COPYRIGHT_PLACEHOLDER, _copyright_notice(attribution))
        files[filename] = content.encode("utf-8")
    return files


def _format_license_files_note(filenames: list[str]) -> str:
    """Render the README sentence pointing at the license files bundled in the archive."""
    quoted = [f"`{name}`" for name in filenames]
    if len(quoted) == 1:
        listed = f"{quoted[0]} file"
    else:
        listed = f"{', '.join(quoted[:-1])} and {quoted[-1]} files"
    return _README_LICENSE_FILES_NOTE.format(license_files=listed)


def _format_licensing_section(
    license: License,
    attribution: ModelAttribution | None,
    license_filenames: list[str],
    model_format: ModelFormat,
) -> str:
    """Render the "Licensing" section of the README, including attribution and modification notices."""
    section = _README_LICENSING.format(license_name=license.name, license_url=license.url)
    if license_filenames:
        section += _format_license_files_note(license_filenames)
    if attribution is not None:
        section += _README_ATTRIBUTION.format(
            name=attribution.name, creators=attribution.creators, source=attribution.source
        )
        if attribution.copyright is not None:
            section += _README_COPYRIGHT_NOTICE.format(copyright=attribution.copyright)
        else:
            section += _README_NO_COPYRIGHT_NOTICE
    section += _README_MODIFICATIONS.format(format_name=_FORMAT_DISPLAY_NAMES[model_format])
    return section


@dataclass(frozen=True)
class DemoFile:
    """A single auxiliary file to be added to the downloaded model archive."""

    name: str
    data: bytes


class DemoFilesService:
    def __init__(self, media_service: MediaService, label_service: LabelService | None = None):
        self._media_service: MediaService = media_service
        self._label_service: LabelService | None = label_service

    def build_demo_files(
        self,
        project_id: UUID,
        model_format: ModelFormat,
        *,
        license: License | None = None,
        attribution: ModelAttribution | None = None,
    ) -> list[DemoFile]:
        """Build the auxiliary deployment files to bundle in with the model archive.

        Only OpenVINO (.xml/.bin) and ONNX (.onnx) variants are supported - for any other
        format, an empty list is returned (PyTorch checkpoints are typically used for
        fine-tuning rather than direct deployment).

        Args:
            project_id: Project that owns the model.
            model_format: Format of the model variant being downloaded.
            license: The license of the model, as declared in its manifest. When provided, a
                "Licensing" section referencing the license is appended to the README, and for
                licenses that require it (e.g. AGPL-3.0), a copy of the full license text is
                included as a LICENSE file (plus any other required file, e.g. NOTICE).
            attribution: Attribution of the original work the model is derived from. When provided,
                the creators, source and copyright notice of the original work are added to the README
                and, for licenses such as MIT and BSD-3-Clause, to the LICENSE file.

        Returns:
            The list of files (name + bytes) to add to the zip archive.
        """
        if model_format not in _MODEL_FILENAME_BY_FORMAT:
            return []

        model_filename = _MODEL_FILENAME_BY_FORMAT[model_format]
        files: list[DemoFile] = []

        # Preserve the original sample image format/extension: 16-bit images (PNG/TIFF)
        # must not be re-encoded to JPEG, which would silently downcast them to 8-bit.
        sample = self._pick_sample_image(project_id=project_id)
        if sample is not None:
            image_filename, image_bytes = sample
            files.append(DemoFile(name=image_filename, data=image_bytes))
        else:
            image_filename = _DEFAULT_IMAGE_FILENAME
            logger.warning(
                "No suitable sample image found in project {}; the model archive will not include a sample image.",
                project_id,
            )

        files.append(DemoFile(name="demo.py", data=_DEMO.encode("utf-8")))
        files.append(DemoFile(name="demo_async.py", data=_DEMO_ASYNC.encode("utf-8")))
        files.append(
            DemoFile(
                name="utils.py",
                data=_UTILS.format(
                    model_filename=model_filename,
                    image_filename=image_filename,
                    label_colors=self._format_label_colors(project_id=project_id),
                ).encode("utf-8"),
            )
        )
        files.append(DemoFile(name="pyproject.toml", data=_PY_PROJECT.encode("utf-8")))

        readme = _README.format(model_filename=model_filename, image_filename=image_filename)
        if license is not None:
            if attribution is None and license.name in LICENSES_REQUIRING_ATTRIBUTION:
                logger.warning(
                    "No attribution registered for a model under the '{}' license (project {}); "
                    "a generic copyright notice is used.",
                    license.name,
                    project_id,
                )
            license_files = load_license_files(license.name, attribution=attribution)
            files.extend(DemoFile(name=name, data=data) for name, data in license_files.items())
            readme += _format_licensing_section(
                license=license,
                attribution=attribution,
                license_filenames=list(license_files),
                model_format=model_format,
            )

        files.append(DemoFile(name="README.md", data=readme.encode("utf-8")))
        return files

    def _format_label_colors(self, project_id: UUID) -> str:
        """Render the project label colours as a Python dict literal for the demo scripts.

        The exported model predicts the project label names, so mapping those names to the
        project label colours makes the demo visualisations match the colours shown in Geti.
        Returns "{}" when the labels cannot be resolved.
        """
        if self._label_service is None:
            return "{}"
        try:
            labels = self._label_service.list_all(project_id=project_id)
        except Exception:
            logger.exception("Could not resolve label colors for project {}; using default colors.", project_id)
            return "{}"

        entries = {label.name: label.color for label in labels if label.name and label.color}
        if not entries:
            return "{}"
        lines = "\n".join(f"    {json.dumps(name)}: {json.dumps(color)}," for name, color in sorted(entries.items()))
        return "{\n" + lines + "\n}"

    def _pick_sample_image(self, project_id: UUID) -> tuple[str, bytes] | None:  # noqa: C901
        """Pick a sample image from the project's dataset.

        Returns a (filename, data) tuple where filename preserves the original
        image extension (so 16-bit PNG/TIFF images are bundled verbatim, without any
        lossy re-encoding) and data is the raw file content.

        Resolution order:
          1. The first available plain image in the project (kept in its original format).
          2. If no images exist but at least one video does, the **middle frame**
             of the first video is decoded and encoded as JPEG (named image.jpg).
          3. Otherwise, returns None.
        """
        try:
            from app.services.media_service import MediaFilters  # local import to avoid cycle at module load
        except Exception:  # pragma: no cover - defensive
            return None

        # 1) Try a real image first.
        try:
            image_media = self._media_service.list_media(
                project_id=project_id,
                filters=MediaFilters(limit=1, offset=0),
                exclude_types=[MediaType.VIDEO, MediaType.VIDEO_FRAME],
            )
        except Exception:
            logger.exception("Failed to list media to pick a sample image for project {}", project_id)
            image_media = []

        for media in image_media:
            try:
                path: Path = self._media_service.get_media_binary_path(project_id=project_id, media=media)
                if path.exists():
                    extension = media.format.value if isinstance(media, Image) else path.suffix.lstrip(".") or "jpg"
                    return f"image.{extension}", path.read_bytes()
            except Exception:
                logger.exception("Failed to read sample image {} for project {}", media.id, project_id)
                continue

        # 2) Fall back to extracting the middle frame of the first available video.
        try:
            video_media = self._media_service.list_media(
                project_id=project_id,
                filters=MediaFilters(limit=1, offset=0),
                exclude_types=[MediaType.IMAGE, MediaType.VIDEO_FRAME],
            )
        except Exception:
            logger.exception("Failed to list videos to extract a sample frame for project {}", project_id)
            return None

        for media in video_media:
            if not isinstance(media, Video) or media.frame_count <= 0:
                continue
            try:
                frame_index = media.frame_count // 2
                video_path = self._media_service.get_media_binary_path(project_id=project_id, media=media)
                video_frame = self._encode_video_frame_as_jpeg(video_path=video_path, frame_index=frame_index)
                if video_frame is None:
                    continue
                logger.info(
                    "No image found in project {}; extracting frame {} from video {} as sample image.",
                    project_id,
                    frame_index,
                    media.id,
                )
                return _DEFAULT_IMAGE_FILENAME, video_frame
            except Exception:
                logger.exception("Failed to extract a sample frame from video {} (project {})", media.id, project_id)
                continue

        return None

    def _encode_video_frame_as_jpeg(self, video_path: Path, frame_index: int) -> bytes | None:
        """Decode a single video frame and return it encoded as JPEG bytes."""
        try:
            from PIL import Image as PILImage  # local import to keep module import cheap
        except Exception:  # pragma: no cover - defensive
            return None

        video_service = self._media_service._get_video_service()
        frame_rgb = video_service.extract_video_frame(video_path=video_path, frame_index=frame_index)
        if frame_rgb is None:
            return None
        image = PILImage.fromarray(frame_rgb)
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        return buffer.getvalue()


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_DEMO = '''\
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Synchronous inference demo for a model exported from Geti.

Loads a sample image, runs inference with OpenVINO Model API, then saves an
output image with the overlaid predictions to result.jpg.
"""
from __future__ import annotations

from utils import load_image, load_model, visualise_result


def main() -> None:
    model = load_model()
    image = load_image()

    print("Running synchronous inference...")
    result = model(image)
    print("Predictions:")
    print(result)

    visualise_result(image, result)


if __name__ == "__main__":
    main()

'''


_DEMO_ASYNC = '''\
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Asynchronous inference demo for a model exported from Geti.

Uses OpenVINO Model API's AsyncPipeline to submit the sample image
asynchronously and retrieve the prediction once it is ready. The resulting
image with the overlaid predictions is saved to result.jpg.
"""

from __future__ import annotations

from model_api.pipelines import AsyncPipeline
from utils import load_image, load_model, visualise_result


def main() -> None:
    model = load_model()
    image = load_image()

    print("Running asynchronous inference...")
    pipeline = AsyncPipeline(model)
    pipeline.submit_data(image, id=0)
    pipeline.await_all()
    result, _meta = pipeline.get_result(0)
    print("Predictions:")
    print(result)

    visualise_result(image, result)


if __name__ == "__main__":
    main()

'''

_UTILS = """\
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from model_api.models import Model
from model_api.visualizer import Visualizer

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "{model_filename}"
IMAGE_PATH = HERE / "{image_filename}"
OUTPUT_PATH = HERE / "result.jpg"

# Colors of the labels as defined in the Geti project, so that the rendered predictions
# match the label colors shown in the Geti UI. Edit or clear this mapping to use the
# Model API default color palette instead.
LABEL_COLORS = {label_colors}

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {{MODEL_PATH}}")
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Sample image not found:{{IMAGE_PATH}}")


def load_model() -> Model:
    print(f"Loading model from {{MODEL_PATH}}...")
    return Model.create_model(str(MODEL_PATH))


def load_image() -> cv2.Mat:
    print(f"Loading image from {{IMAGE_PATH}}...")
    # IMREAD_UNCHANGED preserves the original bit depth (e.g. 16-bit PNG/TIFF images).
    image_raw = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_UNCHANGED)
    if image_raw is None:
        raise RuntimeError(f"Failed to decode image: {{IMAGE_PATH}}")

    # Add explicit channel dimension for 2D grayscale: (H, W) -> (H, W, 1)
    if image_raw.ndim == 2:
        image_raw = image_raw[..., np.newaxis]

    # Convert BGR to RGB for standard 3-channel images
    if image_raw.ndim == 3 and image_raw.shape[2] == 3:
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

    return image_raw


def visualise_result(image, result) -> None:
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    visualizer = Visualizer(label_colors=LABEL_COLORS)
    visualizer.show(image, result)

    display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = visualizer.render(display_image, result)
    cv2.imwrite(str(OUTPUT_PATH), output)
    print(f"Saved annotated result to {{OUTPUT_PATH}}")

"""


_PY_PROJECT = """\
[project]
name = "Geti-demo"
version = "1.0.0"
description = "A minimal demo showcasing how to run inference with a model exported from Geti using OpenVINO Model API."
requires-python = ">=3.13,<3.14"

dependencies = [
    "openvino~=2026.3.0",
    "openvino-model-api[onnx] @ git+https://github.com/open-edge-platform/model_api@master#subdirectory=model_api",
    "opencv-python-headless~=4.13.0",
    "numpy>=2.0",
    "pillow~=12.0",
]
"""


_README = """\
# Geti exported model

This archive contains a model exported from Geti, along with a couple of
ready-to-run inference demos.

## Contents

| File | Description |
| ---- | ----------- |
| `{model_filename}` (+ `model.bin` for OpenVINO IR) | The exported model weights. |
| `{image_filename}` (optional) | Sample input image from the project's dataset, kept in its original format. |
| `demo.py` | Minimal **synchronous** inference example. |
| `demo_async.py` | Minimal **asynchronous** inference example. |
| `utils.py` | Shared utility functions for loading the model/image and visualising the results. |
| `pyproject.toml` | Python dependencies required by the demos. |
| `README.md` | This file. |

The image may be omitted if no image is available. If `{image_filename}` is missing, copy any image into this directory 
and name it `{image_filename}` (or edit the demos to point to a different file).
 
## Setup

The recommended way to set up a clean environment is with
[`uv`](https://docs.astral.sh/uv/) - a fast Python package manager.

### Option 1 - one-shot with `uv`

This will create and activate your venv, then run the script immediately.

```bash
# From the directory where this README lives
uv run demo.py
uv run demo_async.py
```

`uv run` will transparently create a virtual environment, install the
dependencies, and execute the script. You will not remain in the virtual 
environment after the script executes.

### Option 2 - create a persistent virtual environment, then activate it

```bash
# Create and activate a virtual environment (Python 3.10+)
uv sync
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\\Scripts\\activate
```

## Running the demos

Once the environment is ready and activated, simply run:

```bash
# Synchronous inference - writes the annotated result to result.jpg
python demo.py

# Asynchronous inference - writes the annotated result to result_async.jpg
python demo_async.py
```

Both scripts load `{image_filename}`, run inference on it with OpenVINO Model API and
save an output image with the predicted bounding boxes / labels / masks
overlaid on top.

## Notes

* The demos default to running on CPU. To run on a different device (e.g. an
  Intel GPU), edit the scripts and pass device="GPU" to
  Model.create_model.
* For ONNX models, OpenVINO Model API reads the `.onnx` file directly - no
  additional conversion is required.
* These demos are intentionally minimal. For production deployment, refer to
  the [OpenVINO Model API documentation](https://github.com/open-edge-platform/model_api).
"""


_README_LICENSING = """
## Licensing

This model is distributed under the "{license_name}" license, the full text is available [here]({license_url}).
"""

_README_LICENSE_FILES_NOTE = """
A copy of the license is included in the {license_files} of this archive.
"""

_README_ATTRIBUTION = """
### Attribution

This model is derived from the pretrained model "{name}", created by {creators}, available at <{source}>.
"""

_README_COPYRIGHT_NOTICE = """
Copyright notice of the original work:

```text
{copyright}
```
"""

_README_NO_COPYRIGHT_NOTICE = """
The original work does not provide a copyright notice.
"""

_README_MODIFICATIONS = """
### Modifications

The model weights in this archive are a modified version of the original pretrained weights: they were
fine-tuned with Geti on a custom dataset and exported to the {format_name} format.

The original work is provided by its licensors "as is", without warranties of any kind; refer to the license
for the full disclaimer of warranties and limitation of liability.
"""
