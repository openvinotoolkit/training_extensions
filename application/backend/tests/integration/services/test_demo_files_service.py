# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ast
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import cv2
import numpy as np
import pytest
from PIL import Image as PILImage
from sqlalchemy.orm import Session

from app.db.schema import MediaDB, PipelineDB
from app.models import Pipeline, Project
from app.models.media import ImageFormat, Media, VideoFormat
from app.models.model_revision import ModelFormat
from app.services import MediaService
from app.services.demo_files_service import DemoFile, DemoFilesService
from app.services.label_service import LabelService
from app.services.media_service import ImageMetadata


@pytest.fixture
def fxt_demo_files_service(fxt_media_service: MediaService, fxt_label_service: LabelService) -> DemoFilesService:
    return DemoFilesService(media_service=fxt_media_service, label_service=fxt_label_service)


@pytest.fixture
def fxt_project_with_pipeline(
    fxt_db_projects,
    fxt_db_labels,
    fxt_project_service,
    fxt_pipeline_service,
    fxt_db_sources,
    fxt_db_sinks,
    fxt_db_models,
    db_session: Session,
) -> tuple[Project, Pipeline]:
    """Create a Project (mirrors the fixture used in test_media_service.py)."""
    db_project = fxt_db_projects[0]
    db_session.add(db_project)
    db_session.flush()

    db_model = fxt_db_models[0]
    db_model.project_id = db_project.id
    for label in fxt_db_labels:
        label.project_id = db_project.id
    db_session.add_all([db_model, *fxt_db_labels])
    db_session.flush()

    db_pipeline = PipelineDB(project_id=db_project.id)
    db_pipeline.source = fxt_db_sources[0]
    db_pipeline.sink = fxt_db_sinks[0]
    db_pipeline.model_revision = db_model
    db_session.add(db_pipeline)
    db_session.flush()

    return (
        fxt_project_service.get_project_by_id(UUID(db_project.id)),
        fxt_pipeline_service.get_pipeline_by_id(UUID(db_project.id)),
    )


@pytest.fixture
def fxt_project_with_image(
    fxt_project_with_pipeline: tuple[Project, Pipeline],
    fxt_media_service: MediaService,
) -> tuple[Project, Media]:
    """Create a project containing a single real image stored on disk."""
    project, _ = fxt_project_with_pipeline
    image = PILImage.new("RGB", (64, 48), color=(123, 45, 67))
    created = fxt_media_service.create_image(
        ImageMetadata(
            project_id=project.id,
            name="sample",
            image_format=ImageFormat.JPG,
            data=image,
        )
    )
    return project, created


@pytest.fixture
def fxt_project_with_16bit_image(
    request: pytest.FixtureRequest,
    fxt_project_with_pipeline: tuple[Project, Pipeline],
    fxt_media_service: MediaService,
) -> tuple[Project, Media, ImageFormat]:
    """Create a project containing a single real 16-bit image (PNG or TIFF) on disk.

    The image format is supplied indirectly via ``request.param`` so the same fixture
    can be parametrized for both PNG and TIFF.
    """
    project, _ = fxt_project_with_pipeline
    image_format: ImageFormat = request.param
    # A 16-bit (uint16) single-channel image with a value ramp spanning the full range.
    array = np.linspace(0, 65535, num=64 * 48, dtype=np.uint16).reshape(48, 64)
    image = PILImage.fromarray(array)  # mode "I;16"
    created = fxt_media_service.create_image(
        ImageMetadata(
            project_id=project.id,
            name="sample16",
            image_format=image_format,
            data=image,
        )
    )
    return project, created, image_format


@pytest.fixture
def fxt_video_data() -> Callable[[Path, int, int, int], None]:
    """Write a small synthetic AVI video to disk with a known per-frame intensity ramp.

    Each frame is filled with a single gray level equal to ``frame_index`` (clamped to 0..255),
    which lets tests assert that the *middle* frame was the one decoded.
    """

    def _generate(path: Path, frame_count: int = 25, width: int = 64, height: int = 48) -> None:
        fourcc = cv2.VideoWriter.fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(path), fourcc, 25.0, (width, height), isColor=True)
        assert writer.isOpened()
        for i in range(frame_count):
            gray = min(i, 255)
            frame = np.full((height, width, 3), gray, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    return _generate


@pytest.fixture
def fxt_project_with_video_db_row_only(
    fxt_project_with_pipeline: tuple[Project, Pipeline],
    db_session: Session,
) -> Project:
    """Project that has a VIDEO row in the DB but **no** video binary on disk."""
    project, _ = fxt_project_with_pipeline
    db_video = MediaDB(
        type="video",
        name="only_video",
        format="avi",
        size=1024,
        width=640,
        height=480,
        fps=25.0,
        frame_count=100,
    )
    db_video.project_id = str(project.id)
    db_session.add(db_video)
    db_session.flush()
    return project


@pytest.fixture
def fxt_project_with_real_video(
    fxt_project_with_pipeline: tuple[Project, Pipeline],
    fxt_media_service: MediaService,
    fxt_video_data: Callable[..., None],
    tmp_path: Path,
) -> tuple[Project, Media, int]:
    """Create a project containing a single real video (no images) stored on disk.

    Returns the project, the created video media and the total frame count of the video.
    """
    project, _ = fxt_project_with_pipeline
    frame_count = 25
    src = tmp_path / "sample.avi"
    fxt_video_data(src, frame_count=frame_count)
    with open(src, "rb") as data:
        created = fxt_media_service.create_video(
            project_id=project.id,
            name="sample_video",
            video_format=VideoFormat.AVI,
            data=data,
        )
    return project, created, frame_count


class TestDemoFilesServiceIntegration:
    """Integration tests for :class:`DemoFilesService`."""

    def test_non_deployable_format_returns_empty(
        self,
        fxt_demo_files_service: DemoFilesService,
        fxt_project_with_image: tuple[Project, MediaDB],
    ) -> None:
        """PyTorch checkpoints are not deployable -> no demo bundle."""
        project, _ = fxt_project_with_image

        files = fxt_demo_files_service.build_demo_files(project_id=project.id, model_format=ModelFormat.PYTORCH)

        assert files == []

    def test_unknown_project_returns_demo_files_without_image(
        self,
        fxt_demo_files_service: DemoFilesService,
    ) -> None:
        """If no media is available the demo bundle is still produced (no image.jpg)."""
        files = fxt_demo_files_service.build_demo_files(project_id=uuid4(), model_format=ModelFormat.OPENVINO)

        names = [f.name for f in files]
        assert "image.jpg" not in names
        assert names == ["demo.py", "demo_async.py", "utils.py", "pyproject.toml", "README.md"]

    @pytest.mark.parametrize(
        "export_format,expected_format,unexpected_format",
        [(ModelFormat.OPENVINO, "model.xml", "model.onnx"), (ModelFormat.ONNX, "model.onnx", "model.xml")],
        ids=["openvino", "onnx"],
    )
    def test_bundle_contents(
        self,
        fxt_demo_files_service: DemoFilesService,
        fxt_project_with_image: tuple[Project, MediaDB],
        export_format,
        expected_format,
        unexpected_format,
    ) -> None:
        project, _ = fxt_project_with_image

        files = fxt_demo_files_service.build_demo_files(project_id=project.id, model_format=export_format)

        names = [f.name for f in files]
        assert names == ["image.jpg", "demo.py", "demo_async.py", "utils.py", "pyproject.toml", "README.md"]
        # Every entry must be a DemoFile with a non-empty bytes payload.
        for f in files:
            assert isinstance(f, DemoFile)
            assert isinstance(f.data, bytes)
            assert len(f.data) > 0

        by_name = {f.name: f.data for f in files}

        # Demo scripts must reference the OpenVINO IR XML, not the ONNX model.
        utils = by_name["utils.py"].decode("utf-8")
        assert f'MODEL_PATH = HERE / "{expected_format}"' in utils
        assert unexpected_format not in utils

        # Sync vs async hints
        demo = by_name["demo.py"].decode("utf-8")
        demo_async = by_name["demo_async.py"].decode("utf-8")
        assert "synchronous" in demo.lower()
        assert "AsyncPipeline" in demo_async

        # Requirements list the runtime deps used by the demos.
        reqs = by_name["pyproject.toml"].decode("utf-8")
        for pkg in ("openvino", "openvino-model-api", "opencv-python-headless", "numpy", "pillow"):
            assert pkg in reqs

        # README mentions uv and points the user at both demos.
        readme = by_name["README.md"].decode("utf-8")
        assert "uv" in readme.lower()
        assert "demo.py" in readme
        assert "demo_async.py" in readme
        assert "utils.py" in readme
        assert expected_format in readme

    def test_sample_image_matches_stored_binary(
        self,
        fxt_demo_files_service: DemoFilesService,
        fxt_project_with_image: tuple[Project, MediaDB],
        fxt_media_service: MediaService,
    ) -> None:
        """The bundled image.jpg is exactly the bytes of the selected media file."""
        project, media = fxt_project_with_image

        files = fxt_demo_files_service.build_demo_files(project_id=project.id, model_format=ModelFormat.OPENVINO)

        sample = next(f for f in files if f.name == "image.jpg")
        expected_path: Path = fxt_media_service.get_media_binary_path(project_id=project.id, media=media)
        assert sample.data == expected_path.read_bytes()

    @pytest.mark.parametrize(
        "fxt_project_with_16bit_image",
        [ImageFormat.PNG, ImageFormat.TIFF],
        indirect=True,
        ids=["png", "tiff"],
    )
    def test_16bit_image_bundled_verbatim_and_referenced(
        self,
        fxt_demo_files_service: DemoFilesService,
        fxt_project_with_16bit_image: tuple[Project, Media, ImageFormat],
        fxt_media_service: MediaService,
    ) -> None:
        """A non-JPEG (16-bit PNG/TIFF) sample must be bundled verbatim under its original
        extension, and the generated demos/README must reference that exact filename.

        This guards the behavior that 16-bit images are not silently re-encoded to JPEG
        (which would downcast them to 8-bit) and that the sample image extension is
        correctly propagated into the demo scripts and README.
        """
        project, media, image_format = fxt_project_with_16bit_image
        expected_name = f"image.{image_format.value}"

        files = fxt_demo_files_service.build_demo_files(project_id=project.id, model_format=ModelFormat.OPENVINO)

        names = [f.name for f in files]
        # (1) The archive includes image.<ext> (not image.jpg) as the very first entry.
        assert names == [expected_name, "demo.py", "demo_async.py", "utils.py", "pyproject.toml", "README.md"]

        by_name = {f.name: f.data for f in files}

        # The bundled bytes are exactly the stored file (no lossy re-encoding).
        expected_path: Path = fxt_media_service.get_media_binary_path(project_id=project.id, media=media)
        assert by_name[expected_name] == expected_path.read_bytes()

        # ...and the bundled image is genuinely 16-bit (uint16) per channel.
        decoded = cv2.imdecode(np.frombuffer(by_name[expected_name], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert decoded is not None
        assert decoded.dtype == np.uint16

        # (2) The demo scripts reference the actual filename, never the JPEG default.
        script = by_name["utils.py"].decode("utf-8")
        assert f'IMAGE_PATH = HERE / "{expected_name}"' in script
        assert 'HERE / "image.jpg"' not in script

        # ...and so does the README.
        readme = by_name["README.md"].decode("utf-8")
        assert expected_name in readme
        assert "image.jpg" not in readme

    def test_label_colors_are_baked_into_the_demo(
        self,
        fxt_demo_files_service: DemoFilesService,
        fxt_project_with_image: tuple[Project, Media],
    ) -> None:
        """The generated utils.py pins the project label colors so that the demo
        visualizations match the label colors shown in Geti."""
        project, _ = fxt_project_with_image

        files = fxt_demo_files_service.build_demo_files(project_id=project.id, model_format=ModelFormat.OPENVINO)

        script = {f.name: f.data for f in files}["utils.py"].decode("utf-8")
        assert '"cat": "#00FF00",' in script
        assert '"dog": "#FF0000",' in script
        assert "Visualizer(label_colors=LABEL_COLORS)" in script

        # The generated module must be valid Python and expose the mapping.
        module = ast.parse(script)
        assignment = next(
            node
            for node in module.body
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "LABEL_COLORS"
        )
        assert ast.literal_eval(assignment.value) == {"cat": "#00FF00", "dog": "#FF0000"}

    def test_label_colors_default_to_empty_without_label_service(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_image: tuple[Project, MediaDB],
    ) -> None:
        """Without a label service the demo falls back to the Model API default palette."""
        project, _ = fxt_project_with_image
        service = DemoFilesService(media_service=fxt_media_service)

        files = service.build_demo_files(project_id=project.id, model_format=ModelFormat.OPENVINO)

        script = {f.name: f.data for f in files}["utils.py"].decode("utf-8")
        assert "LABEL_COLORS = {}" in script

    def test_video_without_binary_falls_back_gracefully(
        self,
        fxt_demo_files_service: DemoFilesService,
        fxt_project_with_video_db_row_only: Project,
    ) -> None:
        """A project whose only video has no binary on disk must not produce image.jpg
        (and must not raise)."""
        project = fxt_project_with_video_db_row_only

        files = fxt_demo_files_service.build_demo_files(project_id=project.id, model_format=ModelFormat.OPENVINO)

        names = [f.name for f in files]
        assert "image.jpg" not in names
        # The rest of the bundle is still produced.
        assert names == ["demo.py", "demo_async.py", "utils.py", "pyproject.toml", "README.md"]

    def test_video_middle_frame_used_when_no_image_available(
        self,
        fxt_demo_files_service: DemoFilesService,
        fxt_project_with_real_video: tuple[Project, Media, int],
    ) -> None:
        """When the project has no images but does have a video, the middle frame
        of the first video is used as image.jpg.

        The service calls ``MediaService.get_frame_binary`` (which returns a
        ``PIL.Image.Image``) and stores its raw pixel buffer via ``.tobytes()``,
        so the bundled bytes are a flat ``H*W*3`` RGB buffer.
        """
        project, _video, frame_count = fxt_project_with_real_video
        _width, _height = 64, 48  # matches fxt_video_data defaults

        files = fxt_demo_files_service.build_demo_files(project_id=project.id, model_format=ModelFormat.OPENVINO)

        names = [f.name for f in files]
        assert names == ["image.jpg", "demo.py", "demo_async.py", "utils.py", "pyproject.toml", "README.md"]

        sample = next(f for f in files if f.name == "image.jpg")
        assert len(sample.data) > 0

    def test_image_preferred_over_video_for_sample(
        self,
        fxt_demo_files_service: DemoFilesService,
        fxt_project_with_image: tuple[Project, MediaDB],
        fxt_media_service: MediaService,
        fxt_video_data: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """If both images and videos exist, the image must be picked (no video decoding)."""
        project, media = fxt_project_with_image

        # Add a video as well.
        src = tmp_path / "extra.avi"
        fxt_video_data(src, frame_count=10)
        with open(src, "rb") as data:
            fxt_media_service.create_video(
                project_id=project.id,
                name="extra_video",
                video_format=VideoFormat.AVI,
                data=data,
            )

        files = fxt_demo_files_service.build_demo_files(project_id=project.id, model_format=ModelFormat.OPENVINO)

        sample = next(f for f in files if f.name == "image.jpg")
        expected_path: Path = fxt_media_service.get_media_binary_path(project_id=project.id, media=media)
        assert sample.data == expected_path.read_bytes()

    def test_missing_binary_file_is_skipped(
        self,
        fxt_demo_files_service: DemoFilesService,
        fxt_project_with_image: tuple[Project, MediaDB],
        fxt_media_service: MediaService,
    ) -> None:
        """If the picked image's binary file has been removed, image.jpg is omitted."""
        project, media = fxt_project_with_image
        binary_path: Path = fxt_media_service.get_media_binary_path(project_id=project.id, media=media)
        binary_path.unlink()
        assert not binary_path.exists()

        files = fxt_demo_files_service.build_demo_files(project_id=project.id, model_format=ModelFormat.OPENVINO)

        assert "image.jpg" not in [f.name for f in files]

    def test_media_service_failure_does_not_break_bundle(
        self,
        fxt_demo_files_service: DemoFilesService,
        fxt_project_with_image: tuple[Project, MediaDB],
    ) -> None:
        """Errors raised by MediaService.list_media are swallowed: bundle still built."""
        project, _ = fxt_project_with_image

        with patch.object(
            fxt_demo_files_service._media_service,
            "list_media",
            side_effect=RuntimeError("whoops"),
        ):
            files = fxt_demo_files_service.build_demo_files(project_id=project.id, model_format=ModelFormat.OPENVINO)

        names = [f.name for f in files]
        assert "image.jpg" not in names
        assert names == ["demo.py", "demo_async.py", "utils.py", "pyproject.toml", "README.md"]
