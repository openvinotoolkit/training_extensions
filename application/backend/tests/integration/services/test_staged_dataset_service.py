# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import cv2
import numpy as np
import pytest
from datumaro.experimental import Dataset, LazyImage, LazyVideoFrame, MediaInfo, export_dataset
from datumaro.experimental.categories import Categories, LabelCategories
from datumaro.experimental.data_formats.coco.sample import CocoCategories, CocoSample
from datumaro.experimental.export_import import ExportMode
from datumaro.experimental.fields import ImageInfo, Subset

from app.datumaro_converter import (
    MulticlassClassificationImportExportSample,
    MultilabelClassificationImportExportSample,
)
from app.models import AnnotationType, DatasetFormat
from app.models.dataset import DatasetMetadata
from app.services import StagedDatasetService


def _create_dummy_video(path: Path, num_frames: int = 20, width: int = 64, height: int = 64, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for _ in range(num_frames):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _stage_dataset_archive(root: Path, file_name: str, content: bytes = b"data") -> tuple[UUID, Path]:
    dataset_id = uuid4()
    ds_dir = root / str(dataset_id)
    ds_dir.mkdir(parents=True)
    archive_path = ds_dir / file_name
    archive_path.write_bytes(content)
    return dataset_id, archive_path


def _stage_multilabel_dataset(root: Path) -> tuple[UUID, Path]:
    dataset_id = uuid4()
    parent_dir = root / str(dataset_id)
    parent_dir.mkdir(parents=True)
    ds_dir = parent_dir / "dataset"
    categories: dict[str, Categories] = {"label": LabelCategories(labels=("cat", "dog", "bird"))}
    dataset = Dataset(MultilabelClassificationImportExportSample, categories=categories)
    dataset.append(
        MultilabelClassificationImportExportSample(
            id=str(uuid4()),
            media=LazyImage(ds_dir / "images/image1.jpg"),
            media_info=MediaInfo(width=200, height=200),
            label=np.array([0]),
            subset=Subset.TRAINING,
            confidence=np.array([0.9]),
            user_reviewed=True,
        )
    )
    dataset.append(
        MultilabelClassificationImportExportSample(
            id=str(uuid4()),
            media=LazyImage(ds_dir / "images/image2.jpg"),
            media_info=MediaInfo(width=200, height=200),
            label=np.array([1]),
            subset=Subset.TRAINING,
            confidence=np.array([0.8]),
            user_reviewed=True,
        )
    )
    dataset.append(
        MultilabelClassificationImportExportSample(
            id=str(uuid4()),
            media=LazyImage(ds_dir / "images/image2.jpg"),
            media_info=MediaInfo(width=200, height=200),
            label=np.array([]),
            subset=Subset.TRAINING,
            confidence=None,
            user_reviewed=False,
        )
    )
    video_path = root / "videos" / "video1.mp4"
    _create_dummy_video(video_path, num_frames=20)
    dataset.append(
        MultilabelClassificationImportExportSample(
            id=str(uuid4()),
            media=LazyVideoFrame(video_path=video_path, frame_index=10),
            media_info=MediaInfo(width=200, height=200),
            label=np.array([]),
            subset=Subset.TRAINING,
            confidence=np.array([0.9]),
            user_reviewed=True,
        )
    )
    export_dataset(dataset, ds_dir, export_media=ExportMode.SKIP)
    return dataset_id, ds_dir


def _stage_coco_dataset(root: Path) -> tuple[UUID, Path]:
    dataset_id = uuid4()
    parent_dir = root / str(dataset_id)
    parent_dir.mkdir(parents=True)
    ds_dir = parent_dir / "dataset"
    dataset = Dataset(CocoSample, categories={"labels": CocoCategories(labels=("cat", "dog", "bird"))})
    dataset.append(
        CocoSample(
            image=LazyImage(ds_dir / "images/image1.jpg"),
            image_info=ImageInfo(width=200, height=200),
            areas=np.array([1.0], dtype=np.float32),
            iscrowd=np.array([0], dtype=np.int32),
            subset=Subset.TRAINING,
            labels=None,
            polygons=np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]], dtype=np.float32),
            bboxes=np.array([[10, 10, 10, 10]], dtype=np.float32),
            image_id=1,
            caption_group_ids=None,
            captions=None,
            keypoints=None,
        )
    )
    export_dataset(dataset=dataset, output_path=ds_dir, export_media=ExportMode.SKIP)
    return dataset_id, ds_dir


def _stage_multiclass_dataset(root: Path) -> tuple[UUID, Path]:
    dataset_id = uuid4()
    parent_dir = root / str(dataset_id)
    parent_dir.mkdir(parents=True)
    ds_dir = parent_dir / "dataset"
    categories: dict[str, Categories] = {"label": LabelCategories(labels=("cat", "dog", "bird"))}
    dataset = Dataset(MulticlassClassificationImportExportSample, categories=categories)
    dataset.append(
        MulticlassClassificationImportExportSample(
            id=str(uuid4()),
            media=LazyImage(ds_dir / "images/image1.jpg"),
            media_info=MediaInfo(width=200, height=200),
            label=None,
            subset=Subset.UNASSIGNED,
            confidence=None,
            user_reviewed=False,
        )
    )
    dataset.append(
        MulticlassClassificationImportExportSample(
            id=str(uuid4()),
            media=LazyImage(ds_dir / "images/image2.jpg"),
            media_info=MediaInfo(width=200, height=200),
            label=0,
            subset=Subset.UNASSIGNED,
            confidence=0.8,
            user_reviewed=True,
        )
    )
    export_dataset(dataset=dataset, output_path=ds_dir, export_media=ExportMode.SKIP)
    return dataset_id, ds_dir


@pytest.fixture()
def fxt_staged_dataset_service(tmp_path: Path) -> StagedDatasetService:
    return StagedDatasetService(staged_datasets_dir=tmp_path)


class TestStagedDatasetServiceIntegration:
    @pytest.mark.asyncio
    async def test_upload_from_path_moves_file(self, tmp_path: Path) -> None:
        service = StagedDatasetService(staged_datasets_dir=tmp_path / "staged")
        source_path = tmp_path / "uploads" / "upload.part"
        source_path.parent.mkdir()
        source_path.write_bytes(b"zip-bytes")

        staged = await service.upload_from_path(filename="dataset.zip", source_path=source_path)

        assert not source_path.exists()
        target_path = tmp_path / "staged" / str(staged.id) / "dataset.zip"
        assert staged.filename == str(target_path)
        assert target_path.read_bytes() == b"zip-bytes"
        assert staged.size == len(b"zip-bytes")
        assert staged.compressed

    @pytest.mark.asyncio
    async def test_upload_from_path_missing_file_leaves_nothing_behind(self, tmp_path: Path) -> None:
        service = StagedDatasetService(staged_datasets_dir=tmp_path / "staged")

        with pytest.raises(FileNotFoundError):
            await service.upload_from_path(filename="dataset.zip", source_path=tmp_path / "missing.part")

        assert not any((tmp_path / "staged").iterdir())

    @pytest.mark.asyncio
    async def test_upload_integration_writes_file_and_returns_metadata(
        self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService
    ):
        filename = "dataset.zip"
        content = b"hello world!"
        file_obj = BytesIO(content)

        staged_dataset = await fxt_staged_dataset_service.upload(filename=filename, file_obj=file_obj)

        assert staged_dataset.id is not None
        assert staged_dataset.compressed is True
        assert staged_dataset.format == DatasetFormat.UNKNOWN
        assert staged_dataset.size == len(content)

        stored_path = Path(staged_dataset.filename)
        assert stored_path.name == "dataset.zip"
        assert stored_path.is_file()
        assert stored_path.parent.parent == tmp_path
        assert stored_path.read_bytes() == content

    def test_list_all_single_zip_dataset(self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService):
        dataset_id, archive_path = _stage_dataset_archive(tmp_path, "my_coco_dataset.zip", b"123456")

        datasets = fxt_staged_dataset_service.list_all()

        assert len(datasets) == 1
        ds = datasets[0]
        assert ds.id == dataset_id
        assert ds.filename == str(archive_path)
        assert ds.size == archive_path.stat().st_size
        assert ds.compressed is True
        assert ds.format == DatasetFormat.COCO

    def test_list_all_multiple_datasets_and_ignores_non_uuid(
        self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService
    ):
        coco_id, coco_path = _stage_dataset_archive(tmp_path, "train_coco.zip", b"coco-bytes")
        voc_id, voc_path = _stage_dataset_archive(tmp_path, "some_voc.zip", b"voc-bytes")
        geti_id, geti_path = _stage_multilabel_dataset(tmp_path)

        # non-UUID dir
        bad_dir = tmp_path / "not-a-uuid"
        bad_dir.mkdir()
        (bad_dir / "ignored.zip").write_bytes(b"ignored")

        datasets = fxt_staged_dataset_service.list_all()

        # Only 3 valid UUID datasets
        assert {d.id for d in datasets} == {coco_id, voc_id, geti_id}

        coco_ds = next(d for d in datasets if d.id == coco_id)
        voc_ds = next(d for d in datasets if d.id == voc_id)
        geti_ds = next(d for d in datasets if d.id == geti_id)

        assert coco_ds.compressed
        assert coco_ds.format == DatasetFormat.COCO
        assert coco_ds.filename == str(coco_path)
        assert not coco_ds.metadata

        assert voc_ds.compressed
        assert voc_ds.format == DatasetFormat.VOC
        assert voc_ds.filename == str(voc_path)
        assert not voc_ds.metadata

        assert not geti_ds.compressed
        assert geti_ds.format == DatasetFormat.GETI
        assert geti_ds.filename == str(geti_path)
        assert geti_ds.metadata == DatasetMetadata(
            num_images=3,
            num_frames=20,
            num_videos=1,
            annotation_type=AnnotationType.LABEL,
            num_annotations=3,
            num_annotated_images=2,
            num_annotated_frames=1,
            labels=["bird", "cat", "dog"],
        )

    def test_list_all_ignores_empty_uuid_dirs(self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService):
        empty_id = uuid4()
        (tmp_path / str(empty_id)).mkdir()

        _, valid_path = _stage_dataset_archive(tmp_path, "dataset.zip", b"xxx")

        datasets = fxt_staged_dataset_service.list_all()

        assert len(datasets) == 1
        assert datasets[0].filename == str(valid_path)
        assert datasets[0].format == DatasetFormat.UNKNOWN

    def test_list_all_ignores_non_zip(self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService):
        non_zip_id = uuid4()
        non_zip_dir = tmp_path / str(non_zip_id)
        non_zip_dir.mkdir()
        (non_zip_dir / "file.txt").write_text("not a zip")

        datasets = fxt_staged_dataset_service.list_all()

        assert len(datasets) == 0

    @pytest.mark.parametrize("prefix, data_format", [("coco", DatasetFormat.COCO), ("yolo", DatasetFormat.YOLO)])
    def test_find_by_id_returns_dataset_when_present(
        self, prefix: str, data_format: DatasetFormat, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService
    ):
        dataset_id, archive_path = _stage_dataset_archive(tmp_path, f"my_{prefix}_dataset.zip", b"123456")

        result = fxt_staged_dataset_service.find_by_id(dataset_id)

        assert result is not None
        assert result.id == dataset_id
        assert result.filename == str(archive_path)
        assert result.size == 6
        assert result.compressed is True
        assert result.format == data_format

    def test_find_by_id_returns_geti_dataset_when_present(
        self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService
    ):
        dataset_id, dataset_path = _stage_multilabel_dataset(tmp_path)

        result = fxt_staged_dataset_service.find_by_id(dataset_id)

        assert result is not None
        assert result.id == dataset_id
        assert result.filename == str(dataset_path)
        assert result.size > 6 * 1024  # variable based on dataset, but should be >6KB for the sample dataset
        assert result.compressed is False
        assert result.format == DatasetFormat.GETI
        assert result.metadata == DatasetMetadata(
            num_images=3,
            num_frames=20,
            num_videos=1,
            annotation_type=AnnotationType.LABEL,
            num_annotations=3,
            num_annotated_images=2,
            num_annotated_frames=1,
            labels=["bird", "cat", "dog"],
        )

    def test_find_by_id_no_annotation_type_if_multiple_values_in_sample(
        self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService
    ):
        """
        Tests that annotation_type cannot be recommended for datasets with samples that have multiple annotation field
        values set (e.g. both polygons and bboxes).
        """
        dataset_id, dataset_path = _stage_coco_dataset(tmp_path)

        result = fxt_staged_dataset_service.find_by_id(dataset_id)

        assert result is not None
        assert result.id == dataset_id
        assert result.filename == str(dataset_path)
        assert result.compressed is False
        assert result.format == DatasetFormat.GETI
        assert result.metadata == DatasetMetadata(
            num_images=1,
            num_frames=0,
            num_videos=0,
            annotation_type=AnnotationType.UNKNOWN,
            num_annotations=1,
            num_annotated_images=1,
            num_annotated_frames=0,
            labels=["bird", "cat", "dog"],
        )

    def test_find_by_id_annotation_type_if_some_samples_unannotated(
        self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService
    ):
        """Tests that annotation_type will be set when some samples are unannotated."""
        dataset_id, dataset_path = _stage_multiclass_dataset(tmp_path)

        result = fxt_staged_dataset_service.find_by_id(dataset_id)

        assert result is not None
        assert result.id == dataset_id
        assert result.filename == str(dataset_path)
        assert result.compressed is False
        assert result.format == DatasetFormat.GETI
        assert result.metadata == DatasetMetadata(
            num_images=2,
            num_frames=0,
            num_videos=0,
            annotation_type=AnnotationType.LABEL,
            num_annotations=1,
            num_annotated_images=1,
            num_annotated_frames=0,
            labels=["bird", "cat", "dog"],
        )

    def test_find_by_id_returns_geti_dataset_without_metadata_when_archived(
        self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService
    ):
        dataset_id, dataset_path = _stage_dataset_archive(tmp_path, "some_geti.zip", b"geti-bytes")

        result = fxt_staged_dataset_service.find_by_id(dataset_id)

        assert result is not None
        assert result.id == dataset_id
        assert result.filename == str(dataset_path)
        assert result.size == 10
        assert result.compressed
        assert result.format == DatasetFormat.GETI
        assert not result.metadata

    def test_find_by_id_returns_none_when_dir_missing(
        self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService
    ):
        missing_id = uuid4()

        result = fxt_staged_dataset_service.find_by_id(missing_id)

        assert result is None

    def test_find_by_id_returns_none_when_dir_empty(
        self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService
    ):
        dataset_id = uuid4()
        dataset_dir = tmp_path / str(dataset_id)
        dataset_dir.mkdir(parents=True)

        result = fxt_staged_dataset_service.find_by_id(dataset_id)

        assert result is None

    def test_delete_by_id_removes_existing_dir_with_files(
        self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService
    ):
        dataset_id, archive_path = _stage_dataset_archive(tmp_path, "dataset.zip", b"content")

        result = fxt_staged_dataset_service.delete_by_id(dataset_id)

        assert result is True
        assert not archive_path.exists()

    def test_delete_by_id_returns_false_when_dir_missing(
        self, tmp_path: Path, fxt_staged_dataset_service: StagedDatasetService
    ):
        missing_id = uuid4()

        result = fxt_staged_dataset_service.delete_by_id(missing_id)

        assert result is False
        assert not (tmp_path / str(missing_id)).exists()
