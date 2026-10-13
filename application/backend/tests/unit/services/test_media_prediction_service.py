# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, call, patch
from uuid import uuid4

import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.models import (
    BatchInferenceMedia,
    BatchInferencePrediction,
    BatchInferenceResult,
    DatasetItemAnnotation,
    Project,
    Task,
    Video,
)
from app.models.media import (
    Image,
    MediaListPredictionRequest,
    MediaPredictionRequest,
    MediaType,
    VideoFrame,
    VideoRange,
)
from app.models.system import DeviceInfo, DeviceType
from app.services import DatasetService, LabelService, MediaPredictionService, MediaService, ResourceNotFoundError
from app.services.inference import InferenceServer
from app.services.media_numpy_loader import MediaNumpyLoader
from app.services.media_prediction_service import Frame, LoadedMedia


class TestMediaPredictionServiceUnit:
    """Unit tests for MediaPredictionService."""

    @pytest.fixture
    def fxt_label_service(self):
        return MagicMock(spec=LabelService)

    @pytest.fixture
    def fxt_media_service(self):
        return MagicMock(spec=MediaService)

    @pytest.fixture
    def fxt_media_numpy_loader(self):
        return MagicMock(spec=MediaNumpyLoader)

    @pytest.fixture
    def fxt_dataset_service(self):
        return MagicMock(spec=DatasetService)

    @pytest.fixture
    def fxt_inference_server(self):
        return MagicMock(spec=InferenceServer)

    @pytest.fixture
    def fxt_media_prediction_service(
        self, fxt_label_service, fxt_media_service, fxt_inference_server, fxt_media_numpy_loader
    ):
        def _create_media_prediction_service(inference_keyframe_stride: int = 1):
            db_session = MagicMock(spec=Session)
            return MediaPredictionService(
                label_service=fxt_label_service,
                media_service=fxt_media_service,
                inference_server=fxt_inference_server,
                inference_keyframe_stride=inference_keyframe_stride,
                media_numpy_loader=fxt_media_numpy_loader,
                db_session=db_session,
            )

        return _create_media_prediction_service

    @pytest.mark.parametrize(
        "index, frame_indexes, inference_keyframe_stride, expected_result",
        [
            (0, [0, 1, 2, 3, 4, 5], 2, True),
            (1, [0, 1, 2, 3, 4, 5], 2, False),
            (2, [0, 1, 2, 3, 4, 5], 2, True),
            (3, [0, 1, 2, 3, 4, 5], 2, False),
            (4, [0, 1, 2, 3, 4, 5], 2, True),
            (5, [0, 1, 2, 3, 4, 5], 2, True),
        ],
    )
    def test_is_frame_index_to_infer(self, index, frame_indexes, inference_keyframe_stride, expected_result):
        result = MediaPredictionService._is_frame_index_to_infer(
            index=index, frame_indexes=frame_indexes, inference_keyframe_stride=inference_keyframe_stride
        )
        assert result == expected_result

    def test_load_frame_range(self, fxt_media_prediction_service, fxt_media_service):
        project = MagicMock(spec=Project)
        video = MagicMock(spec=Video, id=uuid4())

        # One frame is already annotated
        video_frame = MagicMock(spec=VideoFrame, video_id=video.id, frame_index=0)
        fxt_media_service.search_video_frames_by_video_id_and_indexes.return_value = [video_frame]

        media_prediction_service = fxt_media_prediction_service()
        result = media_prediction_service._load_frame_range(
            project=project, video=video, video_range=VideoRange(start_frame=0, end_frame=2, stride=1)
        )

        # Assert that one annotated and one unannotated frame is returned
        fxt_media_service.search_video_frames_by_video_id_and_indexes.assert_called_once_with(
            project=project, video_id=video.id, frame_indexes=[0, 1, 2]
        )
        assert result == [
            Frame(frame_index=0, annotated_frame=video_frame, skip=False),
            Frame(frame_index=1, skip=False),
            Frame(frame_index=2, skip=False),
        ]

    def test_load_frame_range_skip(self, fxt_media_prediction_service, fxt_media_service):
        project = MagicMock(spec=Project)
        video = MagicMock(spec=Video, id=uuid4())

        # One frame is already annotated
        video_frame = MagicMock(spec=VideoFrame, video_id=video.id, frame_index=0)
        fxt_media_service.search_video_frames_by_video_id_and_indexes.return_value = [video_frame]

        media_prediction_service = fxt_media_prediction_service(2)
        result = media_prediction_service._load_frame_range(
            project=project, video=video, video_range=VideoRange(start_frame=0, end_frame=5, stride=1)
        )

        # Assert that one annotated and one unannotated frame is returned
        fxt_media_service.search_video_frames_by_video_id_and_indexes.assert_called_once_with(
            project=project, video_id=video.id, frame_indexes=[0, 1, 2, 3, 4, 5]
        )
        assert result == [
            Frame(frame_index=0, annotated_frame=video_frame, skip=False),
            Frame(frame_index=1, skip=True),
            Frame(frame_index=2, skip=False),
            Frame(frame_index=3, skip=True),
            Frame(frame_index=4, skip=False),
            Frame(frame_index=5, skip=False),
        ]

    def test_load_media(self, fxt_media_prediction_service, fxt_media_service):
        model_id = uuid4()
        image_id = uuid4()
        video_id = uuid4()
        project = MagicMock(spec=Project, id=uuid4())
        image = MagicMock(spec=Image, id=image_id, type=MediaType.IMAGE)
        video = MagicMock(spec=Video, id=video_id, type=MediaType.VIDEO)

        video_range = VideoRange(start_frame=0, end_frame=1, stride=1)
        fxt_media_service.get_media_by_ids.return_value = [image, video]

        media_prediction_service = fxt_media_prediction_service()
        with patch.object(
            media_prediction_service,
            "_load_frame_range",
            return_value=[
                Frame(frame_index=0, skip=False),
                Frame(frame_index=1, skip=True),
                Frame(frame_index=2, skip=False),
            ],
        ) as mock_load_frame_range:
            result = media_prediction_service._load_media(
                project=project,
                request=MediaListPredictionRequest(
                    model_id=model_id,
                    media=[
                        MediaPredictionRequest(media_id=image_id, range=None),
                        MediaPredictionRequest(media_id=video_id, range=video_range),
                    ],
                    device="AUTO",
                ),
            )

        fxt_media_service.get_media_by_ids.assert_called_once_with(
            project_id=project.id, media_ids=[image_id, video_id]
        )
        mock_load_frame_range.assert_called_once_with(project=project, video=video, video_range=video_range)
        assert result == LoadedMedia(
            single_media=[image],
            video_frames={
                video: [
                    Frame(frame_index=0, skip=False),
                    Frame(frame_index=1, skip=True),
                    Frame(frame_index=2, skip=False),
                ]
            },
        )

    def test_load_media_not_found(self, fxt_media_prediction_service, fxt_media_service):
        model_id = uuid4()
        media_id = uuid4()
        project = MagicMock(spec=Project, id=uuid4())

        fxt_media_service.get_media_by_ids.return_value = []

        media_prediction_service = fxt_media_prediction_service()
        with pytest.raises(ResourceNotFoundError):
            media_prediction_service._load_media(
                project=project,
                request=MediaListPredictionRequest(
                    model_id=model_id,
                    media=[
                        MediaPredictionRequest(media_id=media_id, range=None),
                    ],
                    device="AUTO",
                ),
            )

        fxt_media_service.get_media_by_ids.assert_called_once_with(project_id=project.id, media_ids=[media_id])

    def test_convert_to_inference_input(self, fxt_media_prediction_service, fxt_media_service, fxt_media_numpy_loader):
        image_id = uuid4()
        video_id = uuid4()
        project = MagicMock(spec=Project, id=uuid4())
        image = MagicMock(spec=Image, id=image_id, type=MediaType.IMAGE)
        video = MagicMock(spec=Video, id=video_id, type=MediaType.VIDEO)
        video_frame = MagicMock(spec=VideoFrame, type=MediaType.VIDEO_FRAME, video_id=video_id, frame_index=0)

        not_annotated_video_frame_binary = np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)

        loaded_media = LoadedMedia(
            single_media=[image],
            video_frames={
                video: [
                    Frame(frame_index=0, annotated_frame=video_frame, skip=False),
                    Frame(frame_index=1, skip=False),
                    Frame(frame_index=2, skip=True),
                    Frame(frame_index=3, skip=False),
                ]
            },
        )

        fxt_media_service.get_frame_binaries.return_value = {
            1: not_annotated_video_frame_binary,
            3: not_annotated_video_frame_binary,
        }

        media_prediction_service = fxt_media_prediction_service()
        fxt_media_numpy_loader.load_media_binary.side_effect = [
            np.random.rand(100, 100, 3),
            np.random.rand(100, 100, 3),
        ]
        result_inputs = media_prediction_service._convert_to_inference_input(
            project=project,
            loaded_media=loaded_media,
        )

        fxt_media_numpy_loader.load_media_binary.assert_has_calls(
            [
                call(project_id=project.id, media=image),
                call(project_id=project.id, media=video_frame),
            ]
        )
        fxt_media_service.get_frame_binaries.assert_called_once_with(project=project, video=video, frame_indexes=[1, 3])
        assert len(result_inputs) == 4
        na_frame_input = next(
            inp
            for inp in result_inputs
            if inp.media_id == video_id and (inp.frame_index == 1 or (inp.frame_index == 3))
        )
        np.testing.assert_array_equal(na_frame_input.data, not_annotated_video_frame_binary)

    def test_convert_result(self, fxt_media_prediction_service):
        image_id = uuid4()
        video_id = uuid4()
        image = MagicMock(spec=Image, id=image_id, type=MediaType.IMAGE)
        video = MagicMock(spec=Video, id=video_id, type=MediaType.VIDEO)
        video_frame = MagicMock(spec=VideoFrame, type=MediaType.VIDEO_FRAME, video_id=video_id, frame_index=0)

        image_annotation = MagicMock(spec=DatasetItemAnnotation)
        video_frame_annotation = MagicMock(spec=DatasetItemAnnotation)

        loaded_media = LoadedMedia(
            single_media=[image],
            video_frames={
                video: [
                    Frame(frame_index=0, annotated_frame=video_frame, skip=False),
                    Frame(frame_index=1, skip=False),
                    Frame(frame_index=2, skip=True),
                    Frame(frame_index=3, skip=False),
                ]
            },
        )
        inference_result = {
            (image_id, None): [image_annotation],
            (video_id, 0): [],
            (video_id, 1): [video_frame_annotation],
            (video_id, 3): [],
        }

        media_prediction_service = fxt_media_prediction_service()
        result = media_prediction_service._convert_result(loaded_media=loaded_media, inference_result=inference_result)
        assert result == BatchInferenceResult(
            predictions=[
                BatchInferencePrediction(media=BatchInferenceMedia(id=image_id), prediction=[image_annotation]),
                BatchInferencePrediction(media=BatchInferenceMedia(id=video_id, frame_index=0), prediction=[]),
                BatchInferencePrediction(
                    media=BatchInferenceMedia(id=video_id, frame_index=1), prediction=[video_frame_annotation]
                ),
                BatchInferencePrediction(
                    media=BatchInferenceMedia(id=video_id, frame_index=2), prediction=[video_frame_annotation]
                ),
                BatchInferencePrediction(media=BatchInferenceMedia(id=video_id, frame_index=3), prediction=[]),
            ]
        )

    def test_predict_media(self, fxt_media_prediction_service, fxt_label_service, fxt_inference_server):
        model_id = uuid4()
        task = MagicMock(spec=Task)
        project = MagicMock(spec=Project, id=uuid4(), task=task)

        loaded_media = LoadedMedia(single_media=[], video_frames={})
        inputs = MagicMock(spec=list)
        labels = MagicMock(spec=list)
        batch_inference_result = MagicMock(spec=BatchInferenceResult)
        infer_batch_result = {}

        fxt_label_service.list_all.return_value = labels
        fxt_inference_server.infer_batch.return_value = infer_batch_result

        request = MediaListPredictionRequest(model_id=model_id, media=[], device="AUTO")
        device = DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)

        media_prediction_service = fxt_media_prediction_service()
        with (
            patch.object(media_prediction_service, "_load_media", return_value=loaded_media) as mock_load_media,
            patch.object(
                media_prediction_service, "_convert_to_inference_input", return_value=inputs
            ) as mock_convert_to_inference_input,
            patch.object(
                media_prediction_service, "_convert_result", return_value=batch_inference_result
            ) as mock_convert_result,
        ):
            result = media_prediction_service.predict_media(project=project, request=request, device=device)

        mock_load_media.assert_called_once_with(project=project, request=request)
        mock_convert_to_inference_input.assert_called_once_with(project=project, loaded_media=loaded_media)
        fxt_label_service.list_all.assert_called_once_with(project_id=project.id)
        fxt_inference_server.infer_batch.assert_called_once_with(
            project_id=project.id,
            model_id=model_id,
            model_variant_id=None,
            device=device,
            labels=labels,
            inputs=inputs,
            confidence_threshold=None,
        )
        mock_convert_result.assert_called_once_with(loaded_media=loaded_media, inference_result=infer_batch_result)
        assert result == batch_inference_result

    @pytest.mark.parametrize("confidence_threshold", [0.8, None], ids=["with_threshold", "without_threshold"])
    def test_predict_media_confidence_threshold(
        self, confidence_threshold, fxt_media_prediction_service, fxt_label_service, fxt_inference_server
    ):
        """The requested threshold is forwarded to the inference server together with the batch."""
        project = MagicMock(spec=Project, id=uuid4(), task=MagicMock(spec=Task))
        request = MediaListPredictionRequest(
            model_id=uuid4(), media=[], device="AUTO", confidence_threshold=confidence_threshold
        )
        device = DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)

        media_prediction_service = fxt_media_prediction_service()
        with (
            patch.object(media_prediction_service, "_load_media", return_value=LoadedMedia([], {})),
            patch.object(media_prediction_service, "_convert_to_inference_input", return_value=[]),
            patch.object(media_prediction_service, "_convert_result"),
        ):
            media_prediction_service.predict_media(project=project, request=request, device=device)

        assert fxt_inference_server.infer_batch.call_args.kwargs["confidence_threshold"] == confidence_threshold

    @pytest.mark.parametrize(
        "frame_index, keyframe_indexes, expected",
        [
            (0, [0, 5, 10, 15, 20], 0),
            (2, [0, 5, 10, 15, 20], 0),
            (3, [0, 5, 10, 15, 20], 5),
            (5, [0, 5, 10, 15, 20], 5),
            (7, [0, 5, 10, 15, 20], 5),
            (8, [0, 5, 10, 15, 20], 10),
            (14, [0, 5, 10, 15, 20], 15),
            (20, [0, 5, 10, 15, 20], 20),
            (0, [0], 0),
            (5, [0, 10], 0),  # equidistant, picks the earlier one (<=)
            (100, [0, 10, 20], 20),
        ],
    )
    def test_find_nearest_keyframe_index(self, frame_index, keyframe_indexes, expected):
        result = MediaPredictionService._find_nearest_keyframe_index(frame_index, keyframe_indexes)
        assert result == expected
