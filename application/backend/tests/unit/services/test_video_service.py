# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pytest
from av.container import InputContainer
from av.video.stream import VideoStream

from app.services.video.video_service import VideoService, _decode_group, _group_consecutive


@pytest.fixture
def mock_stream():
    return MagicMock(
        spec=VideoStream,
        time_base=Fraction(1, 25),
        average_rate=Fraction(25, 1),
    )


@pytest.fixture
def mock_container(mock_stream):
    container = MagicMock(spec=InputContainer)
    container.streams.video = [mock_stream]
    return container


@pytest.fixture
def video_service():
    svc = VideoService(cache_config=None)
    yield svc
    svc.close()


def _make_frame(pts: int):
    return MagicMock(
        spec=av.VideoFrame,
        pts=pts,
        **{"to_ndarray.return_value": np.zeros((100, 100, 3), dtype=np.uint8)},
    )


class TestExtractVideoFrames:
    def test_empty_frame_indexes(self, video_service):
        result = video_service.extract_video_frames(Path("video.mp4"), [])
        assert result == {}

    @patch.object(VideoService, "_open_stream")
    def test_single_frame(self, mock_open, video_service, mock_container, mock_stream):
        mock_open.return_value = (mock_container, mock_stream, Fraction(1, 25), Fraction(25, 1))
        mock_container.decode.return_value = [_make_frame(0)]

        result = video_service.extract_video_frames(Path("video.mp4"), [0])

        assert 0 in result
        assert result[0].shape == (100, 100, 3)
        mock_container.seek.assert_called_once()
        mock_container.decode.assert_called_once_with(mock_stream)

    @patch.object(VideoService, "_open_stream")
    def test_multiple_consecutive_frames_single_group(self, mock_open, video_service, mock_container, mock_stream):
        mock_open.return_value = (mock_container, mock_stream, Fraction(1, 25), Fraction(25, 1))
        mock_container.decode.return_value = [_make_frame(i) for i in range(5)]

        result = video_service.extract_video_frames(Path("video.mp4"), [0, 1, 2, 3, 4])

        assert len(result) == 5
        # Consecutive frames within gap=8 -> single group -> single seek
        mock_container.seek.assert_called_once()

    @patch.object(VideoService, "_open_stream")
    def test_distant_frames_multiple_groups(self, mock_open, video_service, mock_container, mock_stream):
        mock_open.return_value = (mock_container, mock_stream, Fraction(1, 25), Fraction(25, 1))
        frames_group1 = [_make_frame(0)]
        frames_group2 = [_make_frame(100)]
        mock_container.decode.side_effect = [iter(frames_group1), iter(frames_group2)]

        result = video_service.extract_video_frames(Path("video.mp4"), [0, 100])

        assert len(result) == 2
        assert mock_container.seek.call_count == 2
        assert mock_container.decode.call_count == 2

    @patch.object(VideoService, "_open_stream")
    def test_duplicate_indexes_deduped(self, mock_open, video_service, mock_container, mock_stream):
        mock_open.return_value = (mock_container, mock_stream, Fraction(1, 25), Fraction(25, 1))
        mock_container.decode.return_value = [_make_frame(0)]

        result = video_service.extract_video_frames(Path("video.mp4"), [0, 0, 0])

        assert 0 in result
        mock_container.seek.assert_called_once()

    @patch.object(VideoService, "_open_stream")
    def test_missing_frame_raises(self, mock_open, video_service, mock_container, mock_stream):
        mock_open.return_value = (mock_container, mock_stream, Fraction(1, 25), Fraction(25, 1))
        mock_container.decode.return_value = []

        with pytest.raises(RuntimeError, match="Cannot read frames"):
            video_service.extract_video_frames(Path("video.mp4"), [5])

    @patch.object(VideoService, "_open_stream")
    def test_reuses_cached_entry(self, mock_open, video_service, mock_container, mock_stream):
        mock_open.return_value = (mock_container, mock_stream, Fraction(1, 25), Fraction(25, 1))
        mock_container.decode.return_value = [_make_frame(0)]

        video_service.extract_video_frames(Path("video.mp4"), [0])
        mock_container.decode.return_value = [_make_frame(1)]
        video_service.extract_video_frames(Path("video.mp4"), [1])

        mock_open.assert_called_once()

    @patch.object(VideoService, "extract_video_frames")
    def test_extract_video_frame_delegates_to_extract_video_frames(self, mock_extract, video_service):
        expected = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_extract.return_value = {42: expected}

        result = video_service.extract_video_frame(Path("video.mp4"), frame_index=42)

        mock_extract.assert_called_once_with(video_path=Path("video.mp4"), frame_indexes=[42])
        np.testing.assert_array_equal(result, expected)


class TestRelease:
    @patch.object(VideoService, "_open_stream")
    def test_release_closes_and_evicts_cached_entry(self, mock_open, video_service, mock_container, mock_stream):
        mock_open.return_value = (mock_container, mock_stream, Fraction(1, 25), Fraction(25, 1))
        mock_container.decode.return_value = [_make_frame(0)]
        video_service.extract_video_frames(Path("video.mp4"), [0])

        video_service.release(Path("video.mp4"))

        mock_container.close.assert_called_once()
        video_service.extract_video_frames(Path("video.mp4"), [0])
        assert mock_open.call_count == 2

    def test_release_unknown_path_is_noop(self, video_service):
        video_service.release(Path("unknown.mp4"))


class TestDecodeGroup:
    def test_seek_to_beginning_for_index_zero(self, mock_container, mock_stream):
        mock_container.decode.return_value = [_make_frame(0)]
        result = {}

        _decode_group(mock_container, mock_stream, [0], result, Fraction(1, 25), Fraction(25, 1))

        mock_container.seek.assert_called_once_with(0, stream=mock_stream)
        assert 0 in result

    def test_seek_nonzero_index(self, mock_container, mock_stream):
        mock_container.decode.return_value = [_make_frame(10)]
        result = {}

        _decode_group(mock_container, mock_stream, [10], result, Fraction(1, 25), Fraction(25, 1))

        seek_call = mock_container.seek.call_args
        assert seek_call[0][0] > 0

    def test_frame_to_ndarray_called_with_rgb24(self, mock_container, mock_stream):
        frame = _make_frame(0)
        mock_container.decode.return_value = [frame]
        result = {}

        _decode_group(mock_container, mock_stream, [0], result, Fraction(1, 25), Fraction(25, 1))

        frame.to_ndarray.assert_called_once_with(format="rgb24")

    def test_collects_all_frames_in_group(self, mock_container, mock_stream):
        mock_container.decode.return_value = [_make_frame(i) for i in range(4)]
        result = {}

        _decode_group(mock_container, mock_stream, [0, 1, 2, 3], result, Fraction(1, 25), Fraction(25, 1))

        assert set(result.keys()) == {0, 1, 2, 3}

    def test_stops_decoding_after_all_targets_found(self, mock_container, mock_stream):
        frames = [_make_frame(i) for i in range(10)]
        mock_container.decode.return_value = frames
        result = {}

        _decode_group(mock_container, mock_stream, [0, 1], result, Fraction(1, 25), Fraction(25, 1))

        assert frames[0].to_ndarray.called
        assert frames[1].to_ndarray.called
        assert not frames[2].to_ndarray.called


class TestGroupConsecutive:
    def test_empty(self):
        assert _group_consecutive([]) == []

    def test_single(self):
        assert _group_consecutive([5]) == [[5]]

    def test_within_gap(self):
        assert _group_consecutive([0, 1, 8]) == [[0, 1, 8]]

    def test_beyond_gap(self):
        assert _group_consecutive([0, 1, 20]) == [[0, 1], [20]]


class TestGetVideoMetadata:
    @patch("app.services.video.video_service.av.open")
    def test_returns_metadata_from_stream_frames(self, mock_av_open, video_service):
        stream = MagicMock(
            spec=VideoStream,
            average_rate=Fraction(30, 1),
            frames=300,
            duration=None,
            time_base=Fraction(1, 30),
        )
        stream.codec_context.width = 1920
        stream.codec_context.height = 1080

        container = MagicMock(
            spec=InputContainer,
            duration=None,
        )
        container.streams.video = [stream]
        container.__enter__ = MagicMock(return_value=container)
        container.__exit__ = MagicMock(return_value=False)
        mock_av_open.return_value = container

        result = video_service.get_video_metadata(Path("video.mp4"))

        assert result.width == 1920
        assert result.height == 1080
        assert result.frame_count == 300
        assert result.fps == 30.0
        mock_av_open.assert_called_once_with(str(Path("video.mp4")))

    @patch("app.services.video.video_service.av.open")
    def test_frame_count_from_stream_duration(self, mock_av_open, video_service):
        stream = MagicMock(
            spec=VideoStream,
            average_rate=Fraction(25, 1),
            frames=0,
            duration=250,
            time_base=Fraction(1, 25),
        )
        stream.codec_context.width = 640
        stream.codec_context.height = 480

        container = MagicMock(
            spec=InputContainer,
            duration=None,
        )
        container.streams.video = [stream]
        container.__enter__ = MagicMock(return_value=container)
        container.__exit__ = MagicMock(return_value=False)
        mock_av_open.return_value = container

        result = video_service.get_video_metadata(Path("video.mp4"))

        assert result.frame_count == 250  # 250 * (1/25) * 25 = 250

    @patch("app.services.video.video_service.av.open")
    def test_frame_count_from_container_duration(self, mock_av_open, video_service):
        stream = MagicMock(
            spec=VideoStream,
            average_rate=Fraction(25, 1),
            frames=0,
            duration=None,
            time_base=Fraction(1, 25),
        )
        stream.codec_context.width = 640
        stream.codec_context.height = 480

        container = MagicMock(
            spec=InputContainer,
            duration=10_000_000,
        )
        container.streams.video = [stream]
        container.__enter__ = MagicMock(return_value=container)
        container.__exit__ = MagicMock(return_value=False)
        mock_av_open.return_value = container

        result = video_service.get_video_metadata(Path("video.mp4"))

        # 10_000_000 / 1_000_000 * 25 = 250
        assert result.frame_count == 250

    @patch("app.services.video.video_service.av.open")
    def test_raises_on_zero_fps(self, mock_av_open, video_service):
        stream = MagicMock(
            spec=VideoStream,
            average_rate=Fraction(0, 1),
        )

        container = MagicMock(spec=InputContainer)
        container.streams.video = [stream]
        container.__enter__ = MagicMock(return_value=container)
        container.__exit__ = MagicMock(return_value=False)
        mock_av_open.return_value = container

        with pytest.raises(RuntimeError, match="Error occurred while getting video metadata"):
            video_service.get_video_metadata(Path("video.mp4"))

    @patch("app.services.video.video_service.av.open")
    def test_raises_on_no_average_rate(self, mock_av_open, video_service):
        stream = MagicMock(
            spec=VideoStream,
            average_rate=None,
        )

        container = MagicMock(spec=InputContainer)
        container.streams.video = [stream]
        container.__enter__ = MagicMock(return_value=container)
        container.__exit__ = MagicMock(return_value=False)
        mock_av_open.return_value = container

        with pytest.raises(RuntimeError, match="Error occurred while getting video metadata"):
            video_service.get_video_metadata(Path("video.mp4"))

    @patch("app.services.video.video_service.av.open")
    def test_raises_on_open_failure(self, mock_av_open, video_service):
        mock_av_open.side_effect = FileNotFoundError("not found")

        with pytest.raises(RuntimeError, match="Error occurred while getting video metadata"):
            video_service.get_video_metadata(Path("nonexistent.mp4"))
