# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Video decoding service with LRU frame caching and TTL-based handle management.

This module provides :class:`VideoService`, a thread-safe service for extracting
video frames backed by PyAV. It keeps file handles open, idle ones are automatically evicted by
a background cleanup thread after a configurable TTL.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic

import av
import numpy as np
from av.codec.context import ThreadType
from loguru import logger
from pydantic import BaseModel, Field


class CacheConfig(BaseModel):
    """Configuration for the video service cache behaviour.

    Attributes:
        ttl: Time-to-live (in seconds) for idle video handles before eviction.
        cleanup_interval: Interval (in seconds) between background cleanup sweeps.
    """

    ttl: float
    cleanup_interval: float


@dataclass
class _CacheEntry:
    """Internal cache entry holding a video container handle.

    Attributes:
        container: Open PyAV input container for the video file.
        stream: Video stream selected from the container.
        time_base: Stream time-base as a :class:`~fractions.Fraction`, used to convert PTS values to seconds.
        avg_rate: Average frame-rate as a :class:`~fractions.Fraction`, used
            to map between PTS values and frame indexes.
        lock: Per-entry lock protecting container seek/decode operations.
        last_access: Monotonic timestamp of the most recent access (used for TTL).
    """

    container: av.container.InputContainer
    stream: av.video.stream.VideoStream
    time_base: Fraction
    avg_rate: Fraction
    lock: Lock = field(default_factory=Lock)
    last_access: float = field(default_factory=monotonic)


class VideoMetadata(BaseModel):
    """Metadata extracted from a video file.

    Attributes:
        width: Video width in pixels.
        height: Video height in pixels.
        frame_count: Total number of frames in the video.
        fps: Average frames per second.
    """

    width: int = Field(..., description="Video width in pixels", gt=0)
    height: int = Field(..., description="Video height in pixels", gt=0)
    frame_count: int = Field(..., description="Total number of frames in the video", gt=0)
    fps: float = Field(..., description="Average frames per second", gt=0)


def _frame_index_from_pts(frame_pts: int, time_base: Fraction, avg_rate: Fraction) -> int:
    """Compute the zero-based frame index from a decoded frame's PTS."""
    return round(frame_pts * time_base * avg_rate)


def _pts_from_index(index: int, time_base: Fraction, avg_rate: Fraction) -> int:
    """Compute the approximate PTS value for a given frame index."""
    return int(Fraction(index) / (avg_rate * time_base))


def _group_consecutive(sorted_indexes: list[int], gap: int = 8) -> list[list[int]]:
    """Group sorted frame indexes into runs where neighbours are within *gap* frames.

    Frames within a small gap are cheaper to decode sequentially than to seek
    to individually, so they are kept in the same group.

    Args:
        sorted_indexes: Ascending, deduplicated frame indexes.
        gap: Maximum distance between consecutive indexes in a single group.

    Returns:
        List of groups, each a list of ascending frame indexes.
    """
    if not sorted_indexes:
        return []
    groups: list[list[int]] = [[sorted_indexes[0]]]
    for idx in sorted_indexes[1:]:
        if idx - groups[-1][-1] <= gap:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def _decode_group(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
    group: list[int],
    result: dict[int, np.ndarray],
    time_base: Fraction,
    avg_rate: Fraction,
) -> None:
    """Seek once to the first index of *group* and decode forward, collecting all needed frames.

    Args:
        container: Open PyAV input container.
        stream: Video stream to decode.
        group: Ascending list of frame indexes to extract (must be non-empty).
        result: Dictionary to populate with ``{frame_index: rgb_ndarray}``.
        time_base: Stream time-base as a :class:`~fractions.Fraction`.
        avg_rate: Stream average frame-rate as a :class:`~fractions.Fraction`.
    """
    target_set = set(group)
    first_target = group[0]
    last_target = group[-1]

    if first_target == 0:
        # Seek to the very beginning - decode count equals frame index.
        container.seek(0, stream=stream)
        decode_offset = 0
    else:
        seek_pts = max(_pts_from_index(first_target, time_base, avg_rate) - 1, 0)
        container.seek(seek_pts, stream=stream)
        # We don't know exactly which frame the seek landed on, so we
        # calibrate using the *minimum* PTS seen in the first few decoded
        # frames (to handle B-frame reordering).
        decode_offset = None  # will be set on the first frame

    # We rely on a simple decode counter to assign frame indexes.
    # PTS-based indexing is only used once - to figure out which frame
    # the seek landed on (for non-zero seeks).  After that, every call
    # to container.decode() yields the next presentation-order frame, so
    # a plain counter is the most reliable approach, especially for
    # containers (AVI) with non-monotonic or offset PTS values.
    frame_counter = 0

    for frame in container.decode(stream):
        if decode_offset is None:
            # First frame after a non-zero seek: calibrate offset from PTS.
            if frame.pts is not None:
                decode_offset = _frame_index_from_pts(frame.pts, time_base, avg_rate)
            else:
                # No PTS at all - best guess is the seek target itself.
                decode_offset = first_target

        frame_idx = decode_offset + frame_counter
        frame_counter += 1

        if frame_idx < first_target:
            continue
        if frame_idx in target_set:
            result[frame_idx] = frame.to_ndarray(format="rgb24")
            target_set.discard(frame_idx)
            if not target_set:
                return
        if frame_idx > last_target + 16:
            # Allow some overshoot for B-frame reordering before giving up
            break


class VideoService:
    """Video decoding service with TTL-based handle management.

    Features:
        - Keeps PyAV container handles open and reuses them across calls.
        - TTL is renewed on each access; expired entries are cleaned up by a
          background thread.

    Example::

        config = CacheConfig(ttl=300, cleanup_interval=60)
        svc = VideoService(cache_config=config)
        frame = svc.extract_video_frame(Path("video.mp4"), frame_index=42)
        svc.close()
    """

    def __init__(self, cache_config: CacheConfig | None = None) -> None:
        """Initialise the video service.

        Args:
            cache_config: Optional cache configuration. When provided, container handles are cached
            with TTL-based eviction.
                When *None*, no caching or background cleanup is performed.
        """
        self._cache_config = cache_config
        self._entries: dict[str, _CacheEntry] = {}
        self._dict_lock = Lock()
        self._stop_event = Event()
        if cache_config is not None:
            self._cleanup_thread = Thread(
                target=self._cleanup_loop,
                daemon=True,
                name="VideoServiceCleanup",
                kwargs={"cleanup_interval": cache_config.cleanup_interval, "ttl": cache_config.ttl},
            )
            self._cleanup_thread.start()

    def get_video_metadata(self, video_path: Path) -> VideoMetadata:
        """Extract metadata from a video file.

        The frame count is determined using the following fallback chain:

        1. ``stream.frames`` — the count reported by the container header.
        2. ``stream.duration * stream.time_base * fps`` — computed from the
           stream-level duration (e.g. when the container is WebM/MKV and
           ``stream.frames`` is 0).
        3. ``container.duration / av.time_base * fps`` — computed from the
           container-level duration as a last resort.

        Args:
            video_path: Path to the video file on disk.

        Returns:
            A :class:`VideoMetadata` instance containing width, height, frame
            count, and FPS.

        Raises:
            RuntimeError: If the video file cannot be opened or read.
        """
        try:
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                if not stream.average_rate or float(stream.average_rate) == 0:
                    raise RuntimeError(f"Cannot determine FPS for video: {video_path}")
                fps = float(stream.average_rate)
                frame_count = stream.frames if stream.frames else 0
                if frame_count == 0 and fps > 0 and stream.duration and stream.time_base:
                    frame_count = round(float(stream.duration * stream.time_base) * fps)
                if frame_count == 0 and fps > 0 and container.duration:
                    frame_count = round((container.duration / av.time_base) * fps)
                width = stream.codec_context.width
                height = stream.codec_context.height
        except Exception as e:
            logger.error(f"Failed getting metadata for video {video_path}", exc_info=e)
            raise RuntimeError("Error occurred while getting video metadata")

        return VideoMetadata(
            width=width,
            height=height,
            frame_count=frame_count,
            fps=fps,
        )

    def extract_video_frame(
        self,
        video_path: Path,
        frame_index: int,
    ) -> np.ndarray:
        """Extract a single video frame.

        This is a convenience wrapper around :meth:`extract_video_frames`.

        Args:
            video_path: Path to the video file on disk.
            frame_index: Zero-based index of the frame to extract.

        Returns:
            Decoded video frame as a numpy array in RGB format with shape
            ``(height, width, 3)``.

        Raises:
            RuntimeError: If the requested frame cannot be decoded.
        """
        frames = self.extract_video_frames(video_path=video_path, frame_indexes=[frame_index])
        return frames[frame_index]

    def extract_video_frames(
        self,
        video_path: Path,
        frame_indexes: list[int],
    ) -> dict[int, np.ndarray]:
        """Extract multiple video frames.

        Missing frames are decoded in grouped batches with a single seek per
        group of consecutive indexes for efficiency.

        Args:
            video_path: Path to the video file on disk.
            frame_indexes: List of zero-based frame indexes to extract.

        Returns:
            Dictionary mapping each requested frame index to the decoded frame
            as a numpy array in RGB format with shape ``(height, width, 3)``.

        Raises:
            RuntimeError: If any requested frame cannot be decoded.
        """
        if not frame_indexes:
            return {}

        path_key = str(video_path)
        entry = self._get_or_create_entry(path_key=path_key)
        entry.last_access = monotonic()

        frames: dict[int, np.ndarray] = {}
        sorted_indexes = sorted(set(frame_indexes))
        groups = _group_consecutive(sorted_indexes)

        with entry.lock:
            for group in groups:
                _decode_group(entry.container, entry.stream, group, frames, entry.time_base, entry.avg_rate)

        missing = set(sorted_indexes) - frames.keys()
        if missing:
            raise RuntimeError(f"Cannot read frames {sorted(missing)} from video: {video_path}")
        return frames

    def close(self) -> None:
        """Stop the background cleanup thread and release all video container handles.

        This method is safe to call even if caching is disabled (i.e.
        ``cache_config`` was *None* at construction time).
        """
        self._stop_event.set()
        if self._cache_config is not None:
            self._cleanup_thread.join(timeout=5)
        with self._dict_lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            with entry.lock:
                entry.container.close()
        logger.debug("VideoService closed, all handles released")

    def release(self, video_path: Path) -> None:
        """Close and evict the cached container handle for a video, if any.

        This must be called before moving or deleting a video file that may have been
        opened by this service, since an open handle prevents such operations on Windows.

        Args:
            video_path: Path to the video file on disk.
        """
        with self._dict_lock:
            entry = self._entries.pop(str(video_path), None)
        if entry is None:
            return
        with entry.lock:
            entry.container.close()
        logger.debug("Released video {}", video_path)

    def _get_or_create_entry(self, path_key: str) -> _CacheEntry:
        """Get an existing cache entry or create a new one with an open PyAV container.

        Args:
            path_key: String representation of the video file path used as the
                cache key.

        Returns:
            A :class:`_CacheEntry` with an open container and video stream.

        Raises:
            RuntimeError: If the video cannot be opened or is missing required
                stream metadata (``time_base`` / ``average_rate``).
        """
        with self._dict_lock:
            entry = self._entries.get(path_key)
            if entry is not None:
                return entry

            container, stream, time_base, avg_rate = VideoService._open_stream(path_key)
            entry = _CacheEntry(
                container=container,
                stream=stream,
                time_base=time_base,
                avg_rate=avg_rate,
            )
            self._entries[path_key] = entry
            logger.debug("Opened video {}", path_key)
            return entry

    @staticmethod
    def _open_stream(
        video_path: str,
    ) -> tuple[av.container.InputContainer, av.video.stream.VideoStream, Fraction, Fraction]:
        container = None
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            if stream.time_base is None or stream.average_rate is None:
                raise RuntimeError(f"Video stream is missing time_base or average_rate: {video_path}")
            stream.thread_type = ThreadType.AUTO
        except Exception as exc:
            if container is not None:
                container.close()
            raise RuntimeError(f"Cannot open video: {video_path}") from exc
        return container, stream, Fraction(stream.time_base), Fraction(stream.average_rate)

    def _cleanup_loop(self, cleanup_interval: int, ttl: float) -> None:
        """Background loop that periodically evicts expired cache entries.

        Args:
            cleanup_interval: Seconds to sleep between eviction sweeps.
            ttl: Time-to-live in seconds; entries idle longer than this are evicted.
        """
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=cleanup_interval)
            if self._stop_event.is_set():
                break
            self._evict_expired(ttl=ttl)

    def _evict_expired(self, ttl: float) -> None:
        """Evict cache entries whose last access exceeds the given TTL.

        Args:
            ttl: Maximum idle time in seconds before an entry is evicted.
        """
        now = monotonic()
        with self._dict_lock:
            expired_keys = [key for key, entry in self._entries.items() if (now - entry.last_access) >= ttl]
            for key in expired_keys:
                entry = self._entries.pop(key)
                entry.container.close()
                logger.debug("Evicted expired video {}", key)
