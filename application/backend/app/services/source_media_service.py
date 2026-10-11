# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

from anyio import to_thread

# Suffix marking a subdirectory that was moved aside by `quarantine_upload`.
_QUARANTINE_MARKER = ".quarantined-"


class SourceMediaService:
    """Stores video files uploaded for use as 'video_file' pipeline sources.

    Uploaded files are kept on disk in a fixed, non-project-scoped location. No database
    record is created: the resulting filesystem path is returned directly to the caller,
    who is expected to store it as the 'video_path' of a video_file source.
    """

    def __init__(self, source_media_dir: Path) -> None:
        self._source_media_dir = source_media_dir

    # TODO: We already used a similar method in staged_dataset_service.py.
    # Consider refactoring to avoid code duplication.
    async def upload(self, filename: str, file_obj: BinaryIO) -> Path:
        """
        Store an uploaded video file using a high-speed threaded block copy.

        A new UUID subdirectory is created under the configured source media root, and the
        incoming file stream is written to a file with the given filename inside it. The
        file copy operation is offloaded to a worker thread using AnyIO to prevent blocking
        the main asynchronous event loop.

        Args:
            filename: Target filename of the uploaded video within its dedicated subdirectory.
                Only the final path component is used, so any directory separators or
                traversal segments (e.g. '../../etc/passwd.mp4') are stripped.
            file_obj: A readable binary stream (such as SpooledTemporaryFile) containing the
                video payload.

        Returns:
            The absolute path to the stored video file.

        Raises:
            ValueError: If the filename has no usable name component once sanitized.
        """
        safe_name = Path(filename).name
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError(f"Invalid filename: {filename!r}")

        target_dir = self._source_media_dir / str(uuid4())
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_name

        def _perform_copy() -> None:
            try:
                file_obj.seek(0)
            except (AttributeError, OSError, ValueError):
                pass

            temp_path = target_path.with_suffix(f"{target_path.suffix}.part")

            try:
                with temp_path.open("wb") as out_f:
                    shutil.copyfileobj(file_obj, out_f, length=1024 * 1024)
                temp_path.replace(target_path)
            except Exception:
                temp_path.unlink(missing_ok=True)
                raise

        try:
            await to_thread.run_sync(_perform_copy)
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise

        return target_path.resolve()

    async def upload_from_path(self, filename: str, source_path: Path) -> Path:
        """
        Store a video file that already exists on disk, by moving (not copying) it.

        A new UUID subdirectory is created under the configured source media root, and the file is moved inside it
        with the given filename. The move is offloaded to a worker thread; when the source file is on the same
        filesystem, it is a cheap rename regardless of the video size.

        Args:
            filename: Target filename of the video within its dedicated subdirectory. Only the final path component
                is used, so any directory separators or traversal segments are stripped.
            source_path: Path of the video to move. The file is no longer at this location on success.

        Returns:
            The absolute path to the stored video file.

        Raises:
            ValueError: If the filename has no usable name component once sanitized.
        """
        safe_name = Path(filename).name
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError(f"Invalid filename: {filename!r}")

        target_dir = self._source_media_dir / str(uuid4())
        target_path = target_dir / safe_name

        def _perform_move() -> None:
            target_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(source_path, target_path)
            except Exception:
                shutil.rmtree(target_dir, ignore_errors=True)
                raise

        await to_thread.run_sync(_perform_move)
        return target_path.resolve()

    def delete_video(self, video_path: str) -> None:
        """
        Remove a previously uploaded video and its dedicated UUID subdirectory.

        This is a best-effort cleanup helper: it only removes files that live inside the
        configured source media root, in their own UUID subdirectory (as created by
        `upload`). Any path outside that root, or sitting directly at its root without a
        subdirectory, is left untouched.

        Args:
            video_path: Absolute or relative path to the video file to remove, as previously
                returned by `upload` (and stored as a video_file source's 'video_path').

        Raises:
            OSError: If removing the subdirectory fails (e.g. permissions, file in use).
                Callers are expected to treat this as best-effort and handle it themselves.
        """
        resolved = Path(video_path).resolve()
        base = self._source_media_dir.resolve()

        if not resolved.is_relative_to(base):
            return

        upload_dir = resolved.parent

        # Only remove per-upload UUID subdirectories created by `upload` (base/<uuid>/<filename>).
        if upload_dir.parent != base:
            return

        shutil.rmtree(upload_dir)

    def find_upload_by_id(self, source_media_id: UUID) -> Path | None:
        """
        Find the video file stored under the given upload's UUID subdirectory.

        Each `upload` creates its own UUID subdirectory containing exactly one video
        file, so the upload's UUID (the subdirectory name) uniquely addresses it.

        Args:
            source_media_id: UUID of the upload to look up.

        Returns:
            The resolved path of the stored video file, or None if no upload with
            this UUID exists (or its subdirectory does not hold exactly one file).
        """
        upload_dir = self._source_media_dir / str(source_media_id)

        if not upload_dir.is_dir():
            return None

        stored_files = [path for path in upload_dir.iterdir() if path.is_file()]
        if len(stored_files) != 1:
            return None

        return stored_files[0].resolve()

    def quarantine_upload(self, video_path: str | Path) -> Path:
        """
        Atomically move an upload's subdirectory aside so it can be revalidated before deletion.

        The subdirectory created by `upload` (base/<uuid>/<filename>) is renamed to a
        sibling directory with a marker suffix, so the original location immediately
        stops serving the file while the caller rechecks whether any source references
        it. Callers either delete the quarantined upload (`delete_quarantined_upload`)
        or put it back (`restore_quarantined_upload`).

        Args:
            video_path: Path of the stored video, as returned by `find_uploads_by_filename`.

        Returns:
            The path of the video file inside the quarantined directory.

        Raises:
            FileNotFoundError: If the path does not address a stored upload
                (base/<uuid>/<file>).
        """
        resolved = Path(video_path).resolve()
        base = self._source_media_dir.resolve()
        upload_dir = resolved.parent

        if not resolved.is_relative_to(base) or upload_dir.parent != base or not resolved.is_file():
            raise FileNotFoundError(video_path)

        quarantined_dir = base / f"{upload_dir.name}{_QUARANTINE_MARKER}{uuid4()}"
        upload_dir.rename(quarantined_dir)
        return quarantined_dir / resolved.name

    def restore_quarantined_upload(self, quarantined_video_path: str | Path) -> None:
        """
        Move a quarantined upload back to its original subdirectory.

        Args:
            quarantined_video_path: Path returned by `quarantine_upload`.

        Raises:
            OSError: If moving the directory back fails.
        """
        quarantined_dir = Path(quarantined_video_path).parent
        original_dir = quarantined_dir.with_name(quarantined_dir.name.split(_QUARANTINE_MARKER)[0])
        quarantined_dir.rename(original_dir)

    def delete_quarantined_upload(self, quarantined_video_path: str | Path) -> None:
        """
        Remove a quarantined upload together with its subdirectory.

        Args:
            quarantined_video_path: Path returned by `quarantine_upload`.

        Raises:
            OSError: If removing the directory fails.
        """
        shutil.rmtree(Path(quarantined_video_path).parent)
