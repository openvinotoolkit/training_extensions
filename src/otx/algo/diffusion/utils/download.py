"""Download files with progress bar."""

from functools import partial
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlparse

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

progress = Progress(
    TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    DownloadColumn(),
    "•",
    TransferSpeedColumn(),
    "•",
    TimeRemainingColumn(),
)


def _copy_url(task_id: TaskID, url: str, path: Path) -> None:
    """Copy data from a url to a local file."""
    parsed = urlparse(url)
    if not parsed.scheme and parsed.netloc:
        msg = f"Invalid URL: {url}"
        raise ValueError(msg)
    response = urlopen(url)  # noqa: S310
    # This will break if the response doesn't contain content length
    progress.update(task_id, total=int(response.info()["Content-length"]))

    with path.open("wb") as dest_file:
        progress.start_task(task_id)
        for data in iter(partial(response.read, 32768), b""):
            dest_file.write(data)
            progress.update(task_id, advance=len(data))


def download(url: str, dest_dir: str) -> str:
    """Download file to the given directory."""
    filename = url.split("/")[-1]

    dest_path = Path(dest_dir) / filename
    if not dest_path.exists():
        with progress:
            task_id = progress.add_task("download", filename=filename, start=False)
            _copy_url(task_id, url, dest_path)
    return str(dest_path)
