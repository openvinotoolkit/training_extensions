# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import sys


def runner(
    cmd: list,
    stdout_stream = sys.stdout.buffer,
    stderr_stream = sys.stderr.buffer,
    **kwargs,
) -> tuple:
    """
    Executes a command and captures its output.

    Args:
        cmd (list): The command to be executed.
        stdout_stream (file-like object, optional): The stream to capture the standard output. Defaults to sys.stdout.buffer.
        stderr_stream (file-like object, optional): The stream to capture the standard error. Defaults to sys.stderr.buffer.
        **kwargs: Additional keyword arguments to be passed to asyncio.create_subprocess_exec().

    Returns:
        tuple: A tuple containing the return code, standard output, and standard error.

    Raises:
        Exception: If an error occurs while executing the command.

    """
    async def stream_handler(in_stream, out_stream) -> bytearray:
        output = bytearray()
        # buffer line
        line = bytearray()
        while True:
            c = await in_stream.read(1)
            if not c:
                break
            line.extend(c)
            if c == b"\n":
                out_stream.write(line)
                output.extend(line)
                line = bytearray()
        return output

    async def run_and_capture(cmd: list) -> tuple:
        environ = os.environ.copy()
        environ["PYTHONUNBUFFERED"] = "1"
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environ,
            **kwargs,
        )

        try:
            stdout, stderr = await asyncio.gather(
                stream_handler(process.stdout, stdout_stream),
                stream_handler(process.stderr, stderr_stream),
            )
        except Exception:
            process.kill()
            raise
        finally:
            rc = await process.wait()
        return rc, stdout, stderr

    rc, stdout, stderr = asyncio.run(run_and_capture(cmd))

    return rc, stdout, stderr


def check_run(cmd: list, **kwargs) -> tuple:
    rc, stdout, stderr = runner(cmd, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()

    if rc != 0:
        stderr_split = stderr.decode("utf-8").splitlines()
        i = -1
        for line in stderr_split:
            i += 1
            if line.startswith("Traceback"):
                break
        stderr = "\n".join(stderr[i:])
    assert rc == 0, stderr
    return rc, stdout, stderr
