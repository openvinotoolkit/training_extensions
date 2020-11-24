import asyncio
import logging


async def run(cmd, forget=False):
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    if forget:
        return '', ''
    stdout, stderr = await proc.communicate()

    if stdout:
        logging.info(f'[CMD]: {cmd}\n[STDOUT]: {stdout}')
    if stderr:
        logging.error(f'[CMD]: {cmd}\n[STDERR]: {stderr}')

    return stdout, stderr
