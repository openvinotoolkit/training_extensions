import asyncio
import os

from aiohttp import web
from routes import setup_routes


async def init():
    app = web.Application()
    setup_routes(app=app)
    host = os.getenv("IDLP_TENSORBOARD_API_HOST", 'idlp_tensorboard_worker')
    port = os.getenv("IDLP_TENSORBOARD_API_PORT", '8812')
    return app, host, port


def main():
    loop = asyncio.get_event_loop()
    app, host, port = loop.run_until_complete(init())
    web.run_app(app, host=host, port=port)


if __name__ == '__main__':
    main()
