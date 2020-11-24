import aiohttp_cors
from aiohttp import web

from views import board


def setup_routes(app, base_url=''):
    app.add_routes([
        web.view(base_url + '/run', board.Tensorboard)
    ])

    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })

    # Configure CORS on all routes.
    for route in list(app.router.routes()):
        cors.add(route)
