import aiohttp_cors
from aiohttp import web

from common.utils.run_cmd import run


class Tensorboard(web.View, aiohttp_cors.CorsViewMixin):

    async def get(self):
        print("Start")
        folder = self.request.query.get("folder")
        cmd_check = 'ps a'
        o, e = await run(cmd_check)
        print(cmd_check)
        cmd_stop = 'pkill -f tensorboard'
        _, _ = await run(cmd_stop)
        print(cmd_stop)
        cmd_start = f'tensorboard --logdir /training/{folder}/tf_logs --port 6006 --host idlp_tensorboard_worker'
        o, e = await run(cmd_start, forget=True)
        print(cmd_start)
        print("end")
        return web.Response(status=200)
