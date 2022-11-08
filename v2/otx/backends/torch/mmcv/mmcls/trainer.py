from otx.backends.torch.mmcv.mmcls.job import MMClsJob
from otx.utils.logger import get_logger

logger = get_logger()


class MMClsTrainer(MMClsJob):
    def run(self, model, data, task_config=None, **kwargs):
        logger.info(f"model = {model}, datasets = {data}, config = {task_config}, kwargs = {kwargs}")
        logger.info(f"job config = {self.config}")
        task_config = self.configure(task_config)

        # some backend specific info
        self.get_torch_version()
        self.get_mmcv_version()
        self.get_mmcls_version()
        self.is_gpu_available()

        return dict(final_ckpt="output_ckpt_path")
