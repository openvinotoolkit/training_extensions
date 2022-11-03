from otx.backends.torch.anomalib.job import AnomalibJob

from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
)
from pytorch_lightning import Trainer, seed_everything

class AnomalibTrainer(AnomalibJob):
    def init_callbacks(self, config):
        self.callbacks = [
            MinMaxNormalizationCallback(),
            MetricsConfigurationCallback(
                adaptive_threshold=config.metrics.threshold.adaptive,
                default_image_threshold=config.metrics.threshold.image_default,
                default_pixel_threshold=config.metrics.threshold.pixel_default,
                image_metric_names=config.metrics.image,
                pixel_metric_names=config.metrics.pixel,
            ),

        ]

    def run(self, model, dataset, config, **kwargs):
        logger.info("[{__file__}] run()")
        self.init_callbacks(config)
        trainer_config = dict(
            cfg0="value0",
            cfg1="value1",
        )
        self.trainer = Trainer(**trainer_config, logger=False, callbacks=self.callbacks)
        self.trainer.fit(model=model, datamodule=dataset)

        results = dict(
            spec=self.spec,
            model=model.state_dict(),
            config=trainer_config
        )
        logger.info(f"results of run = {results}")
        return results