import unittest
import os
from src.utils.utils import load_json, in_config
from src.utils.filenames import generate_filenames
from src.utils.dataset import WholeVolumeSegmentationDataset
from src.tools.train import check_hierarchy
from src.utils.trainer import run_pytorch_training
from src.utils.download_weights import download_checkpoint
import pandas as pd

class TrainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_json("configs/hearts/heart_config.json")
        config["subjects_filename"] = "configs/hearts/heart_subjects.json"
        config["model_name"] = "Distill"
        cls.model_filename = "model_weights/distill_dsm.h5"
        cls.training_log_filename = "model_weights/distill_dsm_training_log.csv"
        cls.system_config = load_json("configs/machine_config.json")
        cls.config = config

        check_hierarchy(config)

        if in_config("add_contours", config["sequence_kwargs"], False):
            config["n_outputs"] = config["n_outputs"] * 2

    def test_config(self):
        self.assertGreaterEqual(self.config["min_learning_rate"], 1e-9)

    def test_trainer(self):
        model_metrics = []
        metric_to_monitor = "val_loss"
        groups = ("training","validation")

        for name in groups:
            key = name + "_filenames"
            if key not in self.config:
                self.config[key] = generate_filenames(self.config, name, self.system_config)

        sequence_class = WholeVolumeSegmentationDataset
        bias = None

        if not os.path.isdir('model_weights'):
            download_checkpoint()

        run_pytorch_training(self.config, self.model_filename, self.training_log_filename,
                                sequence_class=sequence_class,model_metrics=model_metrics,
                                metric_to_monitor=metric_to_monitor, bias=bias, **self.system_config)

        log_data = pd.read_csv("model_weights/distill_dsm_training_log.csv")
        loss_values= log_data.loss
        loss_list = []
        for i in loss_values:
            loss_list.append(i)
        loss1 = loss_list[0]
        loss2 = loss_list[1]

        self.assertLessEqual(loss2, loss1)
