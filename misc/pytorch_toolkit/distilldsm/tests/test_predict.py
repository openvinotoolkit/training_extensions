import unittest
import os
from src.utils.utils import load_json, in_config
from src.utils.filenames import generate_filenames, load_sequence
from src.utils.dataset import WholeVolumeSegmentationDataset
from src.utils.predict_utils import volumetric_predictions
from src.utils.download_weights import download_checkpoint
import pandas as pd

#TO-DO Add download data from gdrive

class InferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_json("configs/hearts/heart_config.json")
        config["subjects_filename"] = "configs/hearts/heart_subjects.json"
        config["model_name"] = "Distill"
        cls.model_filename = "model_weights/distill_dsm.h5"
        cls.out_dir = "model_weights/"
        cls.training_log_filename = "model_weights/distill_dsm_training_log.csv"
        cls.system_config = load_json("configs/machine_config.json")
        cls.config = config

        if not os.path.exists(cls.model_filename):
            download_checkpoint()

        # check_hierarchy(config)

        if in_config("add_contours", config["sequence_kwargs"], False):
            config["n_outputs"] = config["n_outputs"] * 2

    def test_pytorch_inference(self):
        groups = ("validation",)

        for name in groups:
            if "sequence" in self.config:
                sequence = load_sequence(self.config["sequence"])
            else:
                sequence = None
            key = name + "_filenames"
            if key not in self.config:
                self.config[key] = generate_filenames(self.config, name, self.system_config)
        labels = self.config["sequence_kwargs"]["labels"]

        print(self.config[key])

        _, dice = volumetric_predictions(model_filename=self.model_filename, filenames=self.config[key], 
                    prediction_dir=self.out_dir, model_name=self.config["model_name"],
                    n_features=self.config["n_features"], window=self.config["window"],
                    criterion_name=self.config["evaluation_metric"],
                    package="pytorch", n_gpus=1, n_workers=1, batch_size=1,
                    model_kwargs=self.config["model_kwargs"],
                    sequence_kwargs=self.config["sequence_kwargs"],
                    sequence=sequence,
                    n_outputs=self.config["n_outputs"],
                    metric_names=in_config("metric_names", self.config, None),
                    evaluate_predictions=True,
                    resample_predictions=(not False),
                    interpolation="linear",
                    output_template="hearts_Validation_{subject}.nii.gz",
                    segmentation=None,
                    segmentation_labels=labels,
                    threshold=0.5,
                    sum_then_threshold=False,
                    label_hierarchy=False,
                    write_input_images=False)
        self.assertGreaterEqual(dice, 0.78)

    # TO-DO Add test for ONNX and IR

    # def test_onnx_inference(self):
    #     model_dir = self.out_dir
    #     onnx_checkpoint = os.path.join(model_dir, self.config.get('model_onnx_filename'))
    #     if not os.path.exists(onnx_checkpoint):
    #         self.exporter.export_model_onnx()
    #     sample_image_name = self.config['dummy_valid_list'][0]
    #     sample_image_path = os.path.join(self.image_path, sample_image_name)
    #     self.inference.test_onnx(sample_image_path, onnx_checkpoint)
    #     metric = self.inference.validate_models(run_type='onnx', onnx_checkpoint=onnx_checkpoint)
    #     self.assertGreaterEqual(metric, self.config['target_metric'])

    # def test_ir_inference(self):
    #     model_dir = os.path.split(self.config['checkpoint'])[0]
    #     onnx_checkpoint = os.path.join(model_dir, self.config.get('model_name_onnx'))
    #     if not os.path.exists(onnx_checkpoint):
    #         self.exporter.export_model_onnx()
    #     metric = self.inference.validate_models(run_type='openvino', onnx_checkpoint=onnx_checkpoint)
    #     self.assertGreaterEqual(metric, self.config['target_metric'])

