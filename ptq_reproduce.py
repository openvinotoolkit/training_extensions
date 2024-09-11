from otx.core.types import OTXTaskType
import json
import openvino
from jsonargparse import ArgumentParser
import nncf
from pathlib import Path
from otx.engine import Engine
import argparse
from functools import partial


def main(data_root, work_dir, model_name, task):
    ### Initialized an Engine
    engine = Engine(data_root=data_root, task=task, work_dir=work_dir)

    ### test a model before nncf optimization
    print("------------------Testing a model before nncf optimization------------------")
    metrics_before = engine.test(checkpoint=model_name)

    ### derived nncf optimization
    ov_otx_model = engine._auto_configurator.get_ov_model(model_name, label_info=engine.datamodule.label_info)
    # update pipeline for ov usage (simple numpy images)
    optimize_datamodule = engine._auto_configurator.update_ov_subset_pipeline(
                datamodule=engine.datamodule,
                subset="train",
            )

    ov_model = openvino.Core().read_model(model_name)
    # obtain ptq config
    ptq_config_from_ir = read_ptq_config_from_ir(ov_model)

    quantization_dataset = nncf.Dataset(optimize_datamodule.train_dataloader(), partial(transform_fn, ov_otx_model))  # type: ignore[attr-defined]

    compressed_model = nncf.quantize(  # type: ignore[attr-defined]
        ov_model,
        quantization_dataset,
        **ptq_config_from_ir,
    )
    output_model_path = Path(work_dir) / "optimized_model.xml"
    openvino.save_model(compressed_model, output_model_path)

    ### test a model after nncf optimization
    print("------------------Testing a model after nncf optimization------------------")
    metrics_after = engine.test(checkpoint=output_model_path)

    print("metrics_before: ", metrics_before, " metrics_after: ", metrics_after)
    print("accuracy drop: ", metrics_before["test/f1-score"] - metrics_after["test/f1-score"])


def transform_fn(model, data_batch):
    """Data transform function for PTQ."""
    np_data = model._customize_inputs(data_batch)
    image = np_data["inputs"][0]

    core_model = model.model
    resized_image = core_model.resize(image, (core_model.w, core_model.h))
    resized_image = core_model.input_transform(resized_image)
    return core_model._change_layout(resized_image)  # noqa: SLF001


def read_ptq_config_from_ir(ov_model) -> dict:
    """Generates the PTQ (Post-Training Quantization) configuration from the meta data of the given OpenVINO model.

    Args:
        ov_model (Model): The OpenVINO model in which the PTQ configuration is embedded.

    Returns:
        dict: The PTQ configuration as a dictionary.
    """
    from nncf import IgnoredScope  # type: ignore[attr-defined]
    from nncf.common.quantization.structs import QuantizationPreset  # type: ignore[attr-defined]
    from nncf.parameters import ModelType
    from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters

    if "optimization_config" not in ov_model.rt_info["model_info"]:
        return {}

    initial_ptq_config = json.loads(ov_model.rt_info["model_info"]["optimization_config"].value)
    if not initial_ptq_config:
        return {}
    argparser = ArgumentParser()
    if "advanced_parameters" in initial_ptq_config:
        argparser.add_class_arguments(AdvancedQuantizationParameters, "advanced_parameters")
    if "preset" in initial_ptq_config:
        initial_ptq_config["preset"] = QuantizationPreset(initial_ptq_config["preset"])
        argparser.add_argument("--preset", type=QuantizationPreset)
    if "model_type" in initial_ptq_config:
        initial_ptq_config["model_type"] = ModelType(initial_ptq_config["model_type"])
        argparser.add_argument("--model_type", type=ModelType)
    if "ignored_scope" in initial_ptq_config:
        argparser.add_class_arguments(IgnoredScope, "ignored_scope", as_positional=True)

    initial_ptq_config = argparser.parse_object(initial_ptq_config)

    return argparser.instantiate_classes(initial_ptq_config).as_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OTX Inst Seg')
    parser.add_argument('--data_root', type=str, help='Path to dataset')
    parser.add_argument('--model_name', type=str, help='Name of the model')
    parser.add_argument('--work_dir', type=str, help='Path to work_dir')
    parser.add_argument('--task', type=str, help='OTX task', default="INSTANCE_SEGMENTATION")

    args = parser.parse_args()

    data_root = args.data_root
    work_dir = args.work_dir
    model_name = args.model_name
    task = OTXTaskType[args.task]
    main(data_root, work_dir, model_name, task)
