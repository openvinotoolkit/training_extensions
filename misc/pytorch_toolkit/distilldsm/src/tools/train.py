import argparse
import numpy as np
from src.utils.filenames import generate_filenames, load_bias, load_sequence
from src.utils.trainer import run_pytorch_training
from src.utils.dataset import WholeVolumeSegmentationDataset
from src.utils.utils import load_json, in_config, dump_json
from src.tools.inference import format_parser as format_prediction_args
from src.tools.inference import run_inference
from src.utils.script_utils import get_machine_config, add_machine_config_to_parser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", required=True,
                        help="JSON configuration file specifying the parameters for model training.")
    parser.add_argument("--model_filename",
                        help="Location to save the model during and after training. If this filename exists "
                             "prior to training, the model will be loaded from the filename.",
                        required=True)
    parser.add_argument("--training_log_filename",
                        help="CSV filename to save the to save the training and validation results for each epoch.",
                        required=True)
    parser.add_argument("--fit_gpu_mem", type=float,
                        help="Specify the amount of gpu memory available on a single gpu and change the image size to "
                             "fit into gpu memory automatically. Will try to find the largest image size that will fit "
                             "onto a single gpu. The batch size is overwritten and set to the number of gpus available."
                             " The new image size will be written to a new config file ending named "
                             "'<original_config>_auto.json'. This option is experimental and only works with the UNet "
                             "model. It has only been tested with gpus that have 12GB and 32GB of memory.")
    parser.add_argument("--subjects_filename",
                        help="JSON configuration file specifying the ids for model training.", default="", type=str)
    parser.add_argument("--model_name",
                        help="JSON configuration file specifying the ids for model training.", default="", type=str)
    parser.add_argument("--group_average_filenames")
    add_machine_config_to_parser(parser)
    subparsers = parser.add_subparsers(help="sub-commands", dest='sub_command')
    prediction_parser = subparsers.add_parser(name="predict",
                                              help="Run prediction after the model has finished training")
    format_prediction_args(prediction_parser, sub_command=True)
    args = parser.parse_args()

    return args


def check_hierarchy(config):
    if in_config("labels", config["sequence_kwargs"]) and in_config("use_label_hierarchy", config["sequence_kwargs"]):
        config["sequence_kwargs"].pop("use_label_hierarchy")
        labels = config["sequence_kwargs"].pop("labels")
        new_labels = list()
        while len(labels):
            new_labels.append(labels)
            labels = labels[1:]
        config["sequence_kwargs"]["labels"] = new_labels


def compute_unet_number_of_voxels(window, channels, n_layers):
    n_voxels = 0
    for i in range(n_layers):
        n_voxels = n_voxels + ((1/(2**(3*i))) * window[0] * window[1] * window[2] * channels * 2**i * 2)
    return n_voxels


def compute_window_size(step, step_size, ratios):
    step_ratios = np.asarray(ratios) * step * step_size
    mod = np.mod(step_ratios, step_size)
    return np.asarray(step_ratios - mod + np.round(mod / step_size) * step_size, dtype=int)


def update_config_to_fit_gpu_memory(config, n_gpus, gpu_memory, output_filename, voxels_per_gb=17000000.0,
                                    ratios=(1.22, 1.56, 1.0)):
    max_voxels = voxels_per_gb * gpu_memory
    n_layers = len(config["model_kwargs"]["encoder_blocks"])
    step_size = 2**(n_layers - 1)
    step = 1
    window = compute_window_size(step, step_size, ratios)
    n_voxels = compute_unet_number_of_voxels(window, config["model_kwargs"]["base_width"], n_layers)
    while n_voxels <= max_voxels:
        step = step + 1
        window = compute_window_size(step, step_size, ratios)
        n_voxels = compute_unet_number_of_voxels(window, config["model_kwargs"]["base_width"], n_layers)
    window = compute_window_size(step - 1, step_size, ratios).tolist()
    print("Setting window size to {} x {} x {}".format(*window))
    print("Setting batch size to", n_gpus)
    config["window"] = window
    config["model_kwargs"]["input_shape"] = window
    config["batch_size"] = n_gpus
    config["validation_batch_size"] = n_gpus
    print("Writing new configuration file:", output_filename)
    dump_json(config, output_filename)


def main():
    import nibabel as nib
    nib.imageglobals.logger.level = 40
    namespace = parse_args()
    print("Config: ", namespace.config_filename)
    config = load_json(namespace.config_filename)
    if namespace.subjects_filename != "":
        config["subjects_filename"] = namespace.subjects_filename
    if namespace.model_name != "":
        config["model_name"] = namespace.model_name
    print(config["subjects_filename"], config["model_name"])

    if "metric_names" in config and not config["n_outputs"] == len(config["metric_names"]):
        raise ValueError("n_outputs set to {}, but number of metrics is {}.".format(config["n_outputs"],
                                                                                    len(config["metric_names"])))

    print("Model: ", namespace.model_filename)
    print("Log: ", namespace.training_log_filename)
    system_config = get_machine_config(namespace)

    if namespace.fit_gpu_mem and namespace.fit_gpu_mem > 0:
        update_config_to_fit_gpu_memory(config=config, n_gpus=system_config["n_gpus"], gpu_memory=namespace.fit_gpu_mem,
                                        output_filename=namespace.config_filename.replace(".json", "_auto.json"))


    model_metrics = []
    if config['skip_validation']:
        metric_to_monitor = "loss"
        groups = ("training",)
    else:
        metric_to_monitor = "val_loss"
        groups = ("training", "validation")
        

    for name in groups:
        key = name + "_filenames"
        if key not in config:
            config[key] = generate_filenames(config, name, system_config)

    if "sequence" in config:
        sequence_class = load_sequence(config["sequence"])
    else:
        sequence_class = WholeVolumeSegmentationDataset

    if "bias_filename" in config and config["bias_filename"] is not None:
        bias = load_bias(config["bias_filename"])
    else:
        bias = None

    check_hierarchy(config)

    if in_config("add_contours", config["sequence_kwargs"], False):
        config["n_outputs"] = config["n_outputs"] * 2

    run_pytorch_training(config, namespace.model_filename, namespace.training_log_filename,
                            sequence_class=sequence_class,model_metrics=model_metrics,
                            metric_to_monitor=metric_to_monitor, bias=bias, **system_config)

    if namespace.sub_command == "predict":
        run_inference(namespace)


if __name__ == '__main__':
    main()
