"""Model explain demonstration tool."""

# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from pathlib import Path

from otx.algorithms.common.utils.logger import get_logger
from otx.api.entities.explain_parameters import ExplainParameters
from otx.api.entities.task_environment import TaskEnvironment
from otx.cli.manager import ConfigManager
from otx.cli.utils.importing import get_impl_class
from otx.cli.utils.io import (
    get_explain_dataset_from_filelist,
    get_image_files,
    read_label_schema,
    read_model,
    save_saliency_output,
)
from otx.cli.utils.nncf import is_checkpoint_nncf
from otx.cli.utils.parser import (
    add_hyper_parameters_sub_parser,
    get_parser_and_hprams_data,
)

logger = get_logger()

ESC_BUTTON = 27
SUPPORTED_EXPLAIN_ALGORITHMS = ["activationmap", "eigencam", "classwisesaliencymap"]

# pylint: disable=too-many-locals


def get_args():
    """Parses command line arguments."""
    parser, hyper_parameters, params = get_parser_and_hprams_data()

    parser.add_argument(
        "--explain-data-roots",
        required=True,
        help="Comma-separated paths to explain data folders.",
    )
    parser.add_argument(
        "--save-explanation-to",
        default="saliency_dump",
        help="Output path for explanation images.",
    )
    parser.add_argument(
        "--load-weights",
        required=True,
        help="Load model weights from previously saved checkpoint.",
    )
    parser.add_argument(
        "--explain-algorithm",
        default="ClassWiseSaliencyMap",
        help=f"Explain algorithm name, currently support {SUPPORTED_EXPLAIN_ALGORITHMS}."
        "For Openvino task, default method will be selected.",
    )
    parser.add_argument(
        "--process-saliency-maps",
        action="store_true",
        help="Processing of saliency map includes (1) resizing to input image resolution and (2) applying a colormap."
        "Depending on the number of targets to explain, this might take significant time.",
    )
    parser.add_argument(
        "--explain-all-classes",
        action="store_true",
        help="Provides explanations for all classes. Otherwise, explains only predicted classes."
        "This feature is supported by algorithms that can generate explanations per each class.",
    )
    parser.add_argument(
        "--overlay-weight",
        type=float,
        default=0.5,
        help="Weight of the saliency map when overlaying the input image with saliency map.",
    )
    add_hyper_parameters_sub_parser(parser, hyper_parameters, modes=("INFERENCE",))
    override_param = [f"params.{param[2:].split('=')[0]}" for param in params if param.startswith("--")]

    return parser.parse_args(), override_param


def _log_prior_to_saving(args, num_images):
    logger.info("Explain report:")
    if args.process_saliency_maps:
        logger.info(
            "Postprocessing applied. (1) saliency maps resized to the input image resolution "
            "and (2) color map applied."
        )
    else:
        logger.info(
            "No postprocessing applied. Raw low-resolution saliency maps saved as .tiff format images. "
            "Use --process-saliency-maps to apply postprocessing to saliency maps."
        )

    if args.explain_all_classes:
        logger.info(f"Saliency maps generated for each class, per each of {num_images} images.")
    else:
        logger.info(
            "Saliency maps generated ONLY for predicted class(es), if any. "
            "Use --explain-all-classes flag to generate explanations for all classes."
        )


def _log_after_saving(explain_predicted_classes, explained_image_counter, args, num_images):
    if explain_predicted_classes and explained_image_counter == 0:
        logger.info(
            "No predictions were made for provided model-data pair -> no saliency maps generated. "
            "Please adjust training pipeline or use different model-data pair."
        )
    if explained_image_counter > 0:
        logger.info(
            f"Saliency maps saved to {args.save_explanation_to} for {explained_image_counter} "
            f"out of {num_images} images."
        )


def main():
    """Main function that is used for model explanation."""

    args, override_param = get_args()

    config_manager = ConfigManager(args, mode="explain")
    # Auto-Configuration for model template
    config_manager.configure_template()

    # Update Hyper Parameter Configs
    hyper_parameters = config_manager.get_hyparams_config(override_param)

    # Get classes for Task, ConfigurableParameters and Dataset.
    template = config_manager.template
    if any(args.load_weights.endswith(x) for x in (".bin", ".xml", ".zip")):
        task_class = get_impl_class(template.entrypoints.openvino)
    elif args.load_weights.endswith(".pth"):
        if is_checkpoint_nncf(args.load_weights):
            task_class = get_impl_class(template.entrypoints.nncf)
        else:
            task_class = get_impl_class(template.entrypoints.base)
    else:
        raise ValueError(f"Unsupported file: {args.load_weights}")

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=read_label_schema(args.load_weights),
        model_template=template,
    )

    environment.model = read_model(environment.get_model_configuration(), args.load_weights, None)
    task = task_class(task_environment=environment)

    if args.explain_algorithm.lower() not in SUPPORTED_EXPLAIN_ALGORITHMS:
        raise NotImplementedError(
            f"{args.explain_algorithm} currently not supported. \
            Currently only support {SUPPORTED_EXPLAIN_ALGORITHMS}"
        )
    if not Path(args.save_explanation_to).exists():
        Path(args.save_explanation_to).mkdir(parents=True)

    image_files = get_image_files(args.explain_data_roots)
    dataset_to_explain = get_explain_dataset_from_filelist(image_files)
    explain_predicted_classes = not args.explain_all_classes
    explain_parameters = ExplainParameters(
        explainer=args.explain_algorithm,
        process_saliency_maps=args.process_saliency_maps,
        explain_predicted_classes=explain_predicted_classes,
    )
    explained_dataset = task.explain(
        dataset_to_explain.with_empty_annotations(),
        explain_parameters,
    )
    assert len(explained_dataset) == len(image_files)

    _log_prior_to_saving(args, len(image_files))
    explained_image_counter = 0
    for explained_data, (_, filename) in zip(explained_dataset, image_files):
        metadata_list = explained_data.get_metadata()
        if len(metadata_list) > 0:
            explained_image_counter += 1
        elif explain_predicted_classes:  # Explain only predictions
            logger.info(f"No saliency maps generated for {filename} - due to lack of confident predictions.")
        for metadata in metadata_list:
            saliency_data = metadata.data
            fname = f"{Path(Path(filename).name).stem}_{saliency_data.name}".replace(" ", "_")
            save_saliency_output(
                process_saliency_maps=explain_parameters.process_saliency_maps,
                img=explained_data.numpy,
                saliency_map=saliency_data.numpy,
                save_dir=args.save_explanation_to,
                fname=fname,
                weight=args.overlay_weight,
            )
    _log_after_saving(explain_predicted_classes, explained_image_counter, args, len(image_files))

    return dict(retcode=0, template=template.name)


if __name__ == "__main__":
    main()
