Utilize OpenVINO™ Training Extensions APIs in your project
==========================================================

Besides CLI functionality, The OpenVINO™ Training Extension provides APIs that help developers to integrate OpenVINO™ Training Extensions models into their projects.
This tutorial intends to show how to create a dataset, model and use all of the CLI functionality through APIs.

For demonstration purposes we will use the Object Detection SSD model with `WGISD <https://github.com/thsant/wgisd>`_ public dataset as we did for the :doc:`CLI tutorial <../base/how_to_train/detection>`.

.. note::

    To start with we need to `install OpenVINO™ Training Extensions <https://github.com/openvinotoolkit/training_extensions/blob/develop/QUICK_START_GUIDE.md#setup-openvino-training-extensions>`_.

*******************
Dataset preparation
*******************

1. Clone a repository
with `WGISD dataset <https://github.com/thsant/wgisd>`_.

.. code-block::

  cd data
  git clone https://github.com/thsant/wgisd.git
  cd wgisd
  git checkout 6910edc5ae3aae8c20062941b1641821f0c30127

2. We need to rename annotations to
be distinguished by OpenVINO™ Training Extensions Datumaro manager:

.. code-block::

    mv data images && mv coco_annotations annotations && mv annotations/train_bbox_instances.json instances_train.json  && mv annotations/test_bbox_instances.json instances_val.json

Now it is all set to use this dataset inside OpenVINO™ Training Extensions

**********************************
Model template and dataset loading
**********************************

Let's import the necessary modules:

.. code-block::

    import cv2
    import numpy as np

    from otx.api.configuration.helper import create as create_parameters_from_parameters_schema
    from otx.api.entities.inference_parameters import InferenceParameters
    from otx.api.entities.model import ModelEntity
    from otx.api.entities.resultset import ResultSetEntity
    from otx.api.entities.subset import Subset
    from otx.api.entities.task_environment import TaskEnvironment
    from otx.api.usecases.tasks.interfaces.export_interface import ExportType
    from otx.api.usecases.tasks.interfaces.optimization_interface import OptimizationType
    from otx.api.entities.optimization_parameters import OptimizationParameters
    from otx.core.data.adapter import get_dataset_adapter
    from otx.cli.registry import Registry
    from otx.cli.utils.importing import get_impl_class
    from otx.cli.utils.io import read_label_schema, read_model
    from otx.cli.tools.utils.demo.visualization import draw_predictions

We will use the SSD object detection model in that tutorial. Let's initiate the model template:

.. code-block::

    templates_dir = 'src/otx/algorithms'
    registry = Registry(templates_dir)
    model_template = registry.get('Custom_Object_Detection_Gen3_SSD')

Derive hyperparameters from the model template. We can change essential hyperparameters for your configuration:

.. code-block::

    hyper_parameters = model_template.hyper_parameters.data
    hyper_parameters = create_parameters_from_parameters_schema(hyper_parameters)
    # print hyperparameters to see which one is available to modify
    for p in hyper_parameters.learning_parameters.parameters:
        print(f'{p}: {getattr(hyper_parameters.learning_parameters, p)}')

    # Let's modify the batch size and number of epochs to train
    hyper_parameters.learning_parameters.batch_size = 8
    hyper_parameters.learning_parameters.num_iters = 5

The next step is to set up a dataset:

.. code-block::

    dataset_adapter = get_dataset_adapter(task_type = model_template.task_type,
                                          # set a path to the root folder of the wgisd repository
                                          train_data_roots="./wgisd",
                                          val_data_roots="./wgisd",
                                          test_data_roots="./wgisd")
    dataset, labels_schema = dataset_adapter.get_otx_dataset(), dataset_adapter.get_label_schema()


***********************************
Set up the environment and the task
***********************************

.. code-block::

    Task = get_impl_class(model_template.entrypoints.base)

    environment = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=labels_schema,
        model_template=model_template)

    task = Task(task_environment=environment)

*****************************
Training, Validation, Export
*****************************

Create an output model and start actual training:

.. code-block::

    output_model = ModelEntity(
        dataset,
        environment.get_model_configuration(),
    )

    task.train(dataset, output_model)

To perform validation we need to infer our model on the validation dataset, create ``ResultSetEntity`` and save to this entity inference results:

.. code-block::

    validation_dataset = dataset.get_subset(Subset.VALIDATION)

    predicted_validation_dataset = task.infer(
        validation_dataset.with_empty_annotations(),
        InferenceParameters(is_evaluation=True))

    resultset = ResultSetEntity(
        model=output_model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )

    task.evaluate(resultset)

    # print or save validation results
    print(resultset.performance)

To export our model to OpenVINO™ IR format we need to create output model and run exporting task.
To validate the OpenVINO™ IR model, we need to create an openvino task first and then run the evaluation procedure:

.. code-block::

    exported_model = ModelEntity(
        dataset,
        environment.get_model_configuration(),
    )
    task.export(ExportType.OPENVINO, exported_model)

    # substitute the model in the environment with exported_model
    environment.model = exported_model

    # create an openvino task
    ov_task = get_impl_class(model_template.entrypoints.openvino)(environment)

    # validation
    predicted_validation_dataset = ov_task.infer(
        validation_dataset.with_empty_annotations(),
        InferenceParameters(is_evaluation=True))

    resultset = ResultSetEntity(
        model=output_model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    ov_task.evaluate(resultset)

    # print or save the result
    print(resultset.performance)

************
Optimization
************

To run the optimization with POT on the OpenVINO™ IR model, we need to create an output model and run the optimization procedure:

.. code-block::

    optimized_model = ModelEntity(
                dataset,
                environment.get_model_configuration(),
            )

    ov_task.optimize(
        OptimizationType.POT,
        dataset,
        optimized_model,
        OptimizationParameters())

To run the NNCF accuracy-aware training, return model in the environment back, create NNCF task, output model and run optimization procedure:

.. code-block::

    # return PyTorch model back
    environment.model = output_model

    # create an NNCF task based on our environment
    nncf_task = get_impl_class(model_template.entrypoints.nncf)(environment)

    # create output model
    optimized_nncf_model = ModelEntity(
                dataset,
                environment.get_model_configuration(),
            )

    nncf_task.optimize(
        OptimizationType.NNCF,
        dataset,
        optimized_nncf_model,
        OptimizationParameters())

You can validate the optimized model as the usual model. For example for the NNCF model it will look like this:

.. code-block::

    # NNCF task inference
    predicted_validation_dataset = nncf_task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True))

    # ResultSetEntity creating with optimized_nncf_model
    resultset = ResultSetEntity(
        model=optimized_nncf_model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )

    # evaluation
    nncf_task.evaluate(resultset)

    # print or save the result
    print(resultset.performance)


**************************************
Load the model and use it for any data
**************************************

Let's assume, that we have already trained the model and we want to use it in our project. The simple steps on how to load the model and infer it on custom images are presented below.

.. code-block::

    # path to the trained OpenVINO™ Training Extensions weights, can be PyTorch .pth or OpenVINO™ IR .xml
    weights_path = "path/to/trained/weights"

    # create new environment
    environment = TaskEnvironment(
            model=None,
            hyper_parameters=hyper_parameters,
            label_schema=read_label_schema(weights_path),
            model_template=template,
        )

    # read the model and assign it to our environment
    environment.model = read_model(environment.get_model_configuration(), weights_path, None)

    # create task
    task_class = (get_impl_class(template.entrypoints.openvino)
                      if weights_path.endswith(".xml")
                      else get_impl_class(template.entrypoints.base))

    task = task_class(task_environment=environment)

Open some images, convert them to a small dataset, infer and get the annotations from our model.
We can convert these steps to function and use it in a loop with multiple images/frames from video:

.. code-block::

    def get_predictions(task, frame):
        """Returns list of predictions made by task on a frame."""

        empty_annotation = AnnotationSceneEntity(annotations=[], kind=AnnotationSceneKind.PREDICTION)

        item = DatasetItemEntity(
            media=Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            annotation_scene=empty_annotation,
        )

        dataset = DatasetEntity(items=[item])

        start_time = time.perf_counter()
        predicted_validation_dataset = task.infer(
            dataset,
            InferenceParameters(is_evaluation=True),
        )
        elapsed_time = time.perf_counter() - start_time
        item = predicted_validation_dataset[0]
        return item.get_annotations(), elapsed_time

    for img in images_list:
        # use our function to get predictions
        predictions = get_predictions(task, img)

        # we also can draw predictions on the image and visualize the result
        img = draw_predictions(template.task_type, predictions, img, args.fit_to_size)


That's it. Now, we can use OpenVINO™ Training Extensions APIs to create, train, and deploy deep learning models using the OpenVINO™ Training Extension.