# OpenVINO Training eXtensions (OTX)

# Overview
OpenVINO Training eXtensions (OTX) is command-line interface (CLI) framework designed for low-code deep learning model training. OTX lets developers train/inference/optimize models with a diverse combination of model architectures and learning methods. For example, users can train a ResNet18-based SSD ([Single Shot Detection](https://arxiv.org/abs/1512.02325)) model in a semi-supervised manner without worrying about setting a configuration manually. `otx build` and `otx train` commands will automatically analyze users' dataset and do necessary tasks for training the model with best configuration. OTX provides the following features:

- Provide a set of pre-configured models for quick start
    - `otx find` helps you quickly finds the best pre-configured models for common task types like classification, detection and segmentation.
- Configure and train a model from torchvision, [OpenVINO Model Zoo (OMZ)](https://github.com/openvinotoolkit/open_model_zoo)
    - `otx build` can help you configure your own model configuration based on torchvision and OpenVINO Model Zoo models. You can replace backbones and heads for your own preference.
- Provide many learning methods like supervised, semi-supervised, imbalanced-learn, class-incremental, self-supervised representation learning
    - `otx build` helps you automatically identify the best learning methods for your data and model. All you need to do is to set your data (if you don't specify a model, the system will automatically set the best model for you as well). For example, if your dataset has long-tailed and partially-annotated bounding box annotations, OTX auto-configurator will choose a semi-supervised imbalanced-learning method, a proper model with best default parameters. 
- Efficient hyper-parameter optimization
    - OTX has an integrated, efficient hyper-parameter optimization module. So, you don't need to worry about searching right hyper-parameters. Through dataset proxy and built-in hyper-parameter optimizer, you can get much faster hyper-parameter optimization compared to other off-the-shelf tools.
- Support most of widely-used annotation formats
    - OTX uses [datumaro](https://github.com/openvinotoolkit/datumaro), which is designed for dataset building and transformation, as a default interface for dataset management. All supported formats by datumaro are also consumable by OTX without the need of explicit data conversion. If you want to build your own custom dataset format, you can do this via datumaro CLI and API.


# Component view
The diagram below illustrates key components in a high-level design (the colors correspond to [API class diagram](#otx-api)). OTX has two layers: CLI and API. CLI is implemented as a combination of APIs, which consists of components for model management, dataset import/export/iteration, and job interface for inference/train/optimize. A recipe is a concrete combination of model and job interfaces, and takes dataset for training a model. OTX provides both pre-built recipes and models as well as mechanisms to build your own ones in a low-code manner. OTX dataset API uses datumaro as a base class, and thus supports diverse annotation formats.

<figure>
<img src="docs/source/_images/high.png" width="750px">
<figcaption></figcaption>
</figure>

# Architecturally-significant use cases
The diagram below shows architecturally-significant use cases. Basically, OTX use cases cover from model building to final deployment. Users can use OTX CLI for finding/building models and recipes, training with dataset, and optimizing model performance for final deployment. Users also can generate a self-contained package for exportable codes with a single command. API users can customize the built-in workflow for their purpose. For example, new learning methods can be registered or users can extend APIs for supporting other backends like TensorFlow or ONNX.

<figure>
<img src="docs/source/_images/uc.png" width="550px">
<figcaption></figcaption>
</figure>

# Terminology
- **Model** is a collection of backbone, neck, and (task-specific) head which is ready to be consumed by training *Recipe* if *Dataset* is provided.
- **Task** is a type of vision prediction consisting of classification, regression, detection, instance segmentation, anomaly detection, and so on. *Task* and *Model* are usually highly-related, and *Task* determines how the *Model* looks like, especially for neck and head architecture.
- **Job** is a type of computational workload like inference, optimization, training, and other openvino related ones. Job is an elementary operation which can be called by a specific *Recipe*. Job implementation depends on ML framework such as PyTorch or TensorFlow.
- **Recipe** is the collection of *Job*, *Model*, and training related implementation like data augmentation strategy, batch composition policy, job cascading/switching, which are not bound for specific *Dataset*. For example, semi-supervised learning can be a recipe-level implementation since this requires multiple jobs as well as on-the-fly data scheduling with pseudo-labels.
- **Config** is a descriptor of a specific *Recipe* based on Python dictionary type. It can be instantiated from YAML or XML formatted files.
- **Dataset** corresponds to something like torch.utils.data.Dataset which can be used in torch.utils.data.DataLoader. Each dataset can be created for specific purposes like training or evaluation. *Dataset* should provide an iterator for constructing a batch for training and evaluation.


# OTX CLI
The table below shows OTX CLIs and their API mappings. In most cases, users simply put their dataset and run train CLI for training a model.

The following command will analyze a dataset and auto-generate a training configuration (recipe.yaml).
```
otx build <data.yaml>
```
Users can train a model with the generated tranining configuration.
```
otx build <recipe.yaml> <data.yaml>
```

|command|argument|API mapping|result(return)|
|---|---|---|---|
|`otx find`|`--task`|`Registry.find_tasks()`|list of supported task types (classification, detection, etc.)|
|   |`--recipe  <task-type-str>`|`Registry.find_recipes(task)`|list of recipes for the given task (class-incremental, semi-supervised etc.)|
|   |`--model <recipe.yaml>`|`Registry.find_models(recipe)`|list of models for the given recipe (e.g. resnet18-atss-det)|
|   |`--backbone <model.yaml>`|`Registry.find_backbones(model)`|list of backbones for the given model (e.g. resnet18)|
|   |`--all`|`Registry.find_*()`|recursive call of all above|
|`otx build`|`--model <model.yaml> --backbone <backbone.yaml>`|`Builder.build_model(model, backbone)`|*backbone*_model.yaml will be saved to user’s workspace after replacing model.backbone attribute to the given *backbone*|
|   |`--recipe <data.yaml>`|`Builder.build_recipe(data)`|analyze data and generate best recipe yaml automatically and dump it to user’s workspace|
|   |`--recipe <recipe.yaml> --model <model.yaml>`|`Builder.build_recipe(recipe, model)`|*model*_recipe.yaml will be added to user’s workspace after updating recipe.model attribute to the given *model*|
|`otx train`|`<recipe.yaml> <data.yaml>`|`Recipe.train(data)`|path to model ckpt trained on *data*|
|`otx eval`|`<path-to-ckpt> <data.yaml>`<sup>[1](#note1)</sup>|`Recipe.eval(data)`|evaluation (if there is ground-truth) and/or prediction result on *data*|
|   |`<path-to-ckpt> <path-to-input>`|`Recipe.eval(input)`|prediction result on *input* (typically image)|
|   |~~`<path-to-ckpt> <device>`~~|~~`Recipe.eval(device)`~~|~~prediction result on *device* (typically camera)~~|
|`otx optimize`|`<path-to-ckpt> <type-str> <data.yaml>`|`Recipe.optimize(type, data)`|run *type* of optimization to the given *ckpt*<sup>[2](#note2)</sup>|
|`otx export`|`<path-to-ckpt>`|`Recipe.export()`|export *ckpt* model to OpenVINO IR (xml and bin files)|
|~~`otx demo`~~|   |  |  |
|`otx deploy`|`<path-to-ckpt>`|`Recipe.deploy()`|create a zip file containing all exportable codes in a self-contained manner. This includes both PyTorch and OpenVINO inferences|

<a name="#note1"><sup>1</sup></a>ckpt file has a metadata for recipe information. For more information, refer to sequence diagram in OTX API.

<a name="#note2"><sup>2</sup></a>For more information on optimization types, refer to [OpenVINO Model Optimization Guide](https://docs.openvino.ai/latest/openvino_docs_model_optimization_guide.html).

## `otx build` is all you need (recipe auto-configurator)
One of unique characteristics of OTX features is recipe auto-configurator. Unlike other AI training toolkits, OTX provides an automated way of setting training-related stuffs by analyzing users' dataset. This includes task type, model architecture, learning method and hyper-parameters. Among them, hyper-parameters are tuned further by automatic hyper-parameter optimization module in later stages of training. Users can directly edit recipe.yaml file for customization. This auto-configurator also provides estimated times for training and inference so that users can review the configuration before triggering actual jobs.

<figure>
<img src="docs/source/_images/autoconfig.png" width="800px">
<figcaption></figcaption>
</figure>

# OTX API

OTX API consists of three component groups. The first group defines entities for constructing CLIs, registration of models and recipes, and classes for dataset, which could be used and/or modified by users. The second group is for core components like defining jobs and model architecture, which are intended for internal execution. And the third group specifies actual backend implementations that depend on platforms like PyTorch, MMCV, and OpenVINO. The figure below illustrates class diagram, which partially shows backend implementations.
<figure>
<img src="docs/source/_images/class.png" width="800px">
<figcaption></figcaption>
</figure>

- Dataset class inherits datumaro.components.Dataset and works as a base class for other dataset classes from PyTorch and MMCV. For example, MMClsDataset is an adapter for dataset in MMClssification framework. Dataset class provides a rich interface for many task types like classification, detection, segmentation, and so on with widely-used formats like MS-COCO, LabelMe, Open Images, and ImageNet.
- Registry class is designed for managing recipes and models either from built-in or user workspace. When initialization, it retrieves all recipes and models from YAML repo and user workspace and loads them into memory. In Registry, there are two resident types, models and recipes. Models are grouped by recipe specification and recipes are grouped by task specification. New models and recipes can be automatically (de)registered if there are untracked changes in user workspace.
- core.IModel provides a basic interface for creating a model from model configration file, which defines backbone, neck, (task-specific) head, and their names, connections, and parameters. To support jobs with specific checkpoints, this class has checkpoint as a member variable.
- core.IJob provides a set of interfaces for computational workloads like training and inference. It defines a minimal executable unit in AI workflow. The purpose of job is specified within the class. Every job must implement `run()` method for execution. When model instance is required for specific job, the caller (*Recipe*) will pass the information as an argument. core.IJob depends on core.IModel and api.Dataset.
- Backends is a set of components that depend on specific ML framework. OTX API uses Pytorch, MMCV, and OpenVINO as main backends for implementing actual models and recipes. For example, OpenVINO jobs can be implemented in its corresponding backend as below.
    - OpenVINO backends: OpenVINOJob provides a base class for all OpenVINO-related jobs, and OpenVINOInference and OpenVINOExporter classes provides actual implementations. Since OpenVINOOptimizer requires not only OpenVINO but also AI training framework like Pytorch, it inherits both OpenVINOJob and TorchJob simultaneously.
<figure>
<img src="docs/source/_images/backend-ov.png" width="500px">
<figcaption></figcaption>
</figure>

## API sequence for training
<figure>
<img src="docs/source/_images/seq-train.png" width="1200px">
<figcaption></figcaption>
</figure>

## API sequence for evaluation (or inference)
<figure>
<img src="docs/source/_images/seq-eval.png" width="1200px">
<figcaption></figcaption>
</figure>

## API sequence for model and recipe building
<figure>
<img src="docs/source/_images/seq-build.png" width="800px">
<figcaption></figcaption>
</figure>