# OpenVINO Training eXtensions (OTX)

# Overview
OpenVINO Training eXtensions (OTX) is command-line interface (CLI) framework designed for low-code deep learning model training. OTX lets developers train/inference/optimize models with a diverse combination of model architectures and learning methods. For example, users can train a ResNet18 model for detection in a semi-supervised manner without worrying about setting these options manually. `otx build` and `otx train` CLI will automatically analyze users' dataset and do necessary tasks for training the model with best configuration. OTX provides the following features

- Provide a set of pre-configured models for quick start
    - `otx find` can help you quickly finds the best pre-configured models for common task types like classification, detection and segmentation.
- Configure and train a model from torchvision, [OpenVINO Model Zoo (OMZ)](https://github.com/openvinotoolkit/open_model_zoo)
    - `otx build` can help you configure your own model configuration based on torchvision and OpenVINO Model Zoo models. You can replace backbones and heads for your own preference.
- Provide many learning methods like supervised, semi-supervised, imbalanced-learn, class-incremental, self-supervised representation learning
    - `otx train` can help you automatically identify the best learning methods for your data and model. What you really need to do is to set your data (if you don't specify a model, the system will automatically set the best model for you as well). For example, if your dataset has long-tailed and partially-annoated bounding box annotations, OTX auto-configurator will choose a semi-supervised imbalanced-learning model with best configurable parameters. 
- Fast automl of hyper-parameters
    - OTX has an integrated, very efficient hyper-parameter optimization module. So, you don't need to worry about setting the right hyper-parameters. Through dataset proxy and hyper-parameter optimizer auto-configurator, you can get much faster hyper-parameter optimization, which is also configurable based on your computational budget.
- Support most of widely-used annotation formats
    - OTX uses [datumaro](https://github.com/openvinotoolkit/datumaro), which is designed for dataset building and transformation, as a default interface for dataset management. All supported formats by datumaro are also consumable by OTX without the need of explicit data conversion. If you wnat to build your own custom dataset format, you can do this via datumaro CLI and API.


# Component view
The diagram below illustrates key components in a high-level design. OTX has two layers: CLI and API. CLI is implemented as a combination of APIs, which consists of components for model management, dataset import/export/iteration, and job interface for inference/train/optimize. A recipe is a concrete combination of model and job interfaces, and takes dataset for best model training. OTX provides both pre-built recipes and models as well as mechanisms to build your own ones in a low-code manner.

<figure>
<img src="docs/source/_images/high.png" width="800px">
<figcaption></figcaption>
</figure>

# Architecturally-significant use cases
The diagram below shows architecturally-significant use cases. Basically, OTX use cases covers from model building to final deployment. Users can use OTX CLI for low-code model and training recipe building, training with dataset, and optimizing model performance for final deployment. Users can generate a self-contained package for exportable codes with a single command. API users can customize the built-in workflow for their purpose. For example, new learning methods can be registered or extending APIs for supporting other backends like TensorFlow or ONNX.

<figure>
<img src="docs/source/_images/uc.png" width="600px">
<figcaption></figcaption>
</figure>

# Terminology
- **Model** is a collection of backbone, neck, and (task-specific) head which is ready to be consumed by training *Recipe* if *Dataset* is provided.
- **Task** is a type of vision prediction consisting of classification, regression, detection, instance segmentation, anomaly detection, and so on. *Task* and *Model* are usually highly-related, and *Task* determines how the *Model* look like, especially for neck and head architecture.
- **Job** is a type of computational workload consisting of inference, optimization, training, and other openvino related ones. Job is an elementary operation which can be called by a specific *Recipe*
- **Recipe** is the collection of *Job*, *Model*, and training related implementation like data augmentation strategy, batch composition policy, job cascading/switching, which are not bound for specific *Dataset*. For example, semi-supervised learning can be a recipe-level implementation since this requires multiple jobs as well as on-the-fly data scheduling with pseudo-labels.
- **Config** is a descriptor of a specific *Recipe* based on Python dictionary type. It can be instantiated from YAML or XML formatted files.
- **Dataset** corresponds to something like torch.utils.data.Dataset which can be used in torch.utils.data.DataLoader. Each dataset can be created for specific purposes like training or evaluation. *Dataset* should provide an iterator for constructing a batch for training and evaluation.


# OTX CLI
- `otx find`    
- `otx build`
- `otx train`
- `otx optimize`
- `otx export`
- `otx demo`
- `otx deploy`

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
|`otx eval`|`<path-to-ckpt> <data.yaml>`[^1]|`Recipe.eval(data)`|evaluation (if there is ground-truth) and/or prediction result on *data*|
|   |`<path-to-ckpt> <path-to-input>`|`Recipe.eval(input)`|prediction result on *input* (typically image)|
|   |~~`<path-to-ckpt> <device>`~~|~~`Recipe.eval(device)`~~|~~prediction result on *device* (typically camera)~~|
|`otx optimize`|`<path-to-ckpt> <type-str> <data.yaml>`|`Recipe.optimize(type, data)`|run *type* of optimization to the given *ckpt*[^2]|
|`otx export`|`<path-to-ckpt>`|`Recipe.export()`|export *ckpt* model to OpenVINO IR (xml and bin files)|
|~~`otx demo`~~|   |  |  |
|`otx deploy`|`<path-to-ckpt>`|`Recipe.deploy()`|create a zip file containing all exportable codes in a self-contained manner. This includes both PyTorch and OpenVINO inferences|

[^1]ckpt file has a metadata for recipe information. For more information, refer to sequence diagram in OTX API.

[^2]For more information on optimization types, refer to [OpenVINO Model Optimization Guide](https://docs.openvino.ai/latest/openvino_docs_model_optimization_guide.html).

# `otx build` and `otx train` are all you need (recipe auto-configurator)

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

### API sequence for training
<figure>
<img src="docs/source/_images/seq-train.png" width="1200px">
<figcaption></figcaption>
</figure>

### API sequence for evaluation (or inference)
<figure>
<img src="docs/source/_images/seq-eval.png" width="1200px">
<figcaption></figcaption>
</figure>

### API sequence for model and recipe building
<figure>
<img src="docs/source/_images/seq-build.png" width="800px">
<figcaption></figcaption>
</figure>