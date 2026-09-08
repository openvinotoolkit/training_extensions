<!-- markdownlint-disable MD013 MD033 MD041 MD042 -->
<div align="center">

<img src="assets/geti-header.png" alt="Geti™ - A framework to rapidly build and deploy computer vision AI models">

<br>

[![Container build](https://github.com/open-edge-platform/geti/actions/workflows/build.yaml/badge.svg)](https://github.com/open-edge-platform/geti/actions/workflows/build.yaml)
[![Codecov](https://codecov.io/gh/open-edge-platform/geti/branch/develop/graph/badge.svg?token=9HVFNMPFGD)](https://codecov.io/gh/open-edge-platform/geti)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/open-edge-platform/geti/badge)](https://securityscorecards.dev/viewer/?uri=github.com/open-edge-platform/geti)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://img.shields.io/pypi/v/getitune?logo=pypi&logoColor=white)](https://pypi.org/project/getitune/)
[![PyPI downloads](https://static.pepy.tech/badge/getitune)](https://clickpy.clickhouse.com/dashboard/getitune)

**Application**: [Install](application/docs/install.md) • [Upgrade](application/docs/upgrade.md) • [Docs](https://docs.geti.intel.com/) • [License](#license)
<br>
**Library** (`getitune`): [Install](https://docs.geti.intel.com/docs/user-guide/library/get-started/installation) • [Docs](https://docs.geti.intel.com/docs/user-guide/library/get-started/intro)

</div>

**_Geti™_** is an end-to-end platform for building AI computer vision models.

Available as a Docker container or native Windows application, Geti™ guides you through the entire model lifecycle—from dataset preparation and training to optimization and deployment.
Geti™ is optimized for fine-tuning and fast inference across the full Intel® XPU portfolio.

The Geti™ application is powered by **_getitune_**, an open-source engine for model training and optimization, which is also available standalone as a Python library. Geti™ and _getitune_ are both developed in this repository, in the [`application`](application/) and [`library`](library/) folders, respectively.

<p align="center">
 <img src="assets/model-lifecycle-infinity-light.png" width="600" alt="Geti™ - Learning Cycle"/>
</p>

> [!NOTE]
> Geti™ underwent a major revamp in v3.0, resulting in a new application that is much more lightweight and easier to install than before, while adding many new features and SOTA models. This repository contains the latest Geti™ v3. Legacy versions remain available in the old [`geti_v2`](https://github.com/open-edge-platform/geti_v2) repository; to migrate from Geti™ v2 to v3, please follow [this guide](https://docs.geti.intel.com/docs/user-guide/getting-started/installation/migration-from-geti-2x).

> [!NOTE]
> This repository previously hosted the _OpenVINO Training Extensions (OTX)_ project, now fully replaced by _getitune_. The legacy [`otx`](https://pypi.org/project/otx/) package is still available in Pypi although deprecated; it's recommended to migrate to `getitune`, which has a similar interface to `otx` and extends it with several new models.

## Quick start with Geti™

There are several ways to run Geti™, choose the one that best fits your workflow:

- Install as a [**Windows application (MSIX)**](application/docs/install.md#windows-app)
- Run as a [**Docker container**](application/docs/install.md#run-with-docker)
- Install and launch via a [**script**](application/docs/install.md#install-script)
- Build and run [**from source**](application/docs/install.md#run-from-source-for-development) (for development)

For complete, step-by-step instructions - including prerequisites, GPU/accelerator support, container and source builds, the install script, and troubleshooting - see the [Installation guide](application/docs/install.md). To update an existing Geti™ installation to a newer version, follow the [Upgrade guide](application/docs/upgrade.md).

Once Geti™ is up and running, follow the intuitive UI to train your first model.

<p align="center">
  <img src="assets/application.gif" alt="Application demo" width="80%">
</p>

> [!TIP]
> The [documentation](https://docs.geti.intel.com/) is a valuable resource to learn more about Geti™ and its capabilities.  
> New users are encouraged to read the step-by-step tutorial ["Train your first model"](https://docs.geti.intel.com/docs/user-guide/quick-start/training-your-first-model).

## Quick start with Geti Library (`getitune`)

Geti's training engine is published on PyPI and can train, optimize, and deploy models
from Python.

```bash
uv pip install "getitune[xpu]" --extra-index-url https://download.pytorch.org/whl/xpu    # for Intel® XPU acceleration
uv pip install "getitune[cuda]" --extra-index-url https://download.pytorch.org/whl/cu128    # for NVIDIA® CUDA acceleration
uv pip install getitune # CPU-only by default
```

> [!IMPORTANT]  
> The PyPI package does NOT include **Ultralytics YOLO models**, which are distributed under the [AGPL-3.0 license](https://www.ultralytics.com/license). To enable these models, build from source with the `ultralytics` extra as explained in the [getitune documentation](https://docs.geti.intel.com/docs/user-guide/library/get-started/installation).

**Discover available models and train a model in just a few lines of code:**

```python
from getitune.engine import create_engine
from getitune.utils import list_models

# Explore available models for your task
all_models = list_models()                    # List all model names
detection_models = list_models(task="DETECTION")  # Filter by task
recipes = list_models(return_recipes=True)    # Get full recipe YAML paths

# Create an engine using any model name or recipe path
engine = create_engine(
    model="efficientnet_b0",                  # model name, recipe YAML path, or exported IR/ONNX
    data="/path/to/dataset",                  # dataset directory or YAML path
    work_dir="./my_workspace",                # checkpoints and logs directory
    device="auto",                            # "auto", "cpu", "gpu", "xpu".
)

# Train and validate
engine.train(max_epochs=50)
metrics = engine.test()

# Export to OpenVINO IR (default) for deployment
exported_model_path = engine.export()

# load exported OpenVINO model
ov_engine = create_engine(model=exported_model_path, data=engine.datamodule)

# optimize the model for edge deployment
optimized_model_path = ov_engine.optimize()

# test the optimized model
metrics = ov_engine.test()

# predict with the optimized model
predictions = ov_engine.predict()
```

See the [getitune documentation](https://docs.geti.intel.com/docs/user-guide/library/get-started/quick-start) for the full list of recipes,
advanced configuration, dataset support, backend-specific options, and deployment/optimization examples.

## Key Features

<details open>
<summary>🏆 State-of-the-art model catalog</summary>

Train and fine-tune modern architectures such as RF-DETR, DINOv3 DETR, YOLO26, YOLOX, D-FINE, and Mask R-CNN.
Would you like to see a specific model added? Let us know by opening a [GitHub issue](https://github.com/open-edge-platform/geti/issues)!

<!-- markdownlint-disable MD060 -->

<table>
  <thead>
    <tr>
      <th width="30%">Computer Vision Task</th>
      <th>Model Architecture</th>
      <th>Paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="11"><b>Object Detection</b><br>Locate and classify objects with bounding boxes. Common use cases: counting items, defect localization, surveillance.</td>
      <td>D-FINE M / L / X</td>
      <td><a href="https://arxiv.org/abs/2412.04234">DEIM</a> + <a href="https://arxiv.org/abs/2410.13842">D-FINE</a></td>
    </tr>
    <tr>
      <td>DINOv3 DETR S / M / L</td>
      <td><a href="https://arxiv.org/abs/2508.10104">DINOv3</a> + <a href="https://arxiv.org/html/2509.20787v4">DEIMv2</a> + <a href="https://arxiv.org/abs/2005.12872">DETR</a></td>
    </tr>
    <tr>
      <td>ECDet S / M / L / X</td>
      <td><a href="https://arxiv.org/abs/2603.18739">EdgeCrafter</a></td>
    </tr>
    <tr>
      <td>MobileNet V2 ATSS</td>
      <td><a href="https://arxiv.org/abs/1801.04381">MobileNetV2</a> + <a href="https://arxiv.org/abs/1912.02424">ATSS</a></td>
    </tr>
    <tr>
      <td>MobileNet V2 SSD</td>
      <td><a href="https://arxiv.org/abs/1801.04381">MobileNetV2</a> + <a href="https://arxiv.org/abs/1512.02325">SSD</a></td>
    </tr>
    <tr>
      <td>RF-DETR N / S / M / L</td>
      <td><a href="https://arxiv.org/abs/2511.09554">RF-DETR</a></td>
    </tr>
    <tr>
      <td>RT-DETR R50</td>
      <td><a href="https://arxiv.org/abs/2304.08069">RT-DETR</a></td>
    </tr>
    <tr>
      <td>YOLO11 N / S / M / L / X</td>
      <td><a href="https://docs.ultralytics.com/models/yolo11#overview">Ultralytics YOLO11</a></td>
    </tr>
    <tr>
      <td>YOLO12 N / S / M / L / X</td>
      <td><a href="https://arxiv.org/abs/2502.12524">YOLOv12</a></td>
    </tr>
    <tr>
      <td>YOLO26 N / S / M / L / X</td>
      <td><a href="https://arxiv.org/abs/2606.03748">YOLO26</a></td>
    </tr>
    <tr>
      <td>YOLOX Tiny / S / L / X</td>
      <td><a href="https://arxiv.org/abs/2107.08430">YOLOX</a></td>
    </tr>
    <tr>
      <td rowspan="7"><b>Instance Segmentation</b><br>Detect objects and produce pixel-precise masks per instance. Common use cases: medical imaging, robotics, area estimation.</td>
      <td>RTMDet Tiny</td>
      <td><a href="https://arxiv.org/abs/2212.07784">RTMDet</a></td>
    </tr>
    <tr>
      <td>Mask-RCNN EfficientNet B2</td>
      <td><a href="https://arxiv.org/abs/1905.11946">EfficientNet</a> + <a href="https://arxiv.org/abs/1703.06870">Mask R-CNN</a></td>
    </tr>
    <tr>
      <td>Mask-RCNN ResNet50</td>
      <td><a href="https://arxiv.org/abs/1512.03385">ResNet</a> + <a href="https://arxiv.org/abs/1703.06870">Mask R-CNN</a></td>
    </tr>
    <tr>
      <td>Mask-RCNN Swin-T</td>
      <td><a href="https://arxiv.org/abs/2103.14030">Swin Transformer</a> + <a href="https://arxiv.org/abs/1703.06870">Mask R-CNN</a></td>
    </tr>
    <tr>
      <td>RF-DETR N / S / M / L / XL / 2XL</td>
      <td><a href="https://arxiv.org/abs/2511.09554">RF-DETR</a></td>
    </tr>
    <tr>
      <td>YOLO11 N / S / M / L / X</td>
      <td><a href="https://docs.ultralytics.com/models/yolo11#overview">Ultralytics YOLO11</a></td>
    </tr>
    <tr>
      <td>YOLO26 N / S / M / L / X</td>
      <td><a href="https://arxiv.org/abs/2606.03748">YOLO26</a></td>
    </tr>
    <tr>
      <td rowspan="7"><b>Classification</b> (multi-class, multi-label)<br>Assign one or more labels to an entire image. Common use cases: defect classification, product categorization, content tagging.</td>
      <td>ViT Tiny</td>
      <td><a href="https://arxiv.org/abs/2010.11929">ViT</a></td>
    </tr>
    <tr>
      <td>DINOv2 Small</td>
      <td><a href="https://arxiv.org/abs/2304.07193">DINOv2</a></td>
    </tr>
    <tr>
      <td>EfficientNet B0 / B3</td>
      <td><a href="https://arxiv.org/abs/1905.11946">EfficientNet</a></td>
    </tr>
    <tr>
      <td>EfficientNet V2 Small</td>
      <td><a href="https://arxiv.org/abs/2104.00298">EfficientNetV2</a></td>
    </tr>
    <tr>
      <td>MobileNet V3 Large</td>
      <td><a href="https://arxiv.org/abs/1905.02244">MobileNetV3</a></td>
    </tr>
    <tr>
      <td>YOLO26 N / S / M / L / X</td>
      <td><a href="https://arxiv.org/abs/2606.03748">YOLO26</a></td>
    </tr>
    <tr>
      <td>Other models from <i>timm</i> (1600+ backbones)</td>
      <td><a href="https://github.com/huggingface/pytorch-image-models">timm</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-enable MD060 -->

</details>

<details>
<summary>🔄 Interactive end-to-end model training</summary>

Geti™ enables users to start building deep-learning computer vision models with as few as 10-20 images and take them to production in one environment - annotate, train, optimize, run inference, and improve accuracy in a rapid train-predict-annotate loop.

</details>

<details>
<summary>⚡ Hardware-accelerated inference & model optimization</summary>

Every model is automatically exported with [OpenVINO™](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) for deployment across the full Intel® XPU portfolio (Arc™ GPUs, Core™ Ultra processors); NVIDIA® CUDA and CPU-only execution are also supported. Fine-tune and run inference directly on edge and client hardware - including Intel® Panther Lake and Arc™ Battlemage (B-series) GPUs - with no Kubernetes cluster or data-center GPU required. Built-in accuracy-aware INT8 quantization further reduces model size and latency on resource-constrained edge devices with minimal impact on accuracy.

</details>

<details>
<summary>🚀 Integrated deployment & inference</summary>

Build custom pipelines (source → model → sink) to deploy models inside Geti and monitor real-time predictions on video streams. Sources include USB/IP cameras and video files; optional sinks include folder, MQTT, and webhook. Complete pipelines can be exported as OpenVINO™-optimized bundles for edge deployment.

</details>

<details>
<summary>🎨 Multiple computer vision tasks</summary>

Geti™ supports [multiple computer vision tasks](https://docs.geti.intel.com/docs/user-guide/learn-geti/computer-vision-tasks/ai-fundamentals-tasks) that are commonly employed across various use cases - image classification, object detection and instance segmentation from the no-code web interface, with even more tasks available through the `getitune` library.

</details>

<details>
<summary>🧠 Smart annotations</summary>

Smart annotations in Geti™ enable users to easily create bounding boxes and polygons. These smart annotation features coupled with the AI-assisted annotations and state-of-the-art AI models such as the Segment Anything Model keep human experts in the loop while massively reducing the total annotation efforts needed by a human.

<p align="center">
  <img src="application/docs/assets/annotator.webp" width="600" alt="Smart Annotations">
</p>
</details>

<details>
<summary>📦 Model & dataset management</summary>

Track how datasets and models evolve, link models to a specific dataset revision, view exact training hyperparameters, and fine-tune from any previous version. Import and export in COCO, Pascal VOC, YOLO, and a Geti-optimized native format, with label filtering to selectively include or exclude labels on import/export.

</details>

## Ecosystem

Explore other open-source AI projects by Intel®:

- [Anomalib](https://github.com/open-edge-platform/anomalib) - An anomaly detection suite comprising state-of-the-art algorithms and features such as experiment management, hyper-parameter optimization and edge inference.
- [Instant Learn](https://github.com/open-edge-platform/instant-learn) - A framework for developing, benchmarking, and deploying zero-shot visual prompting algorithms on the edge.
- [Datumaro](https://github.com/open-edge-platform/datumaro) - Dataset Management Framework, a Python library and a CLI tool to build, analyze and manage Computer Vision datasets.
- [OpenVINO™](https://github.com/openvinotoolkit/openvino) - Software toolkit for optimizing and deploying deep learning models.
- [OpenVINO™ Model Server](https://github.com/openvinotoolkit/model_server) - A scalable inference server for models optimized with OpenVINO™.
- [Model API](https://github.com/open-edge-platform/model_api) - A set of wrapper classes for particular tasks and model architectures, simplifying data preprocessing and postprocessing as well as routine procedures.
- [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio) - An end-to-end framework for teaching robots to perform tasks through imitation learning from human demonstrations.

## Who uses Geti™?

Geti™ is used by research institutes, industrial partners, universities and AI enthusiasts. Applications range from robotics to medical analysis or industrial quality control. There are some interesting members of the Geti™ community:

- [Intel Foundry](https://medium.com/open-edge-platform/solving-silicon-foundry-woes-with-ai-vision-geti-and-a-robotic-dog-a8382b5d9267)
- [Royal Brompton and Harefield hospitals](https://www.rbht.nhs.uk/artificial-intelligence-theme-new-trust-led-research)
- [WSC Sports](https://www.linkedin.com/posts/wsc-sports-technologies_revolutionizing-sports-broadcasting-with-activity-7161419649878773761-cUM3/)
- [Dell NativeEdge](https://infohub.delltechnologies.com/en-us/p/transforming-edge-ai-with-continuous-learning-meet-intel-geti-and-openvino-on-dell-nativeedge/)
- [Bravent](https://www.linkedin.com/posts/bravent_intelgeti-openvino-manufacturing-activity-7214544905086390272-H19g/)
- [ASRock Industrial](https://www.asrockind.com/en-gb/article/176)
- [PeopleSense.AI](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Intel-Liftoff-Days-2024-Highlights-from-the-Third-Edition/post/1661265)
- [Capgemini](https://www.capgemini.com/insights/expert-perspectives/capgemini-and-intel-corporation-redefining-the-future-of-robotics-and-physical-ai/)

## Contribute

To report a bug or submit a feature request, please open a [GitHub issue](https://github.com/open-edge-platform/geti/issues). If you have an open question, ask in [GitHub Discussions](https://github.com/open-edge-platform/geti/discussions).

For developers who would like to contribute with a pull request, see the [Contributing guide](CONTRIBUTING.md) for details.

<p align="center">
  <b>Thank you 👏 to all our contributors!</b>
</p>

<a href="https://github.com/open-edge-platform/geti/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=open-edge-platform/geti" alt="Contributors" />
</a>

## License

Geti™ is licensed under the [Apache License Version 2.0](LICENSE).

## Disclaimers

Geti™ utilizes FFmpeg.

FFmpeg is an open source project licensed under LGPL and GPL. See [https://www.ffmpeg.org/legal.html](https://www.ffmpeg.org/legal.html). You are solely responsible for determining if your use of FFmpeg requires any additional licenses. Intel is not responsible for obtaining any such licenses, nor liable for any licensing fees due, in connection with your use of FFmpeg.

> [!NOTE]
> Ultralytics YOLO models are distributed under the AGPL-3.0 license, an OSI approved license ideal for open-source research, academic, and personal projects. For commercial use, enhanced support, and tailored licensing terms, please explore flexible Ultralytics licensing options at https://www.ultralytics.com/license.
