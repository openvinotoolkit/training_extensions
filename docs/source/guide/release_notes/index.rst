Releases
########

.. toctree::
  :maxdepth: 1


v2.1.0 (2024.07)
----------------

.. note::
  OpenVINO™ Training Extensions, version 2.1.0 does not include the latest functional and security updates. OpenVINO™ Training Extensions, version 2.2.0 is targeted to be released in September 2024 and will include additional functional and security updates. Customers should update to the latest version as it becomes available.

New features
^^^^^^^^^^^^

- Add a flag to enable OV inference on dGPU
- Add early stopping with warmup. Remove mandatory background label in semantic segmentation task
- RTMDet-tiny enablement for detection task
- Add data_format validation and update in OTXDataModule
- Add torchvision.MaskRCNN
- Add Semi-SL for Multi-class Classification (EfficientNet-B0)
- Decoupling mmaction for action classification (MoviNet, X3D)
- Add Semi-SL Algorithms for mv3-large, effnet-v2, deit-tiny, dino-v2
- RTMDet-tiny enablement for detection task (export/optimize)
- Enable ruff & ruff-format into otx/algo/classification/backbones
- Add TV MaskRCNN Tile Recipe
- Add rotated det OV recipe

Enhancements
^^^^^^^^^^^^

- Change load_stat_dict to on_load_checkpoint
- Add try - except to keep running the remaining tests
- Update instance_segmentation.py to resolve conflict with 2.0.0
- Update XPU install
- Sync rgb order between torch and ov inference of action classification task
- Make Perf test available to load pervious Perf test to skip training stage
- Reenable e2e classification XAI tests
- Remove action detection task support
- Increase readability of pickling error log during HPO & fix minor bug
- Update RTMDet checkpoint url
- Refactor Torchvision Model for Classification Semi-SL
- Add coverage omit mm-related code
- Add docs semi-sl part
- Refactor docs design & Add contents
- Add execution example of auto batch size in docs
- Add Semi-SL for cls Benchmark Test
- Move value to device before logging for metric
- Add .codecov.yaml
- Update benchmark tool for otx2.1
- Collect pretrained weight binary files in one place
- Minimize compiled dependency files
- Update README & CODEOWNERS
- Update Engine's docstring & CLI --help outputs
- Align integration test to exportable code interface update for release branch
- Refactor exporter for anomaly task and fix a bug with exportable code
- Update pandas version constraint
- Include more models to export test into test_otx_e2e
- Move assigning tasks to Models from Engine to Anomaly Model Classes
- Refactoring detection modules

Bug fixes
^^^^^^^^^

- Fix conflicts between develop and 2.0.0
- Fix polygon mask
- Fix vpm intg test error
- Fix anomaly
- Bug fix in Semantic Segmentation + enable DINOV2 export in ONNX
- Fix some export issues. Remove EXPORTABLE_CODE as export parameter.
- Fix `load_from_checkpoint` to apply original model's hparams
- Fix `load_from_checkpoint` args to apply original model's hparams
- Fix zero-shot `learn` for ov model
- Various fixes for XAI in 2.1
- Fix tests to work in a mm-free environment
- Fix a bug in benchmark code
- Update exportable code dependency & fix a bug
- Fix getting wrong shape during resizing
- Fix detection prediction outputs
- Fix RTMDet PTQ performance
- Fix segmentation fault on VPM PTQ
- Fix NNCF MaskRCNN-Eff accuracy drop
- Fix optimize with Semi-SL data pipeline
- Fix MaskRCNN SwinT NNCF Accuracy Drop

Known issues
^^^^^^^^^^^^

- Post-Training Quantization (PTQ) optimization applied to maskrcnn_swint in the instance segmentation task may result in significantly reduced accuracy. This issue is expected to be addressed with an upgrade to OpenVINO and NNCF in a future release.

v2.0.0 (2Q24)
-------------

.. note::
  OpenVINO™ Training Extensions which version 2.0.0 has been updated to include refactoring of the overall
  architecture and functional updates. Users should [install the new environment](https://openvinotoolkit.github.io/training_extensions/latest/guide/get_started/installation.html).

New features
^^^^^^^^^^^^
- Enable New design to provide a more seamless API/CLI that delivers the value of OTX: [Product Design](https://openvinotoolkit.github.io/training_extensions/latest/guide/explanation/product_design.html)
- Moved away from MMLab's libraries to provide a Lightning-based core and training pipeline
- Use Lightning-based modules and trainers to deliver APIs/CLIs in a more user-friendly way
- Support Intel devices for accelerating deep learning model training

Enhancements
^^^^^^^^^^^^
- Support more models for each task
- Improve the API so user can configure efficient training with shorter code
- Provide more customize settings through the CLI and API
- Enhance the Auto-Configuration feature and made it available in the API

Bug fixes
^^^^^^^^^
- Fixing some minor issues

Known issues
^^^^^^^^^^^^
- Anomaly task processing times have increased compared with v1.\* version, with anomaly classification experiencing a slowdown of approximately 26%, anomaly detection by approximately 213%, and anomaly segmentation by approximately 78%.
  (https://github.com/openvinotoolkit/training_extensions/issues/3592)
- Post-Training Quantization (PTQ) optimization applied to `maskrcnn_swint` in the instance segmentation task may result in significantly reduced accuracy compared with v1.\*
  (https://github.com/openvinotoolkit/training_extensions/issues/3593)

v1.6.1 (2024.05)
----------------

Enhancements
^^^^^^^^^^^^
- Update pymongo version to 4.6.3 for resolving CVE-2024-21506
- Use torchvision in MRCNN on CUDA
- Update IPEX version in installation guide documentation
- Update benchmark
- Bump idan version to 3.7
- Support benchmark history summary
- Upgrade MAPI
- Add NMS iou threshold configurable parameter
- Remedy some medium/low severity bandit issues
- Update documentations
- Add perf benchmark test cases for action and visual prompting

Bug fixes
^^^^^^^^^
- Explicitly cast incorrect output type in OV model
- Update QAT configs for rotated detection
- Hotfix :wrench: Bypass ClsIncrSampler for tiling
- [NNCF] Dynamic shape datasets WA
- [Hotfix] :fire: Fixing detection oriented OV inferencer
- Revert adaptive batch size
- Fix e2e tests for XPU
- Remove torch.xpu.optimize for semantic_segmentation task

v1.6.0 (2024.04)
----------------

New features
^^^^^^^^^^^^
- Changed supported Python version range (>=3.9, <=3.11)
- Support MMDetection COCO format
- Develop JsonSectionPageMapper in Rust API
- Add Filtering via User-Provided Python Functions
- Remove supporting MacOS platform
- Support Kaggle image data (`KaggleImageCsvBase`, `KaggleImageTxtBase`, `KaggleImageMaskBase`, `KaggleVocBase`, `KaggleYoloBase`)
- Add `__getitem__()` for random accessing with O(1) time complexity
- Add Data-aware Anchor Generator
- Support bounding box import within Kaggle extractors and add `KaggleCocoBase`

Enhancements
^^^^^^^^^^^^
- Optimize Python import to make CLI entrypoint faster
- Add ImageColorScale context manager
- Enhance visualizer to toggle plot title visibility
- Enhance Datumaro data format detect() to be memory-bounded and performant
- Change RoIImage and MosaicImage to have np.uint8 dtype as default
- Enable image backend and color channel format to be selectable
- Boost up `CityscapesBase` and `KaggleImageMaskBase` by dropping `np.unique`
- Enhance RISE algortihm for explainable AI
- Enhance explore unit test to use real dataset from ImageNet
- Fix each method of the comparator to be used separately

Bug fixes
^^^^^^^^^
- Fix wrong example of Datumaro dataset creation in document
- Fix wrong command to install datumaro from github
- Update document to correct wrong `datum project import` command and add filtering example to filter out items containing annotations.
- Fix label compare of distance method
- Fix Datumaro visualizer's import errors after introducing lazy import
- Fix broken link to supported formats in readme
- Fix Kinetics data format to have media data
- Handling undefined labels at the annotation statistics
- Add unit test for item rename
- Fix a bug in the previous behavior when importing nested datasets in the project
- Fix Kaggle importer when adding duplicated labels
- Fix input tensor shape in model interpreter for OpenVINO 2023.3
- Add default value for target in prune cli
- Remove deprecated MediaManager
- Fix explore command without project

v1.5.2 (2024.01)
----------------

Enhancements
^^^^^^^^^^^^
- Add memory bounded datumaro data format detect
- Remove Protobuf version limitation (<4)

v1.5.1 (2023.11)
----------------

Enhancements
^^^^^^^^^^^^
- Enhance Datumaro data format stream importer performance
- Change image default dtype from float32 to uint8
- Add comparison level-up doc
- Add ImportError to catch GitPython import error

Bug fixes
^^^^^^^^^
- Modify the draw function in the visualizer not to raise an error for unsupported annotation types.
- Correct explore path in the related document.
- Fix errata in the voc document. Color values in the labelmap.txt should be separated by commas, not colons.
- Fix hyperlink errors in the document.
- Fix memory unbounded Arrow data format export/import.
- Update CVAT format doc to bypass warning.

v1.5.0 (4Q23)
-------------

- Enable configurable confidence threshold for otx eval and export
- Add YOLOX variants as new object detector models
- Enable FeatureVectorHook to support action tasks
- Add ONNX metadata to detection, instance segmantation, and segmentation models
- Add a new feature to configure input size
- Introduce the OTXSampler and AdaptiveRepeatDataHook to achieve faster training at the small data regime
- Add a new object detector Lite-DINO
- Add Semi-SL Mean Teacher algorithm for Instance Segmentation task
- Official supports for YOLOX-X, YOLOX-L, YOLOX-S, ResNeXt101-ATSS
- Add new argument to track resource usage in train command
- Add Self-SL for semantic segmentation of SegNext families
- Adapt input size automatically based on dataset statistics
- Refine input data in-memory caching
- Adapt timeout value of initialization for distributed training
- Optimize data loading by merging load & resize operations w/ caching support for cls/det/iseg/sseg
- Support torch==2.0.1
- Set "Auto" as default input size mode


v1.4.4 (4Q23)
-------------

- Update ModelAPI configuration
- Add Anomaly modelAPI changes
- Update Image numpy access

v1.4.3 (4Q23)
-------------

- Re introduce adaptive scheduling for training

v1.4.2 (4Q23)
-------------

- Upgrade nncf version to 2.6.0
- Bump datumaro version to 1.5.0
- Set tox version constraint
- Add model category attributes to model template
- Minor bug fixes

v1.4.1 (3Q23)
-------------

- Update the README file in exportable code
- Minor bug fixes

v1.4.0 (3Q23)
-------------

- Support encrypted dataset training
- Add custom max iou assigner to prevent CPU OOM when large annotations are used
- Auto train type detection for Semi-SL, Self-SL and Incremental: "--train-type" now is optional
- Add per-class XAI saliency maps for Mask R-CNN model
- Add new object detector Deformable DETR
- Add new object detector DINO
- Add new visual prompting task
- Add new object detector ResNeXt101-ATSS
- Introduce channel_last parameter to improve the performance
- Decrease time for making a workspace
- Set persistent_workers and pin_memory as True in detection task
- New algorithm for Semi-SL semantic segmentation based on metric learning via class prototypes
- Self-SL for classification now can recieve just folder with any images to start contrastive pretraining
- Update OpenVINO version to 2023.0, and NNCF verion to 2.5
- Improve XAI saliency map generation for tiling detection and tiling instance segmentation
- Remove CenterCrop from Classification test pipeline and editing missing docs link
- Switch to PTQ for sseg
- Minor bug fixes

v1.3.1 (2Q23)
-------------

- Minor bug fixes

v1.3.0 (2Q23)
-------------

- Support direct annotation input for COCO format
- Action task supports multi GPU training
- Support storage cache in Apache Arrow using Datumaro for action tasks
- Add a simplified greedy labels postprocessing for hierarchical classification
- Support auto adapting batch size
- Support auto adapting num_workers
- Support noisy label detection for detection tasks
- Make semantic segmentation OpenVINO models compatible with ModelAPI
- Support label hierarchy through LabelTree in LabelSchema for classification task
- Enhance exportable code file structure, video inference and default value for demo
- Speedup OpenVINO inference in image classificaiton, semantic segmentation, object detection and instance segmentation tasks
- Refactoring of ONNX export functionality
- Minor bug fixes

v1.2.4 (3Q23)
-------------

- Per-class saliency maps for M-RCNN
- Disable semantic segmentation soft prediction processing
- Update export and nncf hyperparameters
- Minor bug fixes

v1.2.3 (2Q23)
-------------

- Improve warning message for tiling configurable parameter
- Minor bug fixes

v1.2.1 (2Q23)
-------------

- Upgrade mmdeploy==0.14.0 from official PyPI
- Integrate new ignored loss in semantic segmentation
- Optimize YOLOX data pipeline
- Tiling Spatial Concatenation for OpenVINO IR
- Optimize counting train & inference speed and memory consumption
- Minor bug fixes

v1.2.0 (2Q23)
-------------

- Add generating feature cli_report.log in output for otx training
- Support multiple python versions up to 3.10
- Support export of onnx models
- Add option to save images after inference in OTX CLI demo together with demo in exportable code
- Support storage cache in Apache Arrow using Datumaro for cls, det, seg tasks
- Add noisy label detection for multi-class classification task
- Clean up and refactor the output of the OTX CLI
- Enhance DetCon logic and SupCon for semantic segmentation
- Detection task refactoring
- Classification task refactoring
- Extend OTX explain CLI
- Segmentation task refactoring
- Action task refactoring
- Optimize data preprocessing time and enhance overall performance in semantic segmentation
- Support automatic batch size decrease when there is no enough GPU memory
- Minor bug fixes

v1.1.2 (2Q23)
-------------

- Minor bug fixes


v1.1.1 (1Q23)
-------------

- Minor bug fixes

v1.1.0 (1Q23)
-------------

- Add FP16 IR export support
- Add in-memory caching in dataloader
- Add MoViNet template for action classification
- Add Semi-SL multilabel classification algorithm
- Integrate multi-gpu training for semi-supervised learning and self-supervised learning
- Add train-type parameter to otx train
- Add embedding of inference configuration to IR for classification
- Enable VOC dataset in OTX
- Add mmcls.VisionTransformer backbone support
- Parametrize saliency maps dumping in export
- Bring mmdeploy to action recognition model export & Test optimization of action tasks
- Update backbone lists
- Add explanation for XAI & minor doc fixes
- Refactor phase#1: MPA modules


v1.0.1 (1Q23)
-------------

- Refine documents by proof review
- Separate installation for each tasks
- Improve POT efficiency by setting stat_requests_number parameter to 1
- Minor bug fixes


v1.0.0 (1Q23)
-------------

- Installation through PyPI
  - Package will be renamed as OpenVINO™ Training Extensions
- CLI update
  - Update ``otx find`` command to find configurations of tasks/algorithms
  - Introduce ``otx build`` command to customize task or model configurations
  - Automatic algorithm selection for the ``otx train`` command using the given input dataset
- Adaptation of `Datumaro <https://github.com/openvinotoolkit/datumaro>`_ component as a dataset interface
