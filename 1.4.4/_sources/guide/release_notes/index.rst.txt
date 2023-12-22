Releases
########

.. toctree::
   :maxdepth: 1

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
