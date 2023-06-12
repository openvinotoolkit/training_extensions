Releases
========

***************
[v1.3.0] (2Q23)
***************

- Support direct annotation input for COCO format
- Action task supports multi GPU training
- Support storage cache in Apache Arrow using Datumaro for action tasks
- Add a simplified greedy labels postprocessing for hierarchical classification
- Support auto adapting batch size
- Support auto adapting num_workers
- Support noisy label detection for detection tasks

*************
v1.2.0 (1Q23)
*************

- Add generating feature cli_report.log in output for otx training
- Support multiple python versions up to 3.10
- Support export of onnx models
- Add option to save images after inference in OTX CLI demo together with demo in exportable code
- Support storage cache in Apache Arrow using Datumaro for cls, det, seg tasks
- Add noisy label detection for multi-class classification task

*************
v1.1.0 (1Q23)
*************

- Add FP16 IR export support
- Add in-memory caching in dataloader
- Add MoViNet template for action classification
- Add Semi-SL multilabel classification algorithm
- Integrate multi-gpu training for semi-supervised learning and self-supervised learning
- Add train-type parameter to otx train
- Add embedding of inference configuration to IR for classification
- Enable VOC dataset in OTX
- Add mmcls.VisionTransformer backbone support

*************
v1.0.0 (1Q23)
*************

- Installation through PyPI
  - Package will be renamed as OpenVINO™ Training Extensions
- CLI update
  - Update ``otx find`` command to find configurations of tasks/algorithms
  - Introduce ``otx build`` command to customize task or model configurations
  - Automatic algorithm selection for the ``otx train`` command using the given input dataset
- Adaptation of `Datumaro <https://github.com/openvinotoolkit/datumaro>`_ component as a dataset interface
