# Changelog

All notable changes to this project will be documented in this file.

## \[v0.3.0\]

### Added

- Model Preparation Algorithm (MPA)
  - Add new tasks / model templates for Class-Incremental Learning
    - Instance Segmentation (<https://github.com/openvinotoolkit/training_extensions/pull/1142>)
    - Multilabel (<https://github.com/openvinotoolkit/training_extensions/pull/1132>) in Classification
    - Hierarchical-label (<https://github.com/openvinotoolkit/training_extensions/pull/1159>) in Classification
    - SSD and YOLOX model template for Detection (<https://github.com/openvinotoolkit/training_extensions/pull/1156>)
  - Saliency map support
    - Classification (<https://github.com/openvinotoolkit/training_extensions/pull/1166>)
    - Detection (<https://github.com/openvinotoolkit/training_extensions/pull/1155>)
    - Segmentation (<https://github.com/openvinotoolkit/training_extensions/pull/1158>)
  - NNCF (<https://github.com/openvinotoolkit/training_extensions/pull/1157>) support
  - HPO (<https://github.com/openvinotoolkit/training_extensions/pull/1168>) support
  - Balanced Sampler support for Classification (<https://github.com/openvinotoolkit/training_extensions/pull/1139>)
  - Add Adaptive Training for Detection / Instance Segmentation (<https://github.com/openvinotoolkit/training_extensions/pull/1190>)
- Anomaly
  - Add real-life training tests (<https://github.com/openvinotoolkit/training_extensions/pull/898>)
  - Add additional check for early stopping parameter (<https://github.com/openvinotoolkit/training_extensions/pull/1110>)
  - Add DRAEM task implementation (<https://github.com/openvinotoolkit/training_extensions/pull/1203>)

### Changed

- Model Preparation Algorithm (MPA)

  - Replace Class-Incremental Learning models as OTE default models (<https://github.com/openvinotoolkit/training_extensions/pull/1150>)
  - Replace OTE ignored label support with external ignored label
    - Classification (<https://github.com/openvinotoolkit/training_extensions/pull/1132>)
    - Detection (<https://github.com/openvinotoolkit/training_extensions/pull/1128>)
    - Segmentation (<https://github.com/openvinotoolkit/training_extensions/pull/1134>)
  - Enable mixed precision for Classification / Detection / Segmentation (<https://github.com/openvinotoolkit/training_extensions/pull/1198>)
  - Enhance training schedule for Classification (<https://github.com/openvinotoolkit/training_extensions/pull/1212>)
  - Change Model optimization hyper-parameters for Classification / Detection (<https://github.com/openvinotoolkit/training_extensions/pull/1170>)
  - Disable Obsolete test cases for OTE CI (<https://github.com/openvinotoolkit/training_extensions/pull/1220>)

- Anomaly
  - Extend conftest configuration for anomaly backend (<https://github.com/openvinotoolkit/training_extensions/pull/1097>)
  - Expose more params to the UI (<https://github.com/openvinotoolkit/training_extensions/pull/1085>)
  - Change directory structure for anomaly templates (<https://github.com/openvinotoolkit/training_extensions/pull/1105>)
  - Use is_anomalous attribute instead of string matching (<https://github.com/openvinotoolkit/training_extensions/pull/1120>)
  - Set nncf version (<https://github.com/openvinotoolkit/training_extensions/pull/1124>)
  - Move to learning parameters (<https://github.com/openvinotoolkit/training_extensions/pull/1152>)
  - Change OpenVINO MO Command (<https://github.com/openvinotoolkit/training_extensions/pull/1221>)

### Fixed

- Model Preparation Algorithm (MPA)

  - Fix inference issues for Detection (<https://github.com/openvinotoolkit/training_extensions/pull/1167>)
  - Fix model compatibility issue between SC1.1 and 1.2 in Segmentation (<https://github.com/openvinotoolkit/training_extensions/pull/1264>)
  - Some minor issues

- Anomaly
  - Fix non deterministic + sample.py (<https://github.com/openvinotoolkit/training_extensions/pull/1118>)
  - Fix exportable code for anomaly tasks (<https://github.com/openvinotoolkit/training_extensions/pull/1113>)
  - Fix local anomaly segmentation performance bug (<https://github.com/openvinotoolkit/training_extensions/pull/1219>)
  - Fix progress bar (<https://github.com/openvinotoolkit/training_extensions/pull/1223>)
  - Fix inference when model backbone changes (<https://github.com/openvinotoolkit/training_extensions/pull/1242>)

## \[v0.2.0\]

### Added

- Model Preparation Algorithm (MPA), a newly introduced OTE Algorithm backend for advanced transfer learning
  - Class-Incremental Learning support for OTE models
    - Image Classification
    - Object Detection
    - Semantic Segmentation
- Object counting & Rotated object detection are added to Object Detection backend
- Increased support for NNCF / FP16 / HPO
- Ignored label support
- Stop training on NaN losses

### Changed

- Major refactoring
  - Tasks & model templates had been moved to OTE repo from each OTE Algorithm backend

## \[v0.1.1\]

### Fixed

- Some minor issues

## \[v0.1.0\]

### Added

- OTE SDK, defines an interface which can be used by OTE CLI to access OTE Algorithms.
- OTE CLI, contains set of commands needed to operate with deep learning models using OTE SDK Task interfaces.
- OTE Algorithms, contains sub-projects implementing OTE SDK Task interfaces for different deep learning models.
  - Anomaly Classification
  - Image Classification
  - Object Detection
  - Semantic Segmentation
