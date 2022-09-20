# Changelog

All notable changes to this project will be documented in this file.

## \[v0.3.0\]

### Added

- [MPA] Add new tasks / model templates for Class-Incremental Learning
  - Instance Segmentation
  - Multilabel / Hierarchical-label support in Classification
- [MPA] Saliency map support for Classification / Detection / Segmentation
- [MPA] NNCF / HPO support for Multilabel and Hierarchy Classification
- [Anomaly] Add real-life training tests (<https://github.com/openvinotoolkit/training_extensions/pull/898>)
- [Anomaly] Add additional check for early stopping parameter (<https://github.com/openvinotoolkit/training_extensions/pull/1110>)
- [Anomaly] Add DRAEM task implementation (< https://github.com/openvinotoolkit/training_extensions/pull/1203>)

### Changed

- Replace Class-Incremental Learning models in Model Preparation Algorithm (MPA) as OTE default models
- Replace OTE ignored label support with external ignored label
- MPA training schedule enhanced
- [Anomaly] Extend conftest configuration for anomaly backend. (<https://github.com/openvinotoolkit/training_extensions/pull/1097>)
- [Anomaly] Expose more params to the UI (<https://github.com/openvinotoolkit/training_extensions/pull/1085>)
- [Anomaly] Change directory structure for anomaly templates. (<https://github.com/openvinotoolkit/training_extensions/pull/1105>)
- [Anomaly] Use is_anomalous attribute instead of string matching (< https://github.com/openvinotoolkit/training_extensions/pull/1120>)
- [Anomaly] Set nncf version (<https://github.com/openvinotoolkit/training_extensions/pull/1124>)
- [Anomaly] Move to learning parameters (<https://github.com/openvinotoolkit/training_extensions/pull/1152>)
- [Anomaly] Change OpenVINO MO Command (<https://github.com/openvinotoolkit/training_extensions/pull/1221>)

### Fixed

- [Anomaly] Fix non deterministic + sample.py (<https://github.com/openvinotoolkit/training_extensions/pull/1118>)
- [Anomaly] Fix exportable code for anomaly tasks (< https://github.com/openvinotoolkit/training_extensions/pull/1113>)
- [Anomaly] Fix local anomaly segmentation performance bug (< https://github.com/openvinotoolkit/training_extensions/pull/1219>)
- [Anomaly] Fix progress bar (<https://github.com/openvinotoolkit/training_extensions/pull/1223>)
- [Anomaly] Fix inference when model backbone changes (<https://github.com/openvinotoolkit/training_extensions/pull/1242>)

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
