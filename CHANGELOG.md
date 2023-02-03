# Changelog

All notable changes to this project will be documented in this file.

## \[v0.5.0\]

> _**notes**_
> OpenVINO Training Extension (OTE) which version is equal or older then v0.5.0 does not include the latest functional and security updates. OTE Version 1.0.0 is targeted to be released in February 2023 and will include additional functional and security updates. Customers should update to the latest version as it becomes available.

### Added

- Add tiling in rotated detection (<https://github.com/openvinotoolkit/training_extensions/pull/1420>)
- Add Cythonize AugMixAugment (<https://github.com/openvinotoolkit/training_extensions/pull/1478/>)
- Integrate ov-telemetry (<https://github.com/openvinotoolkit/training_extensions/pull/1568>)

### Changed

- Update OpenVINO to 2022.3 release & nncf to the pre-2.4 version (<https://github.com/openvinotoolkit/training_extensions/pull/1393>)

### Fixed

- Fixing h-label head output bug in OV inference (<https://github.com/openvinotoolkit/training_extensions/pull/1458>)
- Fixing deprecated np.bool issue from numpy==1.24.0 (<https://github.com/openvinotoolkit/training_extensions/pull/1455>)
- Fixing tiling OpenVINO backward compatibility (<https://github.com/openvinotoolkit/training_extensions/pull/1516>)
- Fixing indexing in hierarchical classification inference (<https://github.com/openvinotoolkit/training_extensions/pull/1551>)
- Copying feature vector to resolve duplication issue (<https://github.com/openvinotoolkit/training_extensions/pull/1511>)
- Fixing handling ignored samples in hierarchical head (<https://github.com/openvinotoolkit/training_extensions/pull/1599>)
- Some minor issues

## \[v0.4.0\]

### Added

- Model Preparation Algorithm (MPA)
  - Better saliency map support
    - Replace current saliency map generation with Recipro-CAM for cls (<https://github.com/openvinotoolkit/training_extensions/pull/1363>)
    - Class-wise saliency map generation for the detection task (<https://github.com/openvinotoolkit/training_extensions/pull/1402>)
    - OTE Saliency Map Label (<https://github.com/openvinotoolkit/training_extensions/pull/1447>)
  - Improve object counting algorithm for high-res images via image tiling
    - Add Tiling Module (<https://github.com/openvinotoolkit/training_extensions/pull/1200>)
    - Fliter object less than 1 pixel (<https://github.com/openvinotoolkit/training_extensions/pull/1305>)
    - Tiling deployment (<https://github.com/openvinotoolkit/training_extensions/pull/1387>)
    - Enable tiling oriented detection for v0.4.0/geti1.1.0 (<https://github.com/openvinotoolkit/training_extensions/pull/1427>)

### Fixed

- Hot-fix for Detection fix two stage error (<https://github.com/openvinotoolkit/training_extensions/pull/1433>)
- Fixing ZeroDivisionError in iteration counter for detection-classification project trainings (<https://github.com/openvinotoolkit/training_extensions/pull/1449>)
- Some minor issues

## \[v0.3.1\]

### Fixed

- Neural Network Compression Framework (NNCF)

  - Fix CUDA OOM for NNCF optimization model MaskRCNN-EfficientNetB2B (<https://github.com/openvinotoolkit/training_extensions/pull/1319>)

- Model Preparation Algorithm (MPA)
  - Fix 'Shape out of bounds' error when accepting AI predictions for detection oriented (<https://github.com/openvinotoolkit/training_extensions/pull/1326>)
  - Fix weird confidence behaviour issue on predictions for hierarchical classification (<https://github.com/openvinotoolkit/training_extensions/pull/1332>)
  - Fix training failure issue for hierarchical classification (<https://github.com/openvinotoolkit/training_extensions/pull/1329>)
  - Fix training failure issues for segmentation and instance segmentation during inference process (<https://github.com/openvinotoolkit/training_extensions/pull/1338>)
  - Some minor issues

### Security

- Update vulnerable Python dependencies in OTE (<https://github.com/openvinotoolkit/training_extensions/pull/1303>)

## \[v0.3.0\]

### Added

- Model Preparation Algorithm (MPA)
  - Add new tasks / model templates for Class-Incremental Learning
    - Instance Segmentation (<https://github.com/openvinotoolkit/training_extensions/pull/1142>)
    - Classification
      - Multilabel (<https://github.com/openvinotoolkit/training_extensions/pull/1132>)
      - Hierarchical-label (<https://github.com/openvinotoolkit/training_extensions/pull/1159>)
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
