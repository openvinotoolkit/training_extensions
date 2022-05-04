# Changelog

All notable changes to this project will be documented in this file.

## \[v0.1.1\]
### Fixed
* Some minor issues

## \[v0.1.0\]
### Added
* OTE SDK, defines an interface which can be used by OTE CLI to access OTE Algorithms.
* OTE CLI, contains set of commands needed to operate with deep learning models using OTE SDK Task interfaces.
* OTE Algorithms, contains sub-projects implementing OTE SDK Task interfaces for different deep learning models.
  * Anomaly Classification
  * Image Classification
  * Object Detection
  * Semantic Segmentation

## \[v0.2.0\]
### Added
* Model Preparation Algorithm (MPA), a newly introduced OTE Algorithm backend for advanced transfer learning
  * Class-Incremental Learning support for OTE models
    * Image Classification
    * Object Detection
    * Semantic Segmentation
