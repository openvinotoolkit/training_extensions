# Changelog

All notable changes to this project will be documented in this file.

## \[unreleased\]

### New features

### Enhancements

## \[v1.6.1\]

### Enhancements

- Update pymongo version to 4.6.3 for resolving CVE-2024-21506
  (<https://github.com/openvinotoolkit/training_extensions/pull/3396>)
- Use torchvision in MRCNN on CUDA
  (<https://github.com/openvinotoolkit/training_extensions/pull/3347>)
- Update IPEX version in installation guide documentation
  (<https://github.com/openvinotoolkit/training_extensions/pull/3343>)
- Update benchmark
  (<https://github.com/openvinotoolkit/training_extensions/pull/3338>)
- Bump idan version to 3.7
  (<https://github.com/openvinotoolkit/training_extensions/pull/3332>)
- Support benchmark history summary
  (<https://github.com/openvinotoolkit/training_extensions/pull/3307>)
- Pin pymongo version to 4.5.0
  (<https://github.com/openvinotoolkit/training_extensions/pull/3316>)
- Upgrade MAPI
  (<https://github.com/openvinotoolkit/training_extensions/pull/3304>)
- Add NMS iou threshold configurable parameter
  (<https://github.com/openvinotoolkit/training_extensions/pull/3287>)
- Remedy some medium/low severity bandit issues
  (<https://github.com/openvinotoolkit/training_extensions/pull/3208>)
- Update documentations
  (<https://github.com/openvinotoolkit/training_extensions/pull/3280>)
- Add perf benchmark test cases for action and visual prompting
  (<https://github.com/openvinotoolkit/training_extensions/pull/3292>)

### Bug fixes

- Explicitly cast incorrect output type in OV model
  (<https://github.com/openvinotoolkit/training_extensions/pull/3395>)
- Update QAT configs for rotated detection
  (<https://github.com/openvinotoolkit/training_extensions/pull/3375>)
- Hotfix :wrench: Bypass ClsIncrSampler for tiling
  (<https://github.com/openvinotoolkit/training_extensions/pull/3374>)
- [NNCF] Dynamic shape datasets WA
  (<https://github.com/openvinotoolkit/training_extensions/pull/3355>)
- [Hotfix] :fire: Fixing detection oriented OV inferencer
  (<https://github.com/openvinotoolkit/training_extensions/pull/3351>)
- Revert adaptive batch size
  (<https://github.com/openvinotoolkit/training_extensions/pull/3340>)
- Fix e2e tests for XPU
  (<https://github.com/openvinotoolkit/training_extensions/pull/3305>)
- Remove torch.xpu.optimize for semantic_segmentation task
  (<https://github.com/openvinotoolkit/training_extensions/pull/3172>)

## \[1.6.0\]

### New features

- Changed supported Python version range (>=3.9, <=3.11)
  (<https://github.com/openvinotoolkit/datumaro/pull/1269>)
- Support MMDetection COCO format
  (<https://github.com/openvinotoolkit/datumaro/pull/1213>)
- Develop JsonSectionPageMapper in Rust API
  (<https://github.com/openvinotoolkit/datumaro/pull/1224>)
- Add Filtering via User-Provided Python Functions
  (<https://github.com/openvinotoolkit/datumaro/pull/1230>, <https://github.com/openvinotoolkit/datumaro/pull/1233>)
- Remove supporting MacOS platform
  (<https://github.com/openvinotoolkit/datumaro/pull/1235>)
- Support Kaggle image data (`KaggleImageCsvBase`, `KaggleImageTxtBase`, `KaggleImageMaskBase`, `KaggleVocBase`, `KaggleYoloBase`)
  (<https://github.com/openvinotoolkit/datumaro/pull/1240>)
- Add `__getitem__()` for random accessing with O(1) time complexity
  (<https://github.com/openvinotoolkit/datumaro/pull/1247>)
- Add Data-aware Anchor Generator
  (<https://github.com/openvinotoolkit/datumaro/pull/1251>)
- Support bounding box import within Kaggle extractors and add `KaggleCocoBase`
  (<https://github.com/openvinotoolkit/datumaro/pull/1273>)

### Enhancements

- Optimize Python import to make CLI entrypoint faster
  (<https://github.com/openvinotoolkit/datumaro/pull/1182>)
- Add ImageColorScale context manager
  (<https://github.com/openvinotoolkit/datumaro/pull/1194>)
- Enhance visualizer to toggle plot title visibility
  (<https://github.com/openvinotoolkit/datumaro/pull/1228>)
- Enhance Datumaro data format detect() to be memory-bounded and performant
  (<https://github.com/openvinotoolkit/datumaro/pull/1229>)
- Change RoIImage and MosaicImage to have np.uint8 dtype as default
  (<https://github.com/openvinotoolkit/datumaro/pull/1245>)
- Enable image backend and color channel format to be selectable
  (<https://github.com/openvinotoolkit/datumaro/pull/1246>)
- Boost up `CityscapesBase` and `KaggleImageMaskBase` by dropping `np.unique`
  (<https://github.com/openvinotoolkit/datumaro/pull/1261>)
- Enhance RISE algortihm for explainable AI
  (<https://github.com/openvinotoolkit/datumaro/pull/1263>)
- Enhance explore unit test to use real dataset from ImageNet
  (<https://github.com/openvinotoolkit/datumaro/pull/1266>)
- Fix each method of the comparator to be used separately
  (<https://github.com/openvinotoolkit/datumaro/pull/1290>)
- Bump ONNX version to 1.16.0
  (<https://github.com/openvinotoolkit/datumaro/pull/1376>)
- Print the color channel format (RGB) for datum stats command
  (<https://github.com/openvinotoolkit/datumaro/pull/1389>)
- Add ignore_index argument to Mask.as_class_mask() and Mask.as_instance_mask()
  (<https://github.com/openvinotoolkit/datumaro/pull/1409>)

### Bug fixes

- Fix wrong example of Datumaro dataset creation in document
  (<https://github.com/openvinotoolkit/datumaro/pull/1195>)
- Fix wrong command to install datumaro from github
  (<https://github.com/openvinotoolkit/datumaro/pull/1202>, <https://github.com/openvinotoolkit/datumaro/pull/1207>)
- Update document to correct wrong `datum project import` command and add filtering example to filter out items containing annotations.
  (<https://github.com/openvinotoolkit/datumaro/pull/1210>)
- Fix label compare of distance method
  (<https://github.com/openvinotoolkit/datumaro/pull/1205>)
- Fix Datumaro visualizer's import errors after introducing lazy import
  (<https://github.com/openvinotoolkit/datumaro/pull/1220>)
- Fix broken link to supported formats in readme
  (<https://github.com/openvinotoolkit/datumaro/pull/1221>)
- Fix Kinetics data format to have media data
  (<https://github.com/openvinotoolkit/datumaro/pull/1223>)
- Handling undefined labels at the annotation statistics
  (<https://github.com/openvinotoolkit/datumaro/pull/1232>)
- Add unit test for item rename
  (<https://github.com/openvinotoolkit/datumaro/pull/1237>)
- Fix a bug in the previous behavior when importing nested datasets in the project
  (<https://github.com/openvinotoolkit/datumaro/pull/1243>)
- Fix Kaggle importer when adding duplicated labels
  (<https://github.com/openvinotoolkit/datumaro/pull/1244>)
- Fix input tensor shape in model interpreter for OpenVINO 2023.3
  (<https://github.com/openvinotoolkit/datumaro/pull/1251>)
- Add default value for target in prune cli
  (<https://github.com/openvinotoolkit/datumaro/pull/1253>)
- Remove deprecated MediaManager
  (<https://github.com/openvinotoolkit/datumaro/pull/1262>)
- Fix explore command without project
  (<https://github.com/openvinotoolkit/datumaro/pull/1271>)
- Fix enable COCO to import only bboxes
  (<https://github.com/openvinotoolkit/datumaro/pull/1360>)
- Fix resize transform for RleMask annotation
- (<https://github.com/openvinotoolkit/datumaro/pull/1361>)
- Fix import YOLO variants from extractor when `urls` is not specified
  (<https://github.com/openvinotoolkit/datumaro/pull/1362>)

## \[1.5.2\]

### Enhancements

- Add memory bounded datumaro data format detect to release 1.5.1
  (<https://github.com/openvinotoolkit/datumaro/pull/1241>)
- Bump version string to 1.5.2
  (<https://github.com/openvinotoolkit/datumaro/pull/1249>)
- Remove Protobuf version limitation (<4)
  (<https://github.com/openvinotoolkit/datumaro/pull/1248>)

## \[1.5.1\]

### Enhancements

- Enhance Datumaro data format stream importer performance
  (<https://github.com/openvinotoolkit/datumaro/pull/1153>)
- Change image default dtype from float32 to uint8
  (<https://github.com/openvinotoolkit/datumaro/pull/1175>)
- Add comparison level-up doc
  (<https://github.com/openvinotoolkit/datumaro/pull/1174>)
- Add ImportError to catch GitPython import error
  (<https://github.com/openvinotoolkit/datumaro/pull/1174>)

### Bug fixes

- Modify the draw function in the visualizer not to raise an error for unsupported annotation types.
  (<https://github.com/openvinotoolkit/datumaro/pull/1180>)
- Correct explore path in the related document.
  (<https://github.com/openvinotoolkit/datumaro/pull/1176>)
- Fix errata in the voc document. Color values in the labelmap.txt should be separated by commas, not colons.
  (<https://github.com/openvinotoolkit/datumaro/pull/1162>)
- Fix hyperlink errors in the document
  (<https://github.com/openvinotoolkit/datumaro/pull/1159>, <https://github.com/openvinotoolkit/datumaro/pull/1161>)
- Fix memory unbounded Arrow data format export/import
  (<https://github.com/openvinotoolkit/datumaro/pull/1169>)
- Update CVAT format doc to bypass warning
  (<https://github.com/openvinotoolkit/datumaro/pull/1183>)

## \[v1.5.0\]

### New features

- Enable configurable confidence threshold for otx eval and export (<https://github.com/openvinotoolkit/training_extensions/pull/2388>)
- Add YOLOX variants as new object detector models (<https://github.com/openvinotoolkit/training_extensions/pull/2402>)
- Enable FeatureVectorHook to support action tasks (<https://github.com/openvinotoolkit/training_extensions/pull/2408>)
- Add ONNX metadata to detection, instance segmentation, and segmentation models (<https://github.com/openvinotoolkit/training_extensions/pull/2418>)
- Add a new feature to configure input size (<https://github.com/openvinotoolkit/training_extensions/pull/2420>)
- Introduce the OTXSampler and AdaptiveRepeatDataHook to achieve faster training at the small data regime (<https://github.com/openvinotoolkit/training_extensions/pull/2428>)
- Add a new object detector Lite-DINO (<https://github.com/openvinotoolkit/training_extensions/pull/2457>)
- Add Semi-SL Mean Teacher algorithm for Instance Segmentation task (<https://github.com/openvinotoolkit/training_extensions/pull/2444>)
- Official supports for YOLOX-X, YOLOX-L, YOLOX-S, ResNeXt101-ATSS (<https://github.com/openvinotoolkit/training_extensions/pull/2485>)
- Add new argument to track resource usage in train command (<https://github.com/openvinotoolkit/training_extensions/pull/2500>)
- Add Self-SL for semantic segmentation of SegNext families (<https://github.com/openvinotoolkit/training_extensions/pull/2215>)
- Adapt input size automatically based on dataset statistics (<https://github.com/openvinotoolkit/training_extensions/pull/2499>)

### Enhancements

- Refine input data in-memory caching (<https://github.com/openvinotoolkit/training_extensions/pull/2416>)
- Adapt timeout value of initialization for distributed training (<https://github.com/openvinotoolkit/training_extensions/pull/2422>)
- Optimize data loading by merging load & resize operations w/ caching support for cls/det/iseg/sseg (<https://github.com/openvinotoolkit/training_extensions/pull/2438>, <https://github.com/openvinotoolkit/training_extensions/pull/2453>, <https://github.com/openvinotoolkit/training_extensions/pull/2460>)
- Support torch==2.0.1 (<https://github.com/openvinotoolkit/training_extensions/pull/2465>)
- Set "Auto" as default input size mode (<https://github.com/openvinotoolkit/training_extensions/pull/2515>)

### Bug fixes

- Fix F1 auto-threshold to choose best largest confidence (<https://github.com/openvinotoolkit/training_extensions/pull/2371>)
- Fix IBLoss enablement with DeiT-Tiny when class incremental training (<https://github.com/openvinotoolkit/training_extensions/pull/2594>)

### Known issues

- OpenVINO(==2023.0) IR inference is not working well on 2-stage models (e.g. Mask-RCNN) exported from torch>=1.13.1
- NNCF QAT optimization is disabled for MaskRCNN models due to CUDA runtime error in ROIAlign kernel on torch==2.0.1

## \[v1.4.4\]

### Enhancements

- Update ModelAPI configuration(<https://github.com/openvinotoolkit/training_extensions/pull/2564>)
- Add Anomaly modelAPI changes (<https://github.com/openvinotoolkit/training_extensions/pull/2563>)
- Update Image numpy access (<https://github.com/openvinotoolkit/training_extensions/pull/2586>)
- Make max_num_detections configurable (<https://github.com/openvinotoolkit/training_extensions/pull/2647>)

### Bug fixes

- Fix IBLoss enablement with DeiT-Tiny when class incremental training (<https://github.com/openvinotoolkit/training_extensions/pull/2595>)
- Fix mmcls bug not wrapping model in DataParallel on CPUs (<https://github.com/openvinotoolkit/training_extensions/pull/2601>)
- Fix h-label loss normalization issue w/ exclusive label group of singe label (<https://github.com/openvinotoolkit/training_extensions/pull/2604>)
- Fix division by zero in class incremental learning for classification (<https://github.com/openvinotoolkit/training_extensions/pull/2606>)
- Fix saliency maps calculation issue for detection models (<https://github.com/openvinotoolkit/training_extensions/pull/2609>)
- Fix h-label bug of missing parent labels in output (<https://github.com/openvinotoolkit/training_extensions/pull/2626>)

## \[v1.4.3\]

### Enhancements

- Re-introduce adaptive scheduling for training (<https://github.com/openvinotoolkit/training_extensions/pull/2541>)

## \[v1.4.2\]

### Enhancements

- Upgrade nncf version to 2.6.0 (<https://github.com/openvinotoolkit/training_extensions/pull/2459>)
- Bump datumaro version to 1.5.0 (<https://github.com/openvinotoolkit/training_extensions/pull/2470>, <https://github.com/openvinotoolkit/training_extensions/pull/2502>)
- Set tox version constraint (<https://github.com/openvinotoolkit/training_extensions/pull/2472>)
- Add model category attributes to model template (<https://github.com/openvinotoolkit/training_extensions/pull/2439>)

### Bug fixes

- Bug fix for albumentations (<https://github.com/openvinotoolkit/training_extensions/pull/2467>)
- Add workaround for the incorrect meta info M-RCNN (used for XAI) (<https://github.com/openvinotoolkit/training_extensions/pull/2437>)
- Fix label list order for h-label classification (<https://github.com/openvinotoolkit/training_extensions/pull/2440>)
- Modified fq numbers for lite HRNET e2e tests (<https://github.com/openvinotoolkit/training_extensions/pull/2445>)

## \[v1.4.1\]

### Enhancements

- Update the README file in exportable code (<https://github.com/openvinotoolkit/training_extensions/pull/2411>)

### Bug fixes

- Fix broken links in documentation (<https://github.com/openvinotoolkit/training_extensions/pull/2405>)

## \[v1.4.0\]

### New features

- Support encrypted dataset training (<https://github.com/openvinotoolkit/training_extensions/pull/2209>)
- Add custom max iou assigner to prevent CPU OOM when large annotations are used (<https://github.com/openvinotoolkit/training_extensions/pull/2228>)
- Auto train type detection for Semi-SL, Self-SL and Incremental: "--train-type" now is optional (<https://github.com/openvinotoolkit/training_extensions/pull/2195>)
- Add per-class XAI saliency maps for Mask R-CNN model (<https://github.com/openvinotoolkit/training_extensions/pull/2227>)
- Add new object detector Deformable DETR (<https://github.com/openvinotoolkit/training_extensions/pull/2249>)
- Add new object detector DINO (<https://github.com/openvinotoolkit/training_extensions/pull/2266>)
- Add new visual prompting task (<https://github.com/openvinotoolkit/training_extensions/pull/2203>, <https://github.com/openvinotoolkit/training_extensions/pull/2274>, <https://github.com/openvinotoolkit/training_extensions/pull/2311>, <https://github.com/openvinotoolkit/training_extensions/pull/2354>, <https://github.com/openvinotoolkit/training_extensions/pull/2318>)
- Add new object detector ResNeXt101-ATSS (<https://github.com/openvinotoolkit/training_extensions/pull/2309>)

### Enhancements

- Introduce channel_last parameter to improve the performance (<https://github.com/openvinotoolkit/training_extensions/pull/2205>)
- Decrease time for making a workspace (<https://github.com/openvinotoolkit/training_extensions/pull/2223>)
- Set persistent_workers and pin_memory as True in detection task (<https://github.com/openvinotoolkit/training_extensions/pull/2224>)
- New algorithm for Semi-SL semantic segmentation based on metric learning via class prototypes (<https://github.com/openvinotoolkit/training_extensions/pull/2156>)
- Self-SL for classification now can recieve just folder with any images to start contrastive pretraining (<https://github.com/openvinotoolkit/training_extensions/pull/2219>)
- Update OpenVINO version to 2023.0, and NNCF verion to 2.5 (<https://github.com/openvinotoolkit/training_extensions/pull/2090>)
- Improve XAI saliency map generation for tiling detection and tiling instance segmentation (<https://github.com/openvinotoolkit/training_extensions/pull/2240>)
- Remove CenterCrop from Classification test pipeline and editing missing docs link(<https://github.com/openvinotoolkit/training_extensions/pull/2375>)
- Switch to PTQ for sseg (<https://github.com/openvinotoolkit/training_extensions/pull/2374>)

### Bug fixes

- Fix the bug that auto adapt batch size is unavailable with IterBasedRunner (<https://github.com/openvinotoolkit/training_extensions/pull/2182>)
- Fix the bug that learning rate isn't scaled when multi-GPU trianing is enabled(<https://github.com/openvinotoolkit/training_extensions/pull/2254>)
- Fix the bug that label order is misaligned when model is deployed from Geti (<https://github.com/openvinotoolkit/training_extensions/pull/2369>)
- Fix NNCF training on CPU (<https://github.com/openvinotoolkit/training_extensions/pull/2373>)
- Fix H-label classification (<https://github.com/openvinotoolkit/training_extensions/pull/2377>)
- Fix invalid import structures in otx.api (<https://github.com/openvinotoolkit/training_extensions/pull/2383>)
- Add for async inference calculating saliency maps from predictions (Mask RCNN IR) (<https://github.com/openvinotoolkit/training_extensions/pull/2395>)

### Known issues

- OpenVINO(==2023.0) IR inference is not working well on 2-stage models (e.g. Mask-RCNN) exported from torch==1.13.1

## \[v1.3.1\]

### Enhancements

- n/a

### Bug fixes

- Fix a bug that auto adapt batch size doesn't work with cls incr case (<https://github.com/openvinotoolkit/training_extensions/pull/2199>)
- Fix a bug that persistent worker is True even if num_workers is zero (<https://github.com/openvinotoolkit/training_extensions/pull/2208>)

### Known issues

- OpenVINO(==2022.3) IR inference is not working well on 2-stage models (e.g. Mask-RCNN) exported from torch==1.13.1
  (working well up to torch==1.12.1) (<https://github.com/openvinotoolkit/training_extensions/issues/1906>)

## \[v1.3.0\]

### New features

- Support direct annotation input for COCO format (<https://github.com/openvinotoolkit/training_extensions/pull/1921>)
- Action task supports multi GPU training. (<https://github.com/openvinotoolkit/training_extensions/pull/2057>)
- Support storage cache in Apache Arrow using Datumaro for action tasks (<https://github.com/openvinotoolkit/training_extensions/pull/2087>)
- Add a simplified greedy labels postprocessing for hierarchical classification (<https://github.com/openvinotoolkit/training_extensions/pull/2064>).
- Support auto adapting batch size (<https://github.com/openvinotoolkit/training_extensions/pull/2119>)
- Support auto adapting num_workers (<https://github.com/openvinotoolkit/training_extensions/pull/2165>)
- Support noisy label detection for detection tasks (<https://github.com/openvinotoolkit/training_extensions/pull/2109>, <https://github.com/openvinotoolkit/training_extensions/pull/2115>, <https://github.com/openvinotoolkit/training_extensions/pull/2123>, <https://github.com/openvinotoolkit/training_extensions/pull/2183>)

### Enhancements

- Make semantic segmentation OpenVINO models compatible with ModelAPI (<https://github.com/openvinotoolkit/training_extensions/pull/2029>).
- Support label hierarchy through LabelTree in LabelSchema for classification task (<https://github.com/openvinotoolkit/training_extensions/pull/2149>, <https://github.com/openvinotoolkit/training_extensions/pull/2152>).
- Enhance exportable code file structure, video inference and default value for demo (<https://github.com/openvinotoolkit/training_extensions/pull/2051>).
- Speedup OpenVINO inference in image classificaiton, semantic segmentation, object detection and instance segmentation tasks (<https://github.com/openvinotoolkit/training_extensions/pull/2105>).
- Refactoring of ONNX export functionality (<https://github.com/openvinotoolkit/training_extensions/pull/2155>).
- SSD detector Optimization(<https://github.com/openvinotoolkit/training_extensions/pull/2197>)

### Bug fixes

- Fix async mode inference for demo in exportable code (<https://github.com/openvinotoolkit/training_extensions/pull/2154>)
- Fix a bug that auto adapt batch size doesn't work with cls incr case (<https://github.com/openvinotoolkit/training_extensions/pull/2199>)

### Known issues

- OpenVINO(==2022.3) IR inference is not working well on 2-stage models (e.g. Mask-RCNN) exported from torch==1.13.1
  (working well up to torch==1.12.1) (<https://github.com/openvinotoolkit/training_extensions/issues/1906>)

## \[v1.2.3\]

### Bug fixes

- Return raw anomaly map instead of colormap as metadata to prevent applying colormap conversion twice (<https://github.com/openvinotoolkit/training_extensions/pull/2217>)
- Hotfix: use 0 confidence threshold when computing best threshold based on F1

## \[v1.2.2\]

### Enhancements

- Improve warning message for tiling configurable parameter

### Known issues

- OpenVINO(==2022.3) IR inference is not working well on 2-stage models (e.g. Mask-RCNN) exported from torch==1.13.1
  (working well up to torch==1.12.1) (<https://github.com/openvinotoolkit/training_extensions/issues/1906>)

## \[v1.2.1\]

### Enhancements

- Upgrade mmdeploy==0.14.0 from official PyPI (<https://github.com/openvinotoolkit/training_extensions/pull/2047>)
- Integrate new ignored loss in semantic segmentation (<https://github.com/openvinotoolkit/training_extensions/pull/2065>, <https://github.com/openvinotoolkit/training_extensions/pull/2111>)
- Optimize YOLOX data pipeline (<https://github.com/openvinotoolkit/training_extensions/pull/2075>)
- Tiling Spatial Concatenation for OpenVINO IR (<https://github.com/openvinotoolkit/training_extensions/pull/2052>)
- Optimize counting train & inference speed and memory consumption (<https://github.com/openvinotoolkit/training_extensions/pull/2172>)

### Bug fixes

- Bug fix: value of validation variable is changed after auto decrease batch size (<https://github.com/openvinotoolkit/training_extensions/pull/2053>)
- Fix tiling 0 stride issue in parameter adapter (<https://github.com/openvinotoolkit/training_extensions/pull/2078>)
- Fix Tiling NNCF (<https://github.com/openvinotoolkit/training_extensions/pull/2081>)
- Do not skip full img tile classifier + Fix Sequencial Export Issue (<https://github.com/openvinotoolkit/training_extensions/pull/2174>)

## \[v1.2.0\]

### New features

- Add generating feature cli_report.log in output for otx training (<https://github.com/openvinotoolkit/training_extensions/pull/1959>)
- Support multiple python versions up to 3.10 (<https://github.com/openvinotoolkit/training_extensions/pull/1978>)
- Support export of onnx models (<https://github.com/openvinotoolkit/training_extensions/pull/1976>)
- Add option to save images after inference in OTX CLI demo together with demo in exportable code (<https://github.com/openvinotoolkit/training_extensions/pull/2005>)
- Support storage cache in Apache Arrow using Datumaro for cls, det, seg tasks (<https://github.com/openvinotoolkit/training_extensions/pull/2009>)
- Add noisy label detection for multi-class classification task (<https://github.com/openvinotoolkit/training_extensions/pull/1985>, <https://github.com/openvinotoolkit/training_extensions/pull/2034>)
- Add DeiT template for classification tasks as experimental template (<https://github.com/openvinotoolkit/training_extensions/pull/2093)

### Enhancements

- Clean up and refactor the output of the OTX CLI (<https://github.com/openvinotoolkit/training_extensions/pull/1946>)
- Enhance DetCon logic and SupCon for semantic segmentation(<https://github.com/openvinotoolkit/training_extensions/pull/1958>)
- Detection task refactoring (<https://github.com/openvinotoolkit/training_extensions/pull/1955>)
- Classification task refactoring (<https://github.com/openvinotoolkit/training_extensions/pull/1972>)
- Extend OTX explain CLI (<https://github.com/openvinotoolkit/training_extensions/pull/1941>)
- Segmentation task refactoring (<https://github.com/openvinotoolkit/training_extensions/pull/1977>)
- Action task refactoring (<https://github.com/openvinotoolkit/training_extensions/pull/1993>)
- Optimize data preprocessing time and enhance overall performance in semantic segmentation (<https://github.com/openvinotoolkit/training_extensions/pull/2020>)
- Support automatic batch size decrease when there is no enough GPU memory (<https://github.com/openvinotoolkit/training_extensions/pull/2022>)
- Refine HPO usability (<https://github.com/openvinotoolkit/training_extensions/pull/2175>)

### Bug fixes

- Fix backward compatibility with OpenVINO SSD-like detection models from OTE 0.5 (<https://github.com/openvinotoolkit/training_extensions/pull/1970>)

### Known issues

- OpenVINO(==2022.3) IR inference is not working well on 2-stage models (e.g. Mask-RCNN) exported from torch==1.13.1
  (working well up to torch==1.12.1) (<https://github.com/openvinotoolkit/training_extensions/issues/1906>)

## \[v1.1.2\]

### Bug fixes

- Fix exception -> warning for anomaly dump_feature option
- Remove `dataset.with_empty_annotations()` to keep original input structure (<https://github.com/openvinotoolkit/training_extensions/pull/1964>)
- Fix OV batch inference (saliency map generation) (<https://github.com/openvinotoolkit/training_extensions/pull/1965>)
- Replace EfficentNetB0 model download logic by pytorchcv to resolve zip issue (<https://github.com/openvinotoolkit/training_extensions/pull/1967>)

## \[v1.1.1\]

### Bug fixes

- Add missing OpenVINO dependency in exportable code requirement

## \[v1.1.0\]

### New features

- Add FP16 IR export support (<https://github.com/openvinotoolkit/training_extensions/pull/1683>)
- Add in-memory caching in dataloader (<https://github.com/openvinotoolkit/training_extensions/pull/1694>)
- Add MoViNet template for action classification (<https://github.com/openvinotoolkit/training_extensions/pull/1742>)
- Add Semi-SL multilabel classification algorithm (<https://github.com/openvinotoolkit/training_extensions/pull/1805>)
- Integrate multi-gpu training for semi-supervised learning and self-supervised learning (<https://github.com/openvinotoolkit/training_extensions/pull/1534>)
- Add train-type parameter to otx train (<https://github.com/openvinotoolkit/training_extensions/pull/1874>)
- Add embedding of inference configuration to IR for classification (<https://github.com/openvinotoolkit/training_extensions/pull/1842>)
- Enable VOC dataset in OTX (<https://github.com/openvinotoolkit/training_extensions/pull/1862>)
- Add mmcls.VisionTransformer backbone support (<https://github.com/openvinotoolkit/training_extensions/pull/1908>)

### Enhancements

- Parametrize saliency maps dumping in export (<https://github.com/openvinotoolkit/training_extensions/pull/1888>)
- Bring mmdeploy to action recognition model export & Test optimization of action tasks (<https://github.com/openvinotoolkit/training_extensions/pull/1848>)
- Update backbone lists (<https://github.com/openvinotoolkit/training_extensions/pull/1835>)
- Add explanation for XAI & minor doc fixes (<https://github.com/openvinotoolkit/training_extensions/pull/1923>)
- Refactor phase#1: MPA modules

### Bug fixes

- Handle unpickable update_progress_callback (<https://github.com/openvinotoolkit/training_extensions/pull/1892>)
- Dataset Adapter: Avoid duplicated annotation and permit empty image (<https://github.com/openvinotoolkit/training_extensions/pull/1873>)
- Arrange scale between bbox preds and bbox targets in ATSS (<https://github.com/openvinotoolkit/training_extensions/pull/1880>)
- Fix label mismatch of evaluation and validation with large dataset in semantic segmentation (<https://github.com/openvinotoolkit/training_extensions/pull/1851>)
- Fix packaging errors including cython module build / import issues (<https://github.com/openvinotoolkit/training_extensions/pull/1936>)

### Known issues

- OpenVINO(==2022.3) IR inference is not working well on 2-stage models (e.g. Mask-RCNN) exported from torch==1.13.1
  (working well up to torch==1.12.1) (<https://github.com/openvinotoolkit/training_extensions/issues/1906>)

## \[v1.0.1\]

### Enhancements

- Refine documents by proof review
- Separate installation for each tasks
- Improve POT efficiency by setting stat_requests_number parameter to 1
- Introduce new tile classifier to enhance tiling inference performance in MaskRCNN.

### Bug fixes

- Fix missing classes in cls checkpoint
- Fix action task sample codes
- Fix label_scheme mismatch in classification
- Fix training error when batch size is 1
- Fix hang issue when tracing a stack in certain scenario
- Fix pickling error by Removing mmcv cfg dump in ckpt

## \[v1.0.0\]

> _**NOTES**_
>
> OpenVINO™ Training Extensions which version 1.0.0 has been updated to include functional and security updates. Users should update to the latest version.

### New features

- Adaptation of [Datumaro](https://github.com/openvinotoolkit/datumaro) component as a dataset interface
- Integrate hyper-parameter optimizations
- Support action recognition task
- Self-supervised learning mode for representational training
- Semi-supervised learning mode for better model quality

### Enhancements

- Installation via [PyPI package](https://pypi.org/project/otx/)
- Enhance `find` command to find configurations of supported tasks / algorithms / models / backbones
- Introduce `build` command to customize task or model configurations in isolated workspace
- Auto-config feature to automatically select the right algorithm and default model for the `train` & `build` command by detecting the task type of given input dataset
- Improve [documentation](https://openvinotoolkit.github.io/training_extensions/1.0.0/guide/get_started/introduction.html)
- Improve training performance by introducing enhanced loss for the few-shot transfer

### Bug fixes

- Fixing configuration loading issue from the meta data of the model in OpenVINO task for the backward compatibility
- Fixing some minor issues

## \[v0.5.0\]

> _**NOTES**_
>
> OpenVINO Training Extension which version is equal or older then v0.5.0 does not include the latest functional and security updates. OTE Version 1.0.0 is targeted to be released in February 2023 and will include additional functional and security updates. Customers should update to the latest version as it becomes available.

### Added

- Add tiling in rotated detection (<https://github.com/openvinotoolkit/training_extensions/pull/1420>)
- Add Cythonize AugMixAugment (<https://github.com/openvinotoolkit/training_extensions/pull/1478>)
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
