# Contributing to the OpenVINO\* Training Extensions

Welcome to the Training Extensions and thank you for your interest in it! Training Extensions is licensed under the Apache* License, Version 2.0. By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

If you want to help developing this project, you can do it in several ways:

## Asking questions

First, make sure your question was not asked by anybody else [before](https://github.com/openvinotoolkit/training_extensions/issues?q=is%3Aissue+label%3Aquestion). If you cannot find a related issue, create a new one and add the `question` label.

## Suggesting new features

You can suggest the feature by creating a new issue with `enhancement` label, but first, check if it was suggested [before](https://github.com/openvinotoolkit/training_extensions/issues?q=is%3Aissue+label%3Aenhancement).

## Creating bug reports

If you observe any errors or incorrect work of any parts of this repository, you can create a bug report ([template is here](.github/ISSUE_TEMPLATE/bug_report.md)). First, check if bug report about this problem was not created before [here](https://github.com/openvinotoolkit/training_extensions/issues?q=is%3Aissue+label%3Abug). If you don't find a related bug report, submit your own via creating new issue with `bug` label.
When creating a new bug report, make sure:
1. You are on the last version of `develop` branch.
2. You provide enough information (Python version, OS version, libraries and other environment info) for others to easily reproduce this bug\issue.
3. You provide necessary and sufficient code snippet to reproduce the bug (if the bug cannot be reproduced in the already existing code).
4. You describe expected and actual behavior.

## Creating pull requests

If you have an idea how to enhance code of the existing models, you are welcome to create a pull request. You can also add a new model, which does not exist in the repository yet.

### Adding training code of the model

If you want to add training code of your model to this repository, make sure:
1. Your model code is licensed under permissive license (Apache-2.0, MIT, BSD, etc).
2. Source framework of the model is PyTorch\*.
3. You provide the code containing implementation of [OTE SDK Task interfaces](https://github.com/openvinotoolkit/training_extensions/tree/develop/ote_sdk/ote_sdk/usecases/tasks/interfaces):
   1. [ITrainingTask](https://github.com/openvinotoolkit/training_extensions/blob/develop/ote_sdk/ote_sdk/usecases/tasks/interfaces/training_interface.py) - your model must be trainable on some datasets.
   1. [IInferenceTask](https://github.com/openvinotoolkit/training_extensions/blob/develop/ote_sdk/ote_sdk/usecases/tasks/interfaces/inference_interface.py) - your model must be inferable on some datasets.
   1. [IEvaluationTask](https://github.com/openvinotoolkit/training_extensions/blob/develop/ote_sdk/ote_sdk/usecases/tasks/interfaces/evaluate_interface.py) - it must be possible to evaluate some quality metric of you model.
   1. [IExportTask](https://github.com/openvinotoolkit/training_extensions/blob/develop/ote_sdk/ote_sdk/usecases/tasks/interfaces/export_interface.py) - your model must be exportable to the OpenVINO format.
4. You provide a model `template.yaml` file, that describes your model and its configurable parameters. See an example [here](external/anomaly/configs/anomaly_classification/padim/template.yaml).
5. You create a PR to the `develop` branch.
6. You provide `requirements.txt` file and a bash `init_venv.sh` script for creating virtual environment.

## Legal Information

[\*] Other names and brands may be claimed as the property of others.

OpenVINO is a trademark of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

Copyright &copy; 2018-2021 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
```
http://www.apache.org/licenses/LICENSE-2.0
```
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
