# Experiment helper

experiment.py is a powerful tool designed to streamline and automate the process of conducting experiments using OTX.
It simplifies the execution of multiple test cases, automatically parses output values,
and organizes results efficiently.
The primary goal is to reduce the manual effort required in running experiments and enhance overall productivity.

## Key features

### Automated Experiment Execution

- Given multiple variables, it automatically generates all combinations and runs the experiments.
- Proper model files are selected automatically when the "otx eval" or "otx optimize" command is executed, based on the preceding command.

### Fault Tolerance

- Subsequent jobs are executed independently, irrespective of whether the previous job raised an error.
- All failed commands are printed and saved in a file after the entire experiment is finished.

### Automated Experiment Execution

- All possible values from a single workspace are organized and saved in a file.
- Experiment results are aggregated after the completion of all commands.

## How to Use

### Feature 1 : run experiments & aggregate results

Arguments

- -f / --file : Path to the YAML file describing the experiment setup. After all runs, results are aggregated and saved.
- -d / --dryrun : Preview the experiment list before execution. Use with '-f / --file' argument.

Both single experiment and multiple experiments are supported.
Here is the example.

Single Experiment Recipe YAML File:

    output_path: research_framework_demo/det_model_test
    constants: # value in constant can't have other constant or variable.
        det_model_dir: otx/src/otx/algorithms/detection/configs/detection
        dataset_path: dataset
    variables:
        model:
        - cspdarknet_yolox
        - mobilenetv2_atss
        dataset:
        - diopsis/12
    repeat: 2
    command:
        -  otx train ${det_model_dir}/${model}/template.yaml
            --train-data-roots ${dataset_path}/${dataset}
            --val-data-roots ${dataset_path}/${dataset}
            --track-resource-usage
            params
            --learning_parameters.num_iters 20
        -  otx eval
            --test-data-roots ${dataset_path}/${dataset}
        -  otx export
        -  otx eval
            --test-data-roots ${dataset_path}/${dataset}

Multiple Experiment Recipe YAML File:

    output_path: research_framework_demo/cls_det_model_test
    constants:
        dataset_path: some_dataset_path
    experiments:
    - name det
        constants:
            model_dir: otx/src/otx/algorithms/detection/configs/detection
        variables:
            model:
            - cspdarknet_yolox
            - mobilenetv2_atss
            dataset: diopsis/12
        repeat: 2
        command:
            -  otx train ${model_dir}/${model}/template.yaml ...
            -  otx eval ...
    - name: cls
        constants:
            model_dir: otx/src/otx/algorithms/classification/configs
            dataset_path: other_dataset_path
        variables:
        model:
            - efficientnet_b0_cls_incr
            - deit_tiny
        dataset: cifar10_300
        repeat: 2
        command:
            -  otx train ${model_dir}/${model}/template.yaml
                --train-data-roots ${dataset_path}/${dataset} ...
            -  otx eval ...

Arguments for recipe

- output_path (optional) : Output path where all experiment outputs are saved. Default is "./experiment\_{executed_time}"
- constant (optional) :
  It's similar as constant or variable in programming languages.
  You can use it to replace duplicated string by using ${constant_name} in variables or commands.
- variables (optional) :
  It can be used in a similar way to "constant". But it's different in that "otx experiment" makes all combinations and summarize experiment results based on variables.
  For example, if two models and two dataset are given as variable, then total 4 cases will be run as experiment. Also key of each varaible will be row headers of experiment result table.
- repeat (optional) : Number of times to run experiments. Repeated experiments have different random seeds in "otx train" command.
- command (required) : Specifies the commands to run. Supports both single commands and lists of commands.
- experiments (optional) :
  To perform multiple experiments, the user is required to define a list of experiments in this section.
  Each element in the list can contain all the keys mentioned above, excluding output_path.
  The output path for each experiment is automatically set to output_path/name.
  Values outside the experiment element, except for output_path and constants, will be disregarded.
  If `constants` exsist at both upper most level and experiment element,
  they're merged while experiment element takes precedence.
    - name (required) : Specifies the unique name for the experiment. This name will be utilized as the directory name where the output of the experiment is stored.

Upon completion of each experiment, the results are organized within the own workspace.
Following the conclusion of all experiments, all experiment results are aggregated in two distinct formats:
"all experiments result" and "experiment summary" within the specified output_path.
If the repeat parameter is set to a value greater than 1, the results of repeated experiments are averaged in the summary format.

All TensorBoard log files are automatically copied to the output_path/tensorboard directory.
If you want to run tensorboard with all experiments result, you just need to use it as a tensorboard argument.
If there are failed cases, variables and error logs are both printed and saved as a file after the execution of all commands.

Note that all commands within each case are executed within the same workspace,
obviating the need to set a template path from the second command.
When the "otx eval" or "otx optimize" command is executed, the model file (model weight or exported model, etc.)
is automatically selected based on the preceding command.
The output file of "otx eval" is then stored at "workspace_path/outputs/XXXX\_{train, export, optimize, etc.}/"
under the name "performance.json".

### Feature 2 : organize experiment result from single workspace

Arguments

- -p / --path : Path to the workspace. Experiment results in the workspace are organized and saved.

This feature parses all possible values from a single workspace and saves them as a file.
