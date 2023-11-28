Helper for conducting experiments
=================================

When you conduct experiments using OTX, you need to execute all necessary cases and parse proper values from output files, which is time consuimg work.
To reduce that effort, you can use "experiment.py" which automatically run all possible cases and parse values.
And also it provids additional convenient features.


It's festures are as below:

* Given multiple varialbes, it automatically collects all combinations and runs them.
* Each experiment results are organized and saved as a file.
* All experiments results are aggregated after all jobs are done.
* If there are some jobs where error is raised, subsequent jobs will be run independently.
* All failed jobs are listed after all jobs are done.
* Proper model files are selected when "otx eval" is executed.
* Organize all existing values from single workspace and save them as a file.


How to use it
-------------
It has two features now.

If you pass experiment recipe, it automatically counts all available cases and run them.
After every runs are done, experiment results are aggregated and  aggregated results are saved as a file
If you pass workspace, every results which exist are automatically aggregated and save it as a file.


Arguments
* -f / --file : path of file descripting how to run experiments. If it's given, all experiments will be run and results are aggregated.
* -p / --path : path of workspace. If it's given, experiment results in the workspace are aggregated and save it as a file.

You can't set both arguments. Please use either "-f" or "-p".


Feature 1 : run experiments ( -f / --file )
This is a sample experiment recipe yaml file

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


Arguments for recipe
* output_path (optional) : The output path where all experiments outputs are saved. If it isn't set, "./experiment_{executed_time}" is set as default.
* constant (optional) : 
        it's similar as constant or variable in programming language.
        You can use it to replace duplicated string by using ${constant_name} in variable or command.
* variables (optional) : 
        it can be used in a similar way as "constant". But it's different in that "otx experiment" makes all combinations based on variables and summarize experiment results based on the variables.
        For example, if two model and two dataset are given as variable, then 4 cases will be run automatically. Also key of each varaible are added in a column of experiment result table.
* repeat (optional) : you can set how many times you want to run experiments. All repeated experiment have different random seed in "otx train" command.
* command (required) : what you want to run. You can write both single command or list of commands.


After each experiments are done, experiment results are aggregated in own workspace.
After all experiments are done, all experiments results are aggregated in two format, "all experiments result" and "experiment summary" in output_path.
If you set more than 1 to repeat, Averaged experiment results are provided in summary format.
All tensorboard log files are copied at output_path/tensorboard directory. If you want to run tensorboard with all experiments result, you just need to use it as a tensorboard argument.
If there are failed cases, variables and error log of failed cases are printed after all job is done and saved as a file.
All of this command is executed in a same workspace. It means that you don't have to set template path from the second otx cli.
When "otx eval" is executed, model file (model weight or exported model, etc.) is automatically selected according to which command is executed just before.
Output file of "otx eval" is saved at "workspace_path/outputs/XXXX_{train, export, optimize, etc.}/" under the name "performance.json"


Feature 2 : aggregate experiment result from single workspace ( -p / --parse )


You can aggregate experiment result from single workspace.
It tries to parse all results if values exist and saves them as a file.
