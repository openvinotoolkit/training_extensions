import json
from pathlib import Path

from otx.cli.registry import Registry
from tests.regression.regression_test_helpers import (
    REGRESSION_TEST_EPOCHS,
    TIME_LOG,
    get_result_dict,
    get_template_performance,
    load_regression_configuration,
)


class RegressionTestBase:
    def __init__(
        self,
        task_type: str,
        train_type: str,
        label_type: str,
        otx_dir: str,
        reg_path: str,
    ):
        self.performance = {}
        self.task_type = task_type
        self.train_type = train_type
        self.label_type = label_type
        self.otx_dir = otx_dir

        self.templates = Registry("otx/algorithms/action").filter(task_type=self.task_type.upper()).templates
        self.templates_ids = [template.model_template_id for template in self.templates]

        self.result_dict = _init_result_dict(self.task_type)
        self.result_dir = f"/tmp/regression_test_results/{self.task_type}"

        Path(self.result_dir).mkdir(parents=True, exist_ok=True)

        self.config = _load_regression_configuration(self.otx_dir, self.task_type, self.train_type, self.label_type)

    def teardown(self):
        with open(f"{self.result_dir}/result.json", "w") as result_file:
            json.dump(self.result_dict, result_file, indent=4)

    @staticmethod
    def _init_result_dict(task_type) -> Dict[str, Any]:
        result_dict = {task_type: {}}
        if "anomaly" not in task_type:
            for label_type in LABEL_TYPES:
                result_dict[task_type][label_type] = {}
                for train_type in TRAIN_TYPES:
                    result_dict[task_type][label_type][train_type] = {}
                    for test_type in TEST_TYPES:
                        result_dict[task_type][label_type][train_type][test_type] = []
        else:
            for test_type in TEST_TYPES:
                result_dict[task_type][test_type] = {}
                for category in ANOMALY_DATASET_CATEGORIES:
                    result_dict[task_type][test_type][category] = []

        return result_dict

    @staticmethod
    def _load_regression_config(otx_dir: str) -> Dict[str, Any]:
        """Load regression config from path.

        Args:
            otx_dir (str): The path of otx root directory

        Returns:
            Dict[str, Any]: The dictionary that includes data roots
        """
        root_path = Path(otx_dir)
        with open(root_path / ("tests/regression/regression_config.json"), "r") as f:
            reg_config = json.load(f)
        return reg_config

    @staticmethod
    def _load_regression_configuration(
        otx_dir: str, task_type: str, train_type: str = "", label_type: str = ""
    ) -> Dict[str, Union[str, int, float]]:
        """Load dataset path according to task, train, label types.

        Args:
            otx_dir (str): The path of otx root directoy
            task_type (str): ["classification", "detection", "instance segmentation", "semantic segmentation",
                                "action_classification", "action_detection", "anomaly"]
            train_type (str): ["supervised", "semi_supervised", "self_supervised", "class_incr"]
            label_type (str): ["multi_class", "multi_label", "h_label", "supcon"]

        Returns:
            Dict[str, Union[int, float]]: The dictionary that includes model criteria
        """
        reg_config = _load_regression_config(otx_dir)
        result: Dict[str, Union[str, int, float]] = {
            "data_path": "",
            "model_criteria": 0,
        }

        data_root = os.environ.get("CI_DATA_ROOT", "/storageserver/pvd_data/otx_data_archive/regression_datasets")

        if "anomaly" not in task_type:
            if train_type == "" or label_type == "":
                raise ValueError()
            result["regression_criteria"] = reg_config["regression_criteria"][task_type][train_type][label_type]
            result["kpi_e2e_train_time_criteria"] = reg_config["kpi_e2e_train_time_criteria"][task_type][train_type][
                label_type
            ]
            result["kpi_e2e_eval_time_criteria"] = reg_config["kpi_e2e_eval_time_criteria"][task_type][train_type][
                label_type
            ]

            # update data_path using data_root setting
            data_paths = reg_config["data_path"][task_type][train_type][label_type]
            for key, value in data_paths.items():
                data_paths[key] = os.path.join(data_root, value)

            result["data_path"] = data_paths
        else:
            result["regression_criteria"] = reg_config["regression_criteria"][task_type]
            result["kpi_e2e_train_time_criteria"] = reg_config["kpi_e2e_train_time_criteria"][task_type]
            result["kpi_e2e_eval_time_criteria"] = reg_config["kpi_e2e_eval_time_criteria"][task_type]

            # update data_path using data_root setting
            data_paths = reg_config["data_path"][task_type]
            for key, value in data_paths.items():
                if key != "train_params":
                    data_paths[key] = os.path.join(data_root, value)

            result["data_path"] = data_paths

        return result

    @staticmethod
    def get_template_performance(results: List[Dict], template: ModelTemplate):
        """Get proper template performance inside of performance list."""
        performance = None
        for result in results:
            template_name = list(result.keys())[0]
            if template_name == template.name:
                performance = result
                break
        if performance is None:
            raise ValueError("Performance is None.")
        return performance
