import os
import json
import sys
import pytest
import subprocess
import re
import shlex
from prettytable import PrettyTable
from collections import OrderedDict
from yattag import Doc
from pathlib import Path
from tests.conftest import TEST_ROOT, PROJECT_ROOT

BG_COLOR_GREEN_HEX = 'ccffcc'
BG_COLOR_YELLOW_HEX = 'ffffcc'
BG_COLOR_RED_HEX = 'ffcccc'

DIFF_TARGET_MIN_GLOBAL = -0.1
DIFF_TARGET_MAX_GLOBAL = 0.1
DIFF_FP32_MIN_GLOBAL = -1.0
DIFF_FP32_MAX_GLOBAL = 0.1

results_path = str(PROJECT_ROOT)


class TestSotaCheckpoints:
    param_list = []
    train_param_list = []
    ids_list = []
    train_ids_list = []
    row_dict = OrderedDict()
    color_dict = OrderedDict()
    test = None
    cmd = "{} examples/{sample_type}/main.py -m {} --config {conf} \
         --data {dataset}/{data_name}/ --log-dir={res}/logs/ --metrics-dump \
          {res}/{mod_name}.json"

    @staticmethod
    def run_cmd(comm):
        print()
        print(comm)
        print()
        com_line = shlex.split(comm)
        result = subprocess.Popen(com_line, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        err_string = None
        while result.poll() is None:
            stdout_line = result.stdout.readline().decode('utf-8').strip()
            if re.search("Error:", stdout_line):
                err_string = stdout_line
            if stdout_line != "":
                print(stdout_line)
        return err_string

    @staticmethod
    def make_table_row(test, expected_, metrics_type_, key, error_message, metric, diff_target, fp32_metric_=None,
                       diff_fp32=None):
        TestSotaCheckpoints.test = test
        if metric != 0:
            if fp32_metric_ is None:
                fp32_metric_ = "-"
                diff_fp32 = "-"
            if test == 'eval':
                row = [str(key), str(expected_), str(metric), str(fp32_metric_), str(metrics_type_),
                       str(diff_fp32), str(diff_target), str("-")]
            else:
                row = [str(key), str(expected_), str(metric), str(metrics_type_), str(diff_target), str("-")]
        else:
            if fp32_metric_ is None:
                fp32_metric_ = "-"
            if test == 'eval':
                row = [str(key), str(expected_), str("Not executed"), str(fp32_metric_), str(metrics_type_),
                       str("-"), str("-"), str(error_message)]
            else:
                row = [str(key), str(expected_), str("Not executed"), str(metrics_type_), str("-"), str(error_message)]
        return row

    def write_results_table(self, init_table_string):
        result_table = PrettyTable()
        result_table.field_names = init_table_string
        for key in self.row_dict:
            result_table.add_row(self.row_dict[key])
        print()
        print(result_table)

        doc, tag, text = Doc().tagtext()
        doc.asis('<!DOCTYPE html>')
        with tag('p'):
            text('legend: ')
        with tag('p'):
            with tag('span', style="Background-color: #{}".format(BG_COLOR_GREEN_HEX)):
                text('Thresholds for FP32 and Expected are passed')
        with tag('p'):
            with tag('span', style="Background-color: #{}".format(BG_COLOR_YELLOW_HEX)):
                text('Thresholds for Expected is failed, but for FP32 passed')
        with tag('p'):
            with tag('span', style="Background-color: #{}".format(BG_COLOR_RED_HEX)):
                text('Thresholds for FP32 and Expected are failed')
        with tag('table', border="1", cellpadding="5", style="border-collapse: collapse; border: 1px solid black;"):
            with tag('tr'):
                for i in init_table_string:
                    with tag('td'):
                        text(i)
            for key in self.row_dict:
                with tag('tr', bgcolor='{}'.format(self.color_dict[key])):
                    for i in self.row_dict[key]:
                        if i is None:
                            i = '-'
                        with tag('td'):
                            text(i)
        f = open('results.html', 'w')
        f.write(doc.getvalue())
        f.close()

    @staticmethod
    def threshold_check(err, diff_target, diff_fp32_min_=None, diff_fp32_max_=None, fp32_metric=None,
                        diff_fp32=None, diff_target_min=None, diff_target_max=None):
        color = BG_COLOR_RED_HEX
        within_thresholds = False
        if not diff_target_min:
            diff_target_min = DIFF_TARGET_MIN_GLOBAL
        if not diff_target_max:
            diff_target_max = DIFF_TARGET_MAX_GLOBAL
        if not diff_fp32_min_:
            diff_fp32_min_ = DIFF_FP32_MIN_GLOBAL
        if not diff_fp32_max_:
            diff_fp32_max_ = DIFF_FP32_MAX_GLOBAL
        if err is None:
            if fp32_metric is not None:
                if diff_fp32_min_ < diff_fp32 < diff_fp32_max_ and diff_target_min < diff_target < diff_target_max:
                    color = BG_COLOR_GREEN_HEX
                    within_thresholds = True
                elif diff_fp32_min_ < diff_fp32 < diff_fp32_max_:
                    color = BG_COLOR_YELLOW_HEX
            elif diff_target_min < diff_target < diff_target_max:
                color = BG_COLOR_GREEN_HEX
                within_thresholds = True
        return color, within_thresholds

    @staticmethod
    def write_common_metrics_file():
        metric_value = OrderedDict()
        for i in TestSotaCheckpoints.ids_list:
            with open('{}.json'.format(i)) as metric_file:
                metrics = json.load(metric_file)
            metric_value[i] = metrics['Accuracy']

            if os.path.isfile('metrics.json'):
                path = Path('metrics.json')
                data = json.loads(path.read_text(encoding='utf-8'))
                data.update(metric_value)
                path.write_text(json.dumps(data, indent=4), encoding='utf-8')
            else:
                with open('metrics.json', 'w') as outfile:
                    json.dump(metric_value, outfile)

    @staticmethod
    def read_metric(model_name_):
        with open('{}.json'.format(model_name_)) as metric_file:
            metrics = json.load(metric_file)
        return metrics['Accuracy']

    sota_eval_config = json.load(open('{}/sota_checkpoints_eval.json'.format(TEST_ROOT)),
                                 object_pairs_hook=OrderedDict)
    for sample_type_ in sota_eval_config:
        datasets = sota_eval_config[sample_type_]
        for dataset_name in datasets:
            model_dict = datasets[dataset_name]
            for model_name in model_dict:
                config_name = model_dict[model_name].get('config', {})
                reference = None
                if model_dict[model_name].get('reference', {}):
                    reference = model_dict[model_name].get('reference', {})
                expected = model_dict[model_name].get('target', {})
                metric_type = model_dict[model_name].get('metric_type', {})
                if model_dict[model_name].get('resume', {}):
                    resume_file = model_dict[model_name].get('resume', {})
                else:
                    resume_file = None
                if model_dict[model_name].get('batch', {}):
                    batch = model_dict[model_name].get('batch', {})
                else:
                    batch = None
                diff_fp32_min = model_dict[model_name].get('diff_fp32_min') if not None else None
                diff_fp32_max = model_dict[model_name].get('diff_fp32_max') if not None else None
                diff_target_min = model_dict[model_name].get('diff_target_min') if not None else None
                diff_target_max = model_dict[model_name].get('diff_target_max') if not None else None
                param_list.append((config_name, reference, expected, metric_type, dataset_name, sample_type_,
                                   resume_file, batch, diff_fp32_min, diff_fp32_max, model_name, diff_target_min,
                                   diff_target_max))
                ids_list.append(model_name)
                if model_dict[model_name].get('compression_description', {}):
                    train_param_list.append((config_name, expected, metric_type, dataset_name, sample_type_,
                                             model_name))
                    train_ids_list.append(model_name)

    @pytest.mark.parametrize("config_name_, reference_, expected_, metric_type_, dataset_name_, _sample_type_,"
                             " resume_file_, batch_, diff_fp32_min_, diff_fp32_max_, model_name_, diff_target_min_, "
                             "diff_target_max_", param_list,
                             ids=ids_list)
    def test_eval(self, sota_checkpoints_dir, sota_data_dir, config_name_, reference_, expected_, metric_type_,
                  dataset_name_, _sample_type_, resume_file_, batch_, diff_fp32_min_, diff_fp32_max_, model_name_,
                  diff_target_min_, diff_target_max_):
        test = "eval"
        os.chdir(results_path)
        cmd = self.cmd.format(sys.executable, 'test', conf=config_name_, dataset=sota_data_dir, data_name=dataset_name_,
                              sample_type=_sample_type_, res=results_path, mod_name=model_name_)
        if resume_file_:
            resume = resume_file_
            cmd += " --resume {}/{}".format(sota_checkpoints_dir, resume)
        else:
            cmd += " --pretrained"
        if batch_:
            cmd += " -b {}".format(batch_)
        err = self.run_cmd(cmd)
        metric_value = self.read_metric(model_name_)

        fp32_metric = None
        if reference_ is not None:
            with open('{}.json'.format(reference_)) as ref_metric:
                metrics = json.load(ref_metric)
            fp32_metric = metrics['Accuracy']

        diff_target = round((metric_value - expected_), 2)
        diff_fp32 = round((metric_value - fp32_metric), 2) if fp32_metric is not None else None

        self.row_dict[model_name_] = self.make_table_row(test, expected_, metric_type_, model_name_, err,
                                                         metric_value, diff_target, fp32_metric, diff_fp32)
        self.color_dict[model_name_], is_accuracy_within_thresholds = self.threshold_check(err, diff_target,
                                                                                           diff_fp32_min_,
                                                                                           diff_fp32_max_,
                                                                                           fp32_metric,
                                                                                           diff_fp32,
                                                                                           diff_target_min_,
                                                                                           diff_target_max_)
        assert is_accuracy_within_thresholds

    @pytest.mark.parametrize("config_name_, expected_, metric_type_, dataset_name_, _sample_type_, model_name_",
                             train_param_list, ids=train_ids_list)
    def test_train(self, sota_data_dir, config_name_, expected_, metric_type_, dataset_name_, _sample_type_,
                   model_name_):
        os.chdir(results_path)
        test = 'train'
        cmd = self.cmd.format(sys.executable, 'train', conf=config_name_, dataset=sota_data_dir,
                              data_name=dataset_name_, sample_type=_sample_type_, res=results_path,
                              mod_name=model_name_)
        err = self.run_cmd(cmd)
        metric_value = self.read_metric(model_name_)
        diff_target = round((metric_value - expected_), 2)
        self.row_dict[model_name_] = self.make_table_row(test, expected_, metric_type_, model_name_, err, metric_value,
                                                         diff_target)
        self.color_dict[model_name_], is_accuracy_within_thresholds = self.threshold_check(err, diff_target)
        assert is_accuracy_within_thresholds


Tsc = TestSotaCheckpoints


@pytest.fixture(autouse=True, scope="module")
def skip_params(sota_data_dir):
    if sota_data_dir is None:
        pytest.skip('Path to datasets is not set')


@pytest.fixture(autouse=True, scope="class")
def results():
    yield
    Tsc.write_common_metrics_file()
    if Tsc.test == "eval":
        header = ["Model", "Expected", "Measured", "Reference FP32", "Metrics type", "Diff FP32", "Diff Expected",
                  "Error"]
    else:
        header = ["Model", "Expected", "Measured", "Metrics type", "Diff Expected", "Error"]
    Tsc().write_results_table(header)
