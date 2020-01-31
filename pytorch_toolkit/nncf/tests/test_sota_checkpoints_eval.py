import subprocess
import re
import os
import json
import sys
import shlex
from prettytable import PrettyTable
from collections import OrderedDict
import pytest
from yattag import Doc


def run_cmd(comm):
    com_line = shlex.split(comm)
    result = subprocess.Popen(com_line, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    err_string = None
    while result.poll() is None:
        stdout_line = result.stdout.readline().decode('utf-8').strip()
        if re.search("Error:", stdout_line):
            err_string = stdout_line
        print(stdout_line)
    return err_string


def table_maker(fp32_metric_, expected_value, metrics_type, key, error_message, metrics_dict, result_table, diff_target,
                diff_fp32):

    if metrics_dict[key] != 0:
        if fp32_metric_ is None:
            fp32_metric_ = "-"
            diff_fp32 = "-"
        row = [str(key), str(expected_value), str(metrics_dict[key]), str(fp32_metric_),
               str(metrics_type), str(diff_fp32), str(diff_target), str("-")]
        result_table.add_row(row)
    else:
        if fp32_metric_ is None:
            fp32_metric_ = "-"
        row = [str(key), str(expected_value), str("Not executed"),
               str("Not executed"), str(fp32_metric_), str(metrics_type), str("-"), str(error_message)]
        result_table.add_row(row)
    return row


def html_maker(table_dict, color_dict, init_table_string):
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
        for key in table_dict:
            with tag('tr', bgcolor='{}'.format(color_dict[key])):
                for i in table_dict[key]:
                    if i is None:
                        i = '-'
                    with tag('td'):
                        text(i)
    f = open('results.html', 'w')
    f.write(doc.getvalue())
    f.close()


def color_check(diff_target, diff_fp32_min, diff_fp32_max, fp32_metric, diff_fp32):
    color = BG_COLOR_RED_HEX
    if fp32_metric is not None:
        if diff_fp32_min < diff_fp32 < diff_fp32_max and diff_target_min < diff_target < diff_target_max:
            color = BG_COLOR_GREEN_HEX
        elif diff_fp32_min < diff_fp32 < diff_fp32_max:
            color = BG_COLOR_YELLOW_HEX
    elif diff_target_min < diff_target < diff_target_max:
        color = BG_COLOR_GREEN_HEX
    return color


@pytest.fixture(scope='module')
def _params(sota_checkpoints_dir, sota_data_dir):
    if sota_checkpoints_dir is None:
        pytest.skip('Path to models is not set')
    if sota_data_dir is None:
        pytest.skip('Path to datasets is not set')
    return {
        'sota_checkpoints_dir': sota_checkpoints_dir,
        'sota_data_dir': sota_data_dir
    }


BG_COLOR_GREEN_HEX = 'ccffcc'
BG_COLOR_YELLOW_HEX = 'ffffcc'
BG_COLOR_RED_HEX = 'ffcccc'
diff_target_min = -0.1
diff_target_max = 0.1
results_path = os.path.split(os.path.dirname(__file__))[0]


def test_eval(_params):
    row_counter = 0
    row_dict = OrderedDict()
    color_dict = OrderedDict()
    os.chdir(results_path)
    nightly = json.load(open('{}/sota_checkpoints_eval.json'.format(os.path.dirname(__file__))),
                        object_pairs_hook=OrderedDict)
    p = _params
    results_dict = OrderedDict()
    result_table = PrettyTable()
    init_table_string = ["Model", "Expected", "Measured", "Reference FP32", "Metrics type", "Diff FP32",
                         "Diff Expected", "Error"]
    result_table.field_names = init_table_string
    models = p['sota_checkpoints_dir']
    dataset = p['sota_data_dir']
    for sample_type_ in nightly:
        datasets = nightly[sample_type_]
        for data_name in datasets:
            model_name = datasets[data_name]
            for configs in model_name:
                conf_file = model_name[configs].get('config', {})
                reference = None
                if model_name[configs].get('reference', {}):
                    reference = model_name[configs].get('reference', {})
                expected = model_name[configs].get('target', {})
                metric_type = model_name[configs].get('metric_type', {})
                cmd = "{} examples/{sample_type}/main.py -m test --config examples/{sample_type}/configs/{} \
                 --data {}/{}/ --log-dir={res}/logs/ --metrics-dump \
                  {res}/metrics.json".format(sys.executable, conf_file, dataset,
                                             data_name, sample_type=sample_type_, res=results_path)
                if model_name[configs].get('resume', {}):
                    resume = model_name[configs].get('resume', {})
                    cmd += " --resume {}/{}".format(models, resume)
                else:
                    cmd += " --pretrained"
                    resume = os.path.basename(conf_file).replace(".json", ".pth")
                if model_name[configs].get('batch', {}):
                    cmd += " -b {}".format(model_name[configs].get('batch', {}))
                err = run_cmd(cmd)
                diff_fp32_min = model_name[configs].get('diff_fp32_min') if not None else None
                diff_fp32_max = model_name[configs].get('diff_fp32_max') if not None else None
                with open('metrics.json', 'r') as json_file:
                    metrics = json.load(json_file)
                fp32_metric = None
                if reference is not None:
                    reference_model = reference + ".pth"
                    fp32_metric = metrics[reference_model]
                diff_target = round((metrics[resume] - expected), 2)
                diff_fp32 = round((metrics[resume] - fp32_metric), 2) if fp32_metric is not None else None
                row_dict[row_counter] = table_maker(fp32_metric, expected, metric_type, resume, err, metrics,
                                                    result_table, diff_target, diff_fp32)
                color_dict[row_counter] = color_check(diff_target, diff_fp32_min, diff_fp32_max, fp32_metric, diff_fp32)
                results_dict[resume] = [fp32_metric, expected]
                row_counter += 1
    print(result_table)
    html_maker(row_dict, color_dict, init_table_string)
    for values in color_dict.values():
        assert values == BG_COLOR_GREEN_HEX
