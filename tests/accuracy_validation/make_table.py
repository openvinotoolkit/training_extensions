import os
import json
import csv
from collections import OrderedDict
from pathlib import Path
from yattag import Doc

TEST_ROOT = Path(__file__).parent.resolve()
PROJECT_ROOT = TEST_ROOT.parent.parent

BG_COLOR_GREEN_HEX = 'ccffcc'
BG_COLOR_RED_HEX = 'ffcccc'
param_list = []
ids_list = []
DIFF_TARGET_MIN_GLOBAL = -0.1
DIFF_TARGET_MAX_GLOBAL = 0.1
full_results_file = "full_results.csv"
row_dict = OrderedDict()
color_dict = OrderedDict()
ac_metric = {}


try:
    input_csv_file = csv.DictReader(open(full_results_file))
    for row in input_csv_file:
        csv_string = row["metric_value"]
        try:
            value = round(float(csv_string) * 100, 2)
        except ValueError as e:
            value = str(csv_string)
        ac_metric[row['model']] = value
except FileNotFoundError as e:
    print(e)


def make_table_row(expected_, key, error_message, metric, diff_target):
    if metric is not None:
        table_row = [str(key), str(expected_), str(metric), str(diff_target), str('-')]
    else:
        table_row = [str(key), str(expected_), str('Not executed'), str('-'), str(error_message)]
    return table_row


def write_results_table(init_table_string):
    doc, tag, text = Doc().tagtext()
    doc.asis('<!DOCTYPE html>')
    with tag('p'):
        text('legend: ')
    with tag('p'):
        with tag('span', style='Background-color: #{}'.format(BG_COLOR_GREEN_HEX)):
            text('Thresholds Expected are passed')
    with tag('p'):
        with tag('span', style='Background-color: #{}'.format(BG_COLOR_RED_HEX)):
            text('Thresholds for Expected are failed, or model was not evaluate')
    with tag('p'):
        text('If Reference FP32 value in parentheses, it takes from "target" field of .json file')
    with tag('table', border='1', cellpadding='5', style='border-collapse: collapse; border: 1px solid;'):
        with tag('tr'):
            for i in init_table_string:
                with tag('td'):
                    text(i)
        for key in row_dict:
            with tag('tr', bgcolor='{}'.format(color_dict[key])):
                for i in row_dict[key]:
                    if i is None:
                        i = '-'
                    with tag('td'):
                        text(i)
    f = open(PROJECT_ROOT / 'results.html', 'w')
    f.write(doc.getvalue())
    f.close()



eval_config = json.load(open(TEST_ROOT / 'ote_accuracy_validation.json'), object_pairs_hook=OrderedDict)
for domain_name in eval_config:
    model_type = eval_config[domain_name]
    for problem_name in model_type:
        model_dict = model_type[problem_name]
        for model in model_dict:
            test_id = model_dict[model].get('test_id', {})
            expected = model_dict[model].get('target', {})
            alt_export = model_dict[model].get('subfolder')
            

            if metric_value != 0:
                diff_target = round((metric_value - expected), 2)
                diff_fp32 = round((metric_value - fp32_metric), 2) if fp32_metric != "-" else "-"

                row_dict[model_name] = [str(model_name), str(metric_type), str(expected), str(metric_value),
                                        str(fp32_metric), str(diff_fp32), str(diff_target)]
            else:
                row_dict[model_name] = [str(model_name), str(metric_type), str(expected), str("Not executed"),
                                        str("-"), str("-"), str("-")]

            if reference is None and model_name in ac_metric:
                fp32_ov_metric = "-"
                if ac_metric[model_name] != 0:
                    ref_fp32_ov_dict[model_name] = ac_metric[model_name]
            elif reference in ac_metric:
                fp32_ov_metric = ref_fp32_ov_dict[reference]
            else:
                fp32_ov_metric = fp32_metric
            if model_name in ac_metric:
                diff_target = round((ac_metric[model_name] - expected), 2)
                diff_fp32 = round((ac_metric[model_name] - fp32_ov_metric), 2) if fp32_ov_metric != "-" else "-"
                row_dict_ov[model_name] = [str(ac_metric[model_name]), str(fp32_ov_metric), str(diff_fp32),
                                           str(diff_target)]
            else:
                row_dict_ov[model_name] = [str("Not executed"), str(fp32_ov_metric), str("-"), str("-")]

            if reference is None and model_name in ac_metric_q_dq:
                fp32_ov_q_dq_metric = "-"
                if ac_metric_q_dq[model_name] != 0:
                    ref_fp32_ov_q_dq_dict[model_name] = ac_metric_q_dq[model_name]
            elif reference in ac_metric_q_dq:
                fp32_ov_q_dq_metric = ref_fp32_ov_q_dq_dict[reference]
            else:
                fp32_ov_q_dq_metric = fp32_metric
            if model_name in ac_metric_q_dq:
                diff_target = round((ac_metric_q_dq[model_name] - expected), 2)
                diff_fp32 = round((ac_metric_q_dq[model_name] - fp32_ov_q_dq_metric), 2)\
                    if fp32_ov_q_dq_metric != "-" else "-"
                row_dict_ov_q_dq[model_name] = [str(ac_metric_q_dq[model_name]), str(fp32_ov_q_dq_metric),
                                                str(diff_fp32), str(diff_target)]
            else:
                row_dict_ov_q_dq[model_name] = [str("Not executed"), str(fp32_ov_q_dq_metric), str("-"), str("-")]

            color_dict[model_name] = threshold_check(row_dict[model_name], diff_fp32_min, diff_fp32_max,
                                                     diff_target_min, diff_target_max)
            color_dict_ov[model_name] = threshold_check_ov(row_dict_ov[model_name], diff_fp32_min, diff_fp32_max,
                                                           diff_target_min, diff_target_max)
            color_dict_ov_q_dq[model_name] = threshold_check_ov(row_dict_ov_q_dq[model_name], diff_fp32_min,
                                                                diff_fp32_max, diff_target_min, diff_target_max)
            table_row[model_name] = row_dict[model_name] + row_dict_ov[model_name] + row_dict_ov_q_dq[model_name]







header = ['Model', 'Expected', 'Measured', 'Diff Expected', 'Error']
write_results_table(header)
