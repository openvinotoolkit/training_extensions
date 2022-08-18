import argparse
import os
import json
from pyexpat import model
import numpy as np
from subprocess import run
from ote_cli.utils.tests import collect_env_vars
from omegaconf import OmegaConf
from comet_ml import Experiment
import torch


def ote_train(kwargs, env_root):
    command_line = [
        "ote",
        "train",
        kwargs["model_template_path"],
        "--train-ann-file", kwargs["train_ann_file"],
        "--train-data-roots", kwargs["train_data_roots"],
        "--val-ann-file", kwargs["val_ann_file"],
        "--val-data-roots", kwargs["val_data_roots"],
        "--save-model-to", kwargs["save_model_dir"],
    ]
    if "load_weights" in kwargs:
        command_line.extend(
            ["--load-weights", kwargs["load_weights"]]
        )
    if "params" in kwargs:
        command_line.append("params")
        for param in kwargs["params"]:
            command_line.extend(param.split())
    print(command_line)
    assert run(command_line, env=collect_env_vars(env_root)).returncode == 0
    assert os.path.exists(f'{kwargs["save_model_dir"]}/weights.pth')
    assert os.path.exists(f'{kwargs["save_model_dir"]}/label_schema.json')


def ote_eval(model_template_path, test_ann_file, test_data_roots, save_performance, weight_path, env_root):
    command_line = [
        "ote",
        "eval",
        model_template_path,
        "--test-ann-file", test_ann_file,
        "--test-data-roots", test_data_roots,
        "--load-weights", weight_path,
        "--save-performance", save_performance
    ]
    print(command_line)
    assert run(command_line, env=collect_env_vars(env_root)).returncode == 0
    assert os.path.exists(save_performance)


def prepare_train(config, dataset_name, seed, num_sample) -> dict:
    dataset_cfg = config.datasets[dataset_name]
    filename, ext = os.path.splitext(dataset_cfg.annotations_train)
    train_ann_file = f'{filename}_seed{seed}_{num_sample}{ext}'
    train_cfg = dict(
        train_ann_file=os.path.join(dataset_cfg.anno_root, train_ann_file),
        train_data_roots=os.path.join(dataset_cfg.img_root, dataset_cfg.images_train_dir),
        val_ann_file=os.path.join(dataset_cfg.anno_root, dataset_cfg.annotations_val),
        val_data_roots=os.path.join(dataset_cfg.img_root, dataset_cfg.images_val_dir),
        save_model_dir=os.path.join(config.work_dir, dataset_name, f"seed{seed}_{num_sample}"),
        params=config.params
    )
    return train_cfg


def collect_stat(json_path):
    with open(json_path, 'r') as f:
        metric_dict = json.load(f)
    return metric_dict

def main(args):
    if args.key:
        experiment = Experiment(api_key=args.key, project_name=f"Vitens Tiling")
    config = OmegaConf.load(args.yaml)
    prev_saved_model_dir = ""
    metrics = dict()

    for seed in config.seeds:
        for dataset_name, dataset_cfg in config.datasets.items():
            if dataset_name not in metrics:
                metrics[dataset_name] = dict()
            for i, num_sample in enumerate(dataset_cfg.num_samples):
                kwargs = dict(model_template_path=config.template)
                train_dict = prepare_train(config, dataset_name, seed, num_sample)
                if i > 0 and prev_saved_model_dir:
                    train_dict.update(load_weights=os.path.join(prev_saved_model_dir, "weights.pth"))
                kwargs.update(train_dict)
                if args.collect:
                    if os.path.exists(os.path.join(kwargs['save_model_dir'])):
                        test_metric = collect_stat(os.path.join(kwargs['save_model_dir'], 'test_performance.json'))
                        test_f1_metric = collect_stat(
                            os.path.join(kwargs['save_model_dir'], 'test_performance.json.f1.json'))

                        if num_sample not in metrics[dataset_name]:
                            metrics[dataset_name][num_sample] = []
                        metrics[dataset_name][num_sample].append(test_metric['mae%'])
                        # metrics[dataset_name][num_sample].append(round(test_f1_metric['f-measure'], 3))
                else:
                    if not args.skip_train:
                        prev_saved_model_dir = train_dict['save_model_dir']
                        if os.path.exists(train_dict['save_model_dir']):
                            print(f"Skip {dataset_name}-Seed:{seed}-N:{num_sample}")
                            continue
                        ote_train(kwargs, config.env_root)
                    # eval validation
                    if os.path.exists(os.path.join(kwargs['save_model_dir'], 'weights.pth')):
                        ote_eval(
                            model_template_path=config.template,
                            test_ann_file=os.path.join(dataset_cfg.anno_root, dataset_cfg.annotations_val),
                            test_data_roots=os.path.join(dataset_cfg.img_root, dataset_cfg.images_val_dir),
                            save_performance=os.path.join(kwargs['save_model_dir'], 'val_performance.json'),
                            weight_path=os.path.join(kwargs['save_model_dir'], 'weights.pth'),
                            env_root=config.env_root
                        )
                        # eval test
                        ote_eval(
                            model_template_path=config.template,
                            test_ann_file=os.path.join(dataset_cfg.anno_root, dataset_cfg.annotations_test),
                            test_data_roots=os.path.join(dataset_cfg.img_root, dataset_cfg.images_test_dir),
                            save_performance=os.path.join(kwargs['save_model_dir'], 'test_performance.json'),
                            weight_path=os.path.join(kwargs['save_model_dir'], 'weights.pth'),
                            env_root=config.env_root
                        )

    for dataset_name, num_samples in metrics.items():
        for n, seed_performances in num_samples.items():
            avg = np.mean(metrics[dataset_name][n])
            print(f'{dataset_name} - N: {n} - avg: {avg} - {metrics[dataset_name][n]}')


def parse_args():
    parser = argparse.ArgumentParser(description='Run Exp via OTE CLI')
    parser.add_argument('--yaml', default="./params.yml")
    parser.add_argument('--collect', help="Collect experiment results without training", action='store_true')
    parser.add_argument('--skip-train', help="Skip Training", action='store_true')
    parser.add_argument('--key', default="")
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
