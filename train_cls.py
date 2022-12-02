import importlib
import os
import shutil
import subprocess
from typing import Dict, List, Optional

from mmcv.utils.config import Config


def main(
    GPUS: int,
    EXP: str,
    DATASETS: List[Dict],
    MODELS: List[str],
    MODES: List[str],
    NUM_DATAS: List[int],
    LRS: List[float],
    SEEDS: List[int],
):

    DATA_ROOT = "/home/aarrufat/Datasets/cls"
    NUM_EPOCHS = 20
    BATCH_SIZE = 32
    cfg = None
    for DATASET in DATASETS:
        for MODEL in MODELS:
            for MODE in MODES:
                assert MODE in ["ce", "bt"]
                for NUM_DATA in NUM_DATAS:
                    for LR in LRS:
                        for seed in SEEDS:
                            # Barlow Twins uses a Two-Crop transform, so we reduce the
                            # batch-size to fit in memory.  Moreover, since we are not using
                            # SAM in BarlowTwins, the time complexity remains the same.
                            if MODE == "bt":
                                BATCHSIZE = BATCH_SIZE // 2
                            else:
                                BATCHSIZE = BATCH_SIZE
                            # set command
                            BASELINE_ROOT = (
                                f"otx/algorithms/classification/configs/{MODEL}"
                            )
                            SELFSL_ROOT = f"{BASELINE_ROOT}/selfsl"
                            RECIPE_ROOT = f"{SELFSL_ROOT}/{DATASET['name']}"
                            OPTIONS = f"batch{BATCHSIZE}_lr{LR}"
                            WORKDIR = f"work_dirs/supcon/classification/{EXP}/{DATASET['name']}/{NUM_DATA}/{OPTIONS}/{MODE}_{MODEL}/seed{seed}"
                            RECIPE = f"{RECIPE_ROOT}/{OPTIONS}/{MODE}_{MODEL}"

                            os.makedirs(RECIPE, exist_ok=True)

                            if MODE != "ce":
                                shutil.copy(
                                    os.path.join(
                                        BASELINE_ROOT, "class_selfsl.yaml"
                                    ),
                                    os.path.join(RECIPE, "class_selfsl.yaml"),
                                )
                                tmp = Config.fromfile(
                                    os.path.join(BASELINE_ROOT, "class_selfsl.yaml")
                                )
                                tmp.dump(os.path.join(RECIPE, "class_selfsl.yaml"))
                            else:
                                tmp = Config.fromfile(
                                    "venv/lib/python3.8/site-packages/recipes/stages/classification/incremental.yaml"
                                )
                                tmp.dump(os.path.join(RECIPE, "incremental.yaml"))

                            ################## update template.yaml ##################
                            if MODE == "ce":
                                if os.path.exists(
                                    os.path.join(BASELINE_ROOT, "template.yaml")
                                ):
                                    shutil.copy(
                                        os.path.join(
                                            BASELINE_ROOT, "template.yaml"
                                        ),
                                        os.path.join(RECIPE, "template.yaml"),
                                    )
                                elif os.path.exists(
                                    os.path.join(
                                        BASELINE_ROOT, "template_experiment.yaml"
                                    )
                                ):
                                    shutil.copy(
                                        os.path.join(
                                            BASELINE_ROOT,
                                            "template_experiment.yaml",
                                        ),
                                        os.path.join(RECIPE, "template.yaml"),
                                    )
                            else:
                                shutil.copy(
                                    os.path.join(
                                        BASELINE_ROOT, "template_selfsl.yaml"
                                    ),
                                    os.path.join(RECIPE, "template.yaml"),
                                )

                            template = Config.fromfile(
                                os.path.join(RECIPE, "template.yaml")
                            )

                            ##########################################################

                            ##################### update hparams #####################

                            template["hyper_parameters"]["base_path"] = (
                                "../" * 4
                                + template["hyper_parameters"]["base_path"]
                            )

                            template["hyper_parameters"]["parameter_overrides"][
                                "learning_parameters"
                            ]["batch_size"]["default_value"] = BATCHSIZE

                            template["hyper_parameters"]["parameter_overrides"][
                                "learning_parameters"
                            ]["learning_rate"]["default_value"] = LR

                            template["hyper_parameters"]["parameter_overrides"][
                                "learning_parameters"
                            ]["num_iters"]["default_value"] = NUM_EPOCHS

                            template.dump(os.path.join(RECIPE, "template.yaml"))

                            ######## updated models and recipes for non-ce trainings ########

                            model_name = "model"
                            if MODE == "bt":
                                model_name = "model_selfsl"

                            #### Model update
                            cfg = importlib.import_module(
                                os.path.join(BASELINE_ROOT, model_name).replace(
                                    "/", "."
                                )
                            )

                            model_cfg_for_update = [
                                k
                                for k in [
                                    "_base_",
                                    "model",
                                    "fp16",
                                    "load_from",
                                    "__width_mult",
                                    "ignore",
                                ]
                                if hasattr(cfg, k)
                            ]
                            new_cfg = {
                                k: getattr(cfg, k) for k in model_cfg_for_update
                            }

                            ## update _base_
                            if "_base_" in new_cfg:
                                new_cfg["_base_"] = [
                                    "../"
                                    * (
                                        len(RECIPE.split("/"))
                                        - len(BASELINE_ROOT.split("/"))
                                    )
                                    + b
                                    for b in new_cfg["_base_"].split("\n")
                                ]  # 4

                            new_cfg = Config(
                                cfg_dict=new_cfg,
                                filename=os.path.join(
                                    BASELINE_ROOT, model_name + ".py"
                                ),
                            )

                            new_cfg.dump(os.path.join(RECIPE, model_name + ".py"))

                            os.makedirs(WORKDIR, exist_ok=True)
                            DATA_PATH = os.path.join(DATA_ROOT, DATASET["name"])
                            command = (
                                f"CUDA_VISIBLE_DEVICES={GPUS} otx train "
                                f"{os.path.join(RECIPE, 'template.yaml')} "
                                f"--train-data-roots={DATA_PATH} "
                                f"--val-data-roots={DATA_PATH} "
                                f"--val-ann-files={DATA_PATH}/test.json "
                                f"--train-ann-files={DATA_PATH}/train_{NUM_DATA}samples_seed{seed}.json "
                                f"--save-model-to={WORKDIR} "
                            )
                            print(command)

                            command += f"2>&1 | tee {WORKDIR}/output.log"
                            subprocess.run(command, shell=True)

                            # move logs in the tmp dir to the real dir
                            with open(f"{WORKDIR}/output.log", "r") as f:
                                logs = f.readlines()
                                dir_logs = [d for d in logs if "work dir = " in d][
                                    0
                                ]
                                work_dir = dir_logs[
                                    dir_logs.rfind("= ") + 2 :
                                ].rstrip()

                            files = os.listdir(work_dir)
                            for file in files:
                                try:
                                    shutil.move(
                                        os.path.join(work_dir, file),
                                        os.path.join(WORKDIR, file),
                                    )
                                except:
                                    pass


if __name__ == "__main__":

    main(
        GPUS=1,
        EXP="test",
        DATASETS=[
            {"name": "autism", "num_classes": 2},
            {"name": "cifar10", "num_classes": 10},
            {"name": "cifar100", "num_classes": 100},
            {"name": "eurosat", "num_classes": 10},
            {"name": "ibean", "num_classes": 3},
            {"name": "malaria", "num_classes": 2},
            {"name": "plant_leaf", "num_classes": 4},
            {"name": "uc_merced", "num_classes": 21},
            {"name": "xray", "num_classes": 2},
        ],
        MODELS=[
            "efficientnet_b0_cls_incr",
            "efficientnet_v2_s_cls_incr",
            "mobilenet_v3_small_cls_incr",
        ],
        MODES=["ce", "bt"],
        NUM_DATAS=[10, 20],
        LRS=[0.005, 0.01],
        SEEDS=[
            4216,
            5059,
            8507,
            9205,
            14356,
            22708,
            22785,
            31095,
            32408,
            32691,
        ],
    )
