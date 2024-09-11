import argparse
from pathlib import Path
import pandas as pd


DATASETS = [
    "Vitens-Coliform-coco",
    "Vitens-Aeromonas-coco",
    "car-seg.v1i.coco-mmdetection",
    "factory_package.v1i.coco-mmdetection",
    "Chicken-Real-Time-coco-roboflow",
    "skindetect-roboflow",
    "wgisd-coco",
    "BlueBerry23.v1i.coco-mmdetection",
]


def read_train_csv(csv_path):
    df = pd.read_csv(csv_path)
    avg_iter_time = df["train/iter_time"].mean()
    avg_data_time = df["train/data_time"].mean()
    total_epoch = df["epoch"].max()
    avg_gpu_meomry = 0.0

    return avg_iter_time, avg_data_time, avg_gpu_meomry, total_epoch


def read_log(log_path):
    elapsed_time = 0.0
    fps = 0.0
    f1_score = 0.0
    f = open(log_path, "r")
    lines = f.readlines()
    for line in lines:
        if "Elapsed time" in line:
            elapsed_time_ls = (line.split("Elapsed time: ")[1].split(" ")[0]).strip()
            elapsed_time_ls = [float(t) for t in elapsed_time_ls.split(":")]
            elapsed_time = elapsed_time_ls[0] * 60 * 60 + elapsed_time_ls[1] * 60 + elapsed_time_ls[2]
            elapsed_time /= 60.0
        if "FPS" in line:
            fps = float(line.split("FPS: ")[1].split(" ")[0])
        if "test/f1-score" in line:
            f1_score = line.strip()
            f1_score = f1_score.replace("test/f1-score", "").replace(" ", "").replace("│", "")
            f1_score = float(f1_score)
    return elapsed_time, fps, f1_score


def collect_results(folder):
    model_folder = Path(folder)
    print(model_folder.name)
    total_time = 0
    full_string = ""
    for i, dataset_name in enumerate(DATASETS):
        dataset_folder = model_folder / f"{dataset_name}"
        output_folder = dataset_folder / ".latest"
        try:
            train_folder = list(output_folder.glob("train"))[0]
            train_csv = train_folder / "csv" / "version_0" / "metrics.csv"
            train_log = dataset_folder / "train_raw.log"

            elapsed_time, _ , _ = read_log(train_log)
            _, cuda_fps, torch_accuracy = read_log(dataset_folder / "torch_test_raw.log")
            _, ov_fps, ov_accuracy = read_log(dataset_folder / "ov_test_raw.log")
            cuda_latency = 1/cuda_fps
            ov_latency = 1/ov_fps

            avg_iter_time, avg_data_time, avg_gpu_meomry, total_epoch = read_train_csv(train_csv)
            total_time += elapsed_time
            full_string += f"{torch_accuracy},{ov_accuracy},{avg_iter_time},{elapsed_time},{ov_latency},{cuda_latency},{total_epoch},"
        except:
            print(f"{dataset_name} error")
            full_string += f"{0.0},{0.0},{0.0},{0.0},{0.0},{0.0},{0.0},"
    print(full_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()
    collect_results(args.folder)
