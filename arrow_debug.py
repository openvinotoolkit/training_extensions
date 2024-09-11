import argparse
from otx.tools.converter import ConfigConverter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--geti_config_path", type=str)
    parser.add_argument("--arrow_file_path", type=str)
    parser.add_argument("--val_nums_workers", type=int, default=2)
    args = parser.parse_args()

    work_dir = args.work_dir
    geti_config_path = args.geti_config_path
    arrow_file_path = args.arrow_file_path

    otx_config = ConfigConverter.convert(config_path=geti_config_path)

    otx_config["data"]["data_format"] = "arrow"
    otx_config["data"]["train_subset"]["subset_name"] = "TRAINING"
    otx_config["data"]["val_subset"]["subset_name"] = "VALIDATION"
    otx_config["data"]["test_subset"]["subset_name"] = "TESTING"

    otx_config["data"]["val_subset"]["num_workers"] = args.val_nums_workers

    engine, train_kwargs = ConfigConverter.instantiate(
        config=otx_config,
        work_dir=work_dir,
        data_root=arrow_file_path,
    )

    engine.train(**train_kwargs)
