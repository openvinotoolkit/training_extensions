import shutil
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, default="../../../data/flowers_photo", help="path to data root.")
    args = parser.parse_args()
    root_path = args.data_root
    categories = os.listdir(root_path)
    train_path = os.path.join(root_path, "train")
    val_path =  os.path.join(root_path, "val")
    try:
        os.makedirs(train_path)
        os.makedirs(val_path)
    except FileExistsError:
        # directory already exists
        raise FileExistsError("please, remove train and/or val folders to continue")

    for cat in categories:
        if os.path.isdir(os.path.join(root_path, cat)):
            all_im = os.listdir(os.path.join(root_path, cat))
            trsh = len(all_im) - int(len(all_im)/4)
            train = all_im[:trsh]
            val = all_im[trsh:]
            print(f"{cat}: {len(train)} train images")
            print(f"{cat}: {len(val)} val images")
            os.mkdir(os.path.join(train_path, cat))
            os.mkdir(os.path.join(val_path, cat))
            for im in train:
                shutil.move(os.path.join(root_path, cat, im), os.path.join(train_path, cat, im))
            for im in val:
                shutil.move(os.path.join(root_path, cat, im), os.path.join(val_path, cat, im))

if __name__ == "__main__":
    main()