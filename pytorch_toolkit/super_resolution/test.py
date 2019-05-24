import argparse
import dataset
import metrics
import time
import torch.utils.data as Data
from tqdm import tqdm
import train

parser = argparse.ArgumentParser(description="PyTorch SR test")
parser.add_argument("--test_data_path", default="", type=str, help="path to test data")
parser.add_argument("--exp_name", default="test", type=str, help="experiment name")
parser.add_argument("--models_path", default="/models", type=str, help="path to models folder")
parser.add_argument("--scale", type=int, default=4, help="Upsampling factor for SR")
parser.add_argument("--border", type=int, default=4, help="Ignored border")


def main():
    opt = parser.parse_args()
    print(opt)

    test_set = dataset.DatasetFromSingleImages(path=opt.test_data_path, patch_size=None, aug_resize_factor_range=None, scale=opt.scale)

    batch_sampler = Data.BatchSampler(
        sampler=Data.SequentialSampler(test_set),
        batch_size=1,
        drop_last=True
    )

    evaluation_data_loader = Data.DataLoader(dataset=test_set, num_workers=0, batch_sampler=batch_sampler)

    trainer = train.Trainer(name=opt.exp_name, models_root=opt.models_path, resume=True)
    trainer.load_best()

    psnr = metrics.PSNR(name='PSNR', border=opt.border)

    tic = time.time()
    count = 0
    for batch in tqdm(evaluation_data_loader):
        output = trainer.predict(batch=batch)

        psnr.update(batch[1], output)
        count += 1

    toc = time.time()

    print("FPS: {}, SAMPLES: {}".format(float(count) / (toc - tic), count))
    print("PSNR: {}".format(psnr.get()))


if __name__ == "__main__":
    main()
