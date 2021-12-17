The `handwritten_sample` LMDB dataset is created by running the following scripts:

```
python reshape.py # resize input .jpg images in current folder to size (96, 2000) and save to data/ directory
python create_lmdb_dataset.py --inputPath ./data --gtFile gt.txt --outputPath handwritten_sample/ # create LMDB dataset from the resized images
```

The second script is from [here](https://github.com/clovaai/deep-text-recognition-benchmark#when-you-need-to-train-on-your-own-dataset-or-non-latin-language-datasets).
