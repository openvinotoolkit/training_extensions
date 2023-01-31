OpenVINO Training Extension interacts with the anomaly detection library ([Anomalib](https://github.com/openvinotoolkit/anomalib)) by providing interfaces in the `external/anomaly` of this repository. The `sample.py` file contained in this folder serves as an end-to-end example of how these interfaces are used. To begin using this script, first ensure that `otx_cli`, `otx_sdk` and `external/anomaly` dependencies are installed.

To get started, we provide a handy script in `adapters/anomalib/data/create_mvtec_ad_json_annotations.py` to help generate annotation json files for MVTec dataset. Assuming that you have placed the MVTec dataset in a directory your home folder (`~/dataset/MVTec`), you can run the following command to generate the annotations.

```bash
python create_mvtec_ad_json_annotations.py --data_path ~/datasets/MVTec --annotation_path ~/training_extensions/data/MVtec/
```

This will generate three folders in `~/training_extensions/data/MVtec/` for classification, segmentation and detection task.

Then, to run sample.py you can use the following command.

```bash
python tools/sample.py \
    --dataset_path ~/datasets/MVTec \
    --category bottle \
    --train-ann-files ../../data/MVtec/bottle/segmentation/train.json \
    --val-ann-files ../../data/MVtec/bottle/segmentation/val.json \
    --test-ann-files ../../data/MVtec/bottle/segmentation/test.json \
    --model_template_path ./configs/anomaly_segmentation/padim/template.yaml
```

Optionally, you can also optimize to `nncf` or `pot` by using the `--optimization` flag
