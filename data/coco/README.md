COCO (Common Objects in Context) Dataset: http://cocodataset.org

Prepare the dataset:
1. Download the dataset from link above.
2. Update annotation with script `add_full_image_path.py`.
    ```
    python add_full_image_path.py instances_train2017.json ./train2017 instances_train2017_full_paths.json 

    ```