# Common Objects in Context (COCO) Dataset

Prepare the dataset:

1. Download the [dataset](http://cocodataset.org). 

2. Update the annotation with the `add_full_image_path.py` script:
    ```
    python add_full_image_path.py instances_train2017.json ./train2017 instances_train2017_full_paths.json 

    ```