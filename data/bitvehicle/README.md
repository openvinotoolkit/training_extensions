# BIT-Vehicle Dataset

## Prepare the Dataset

1. Download the [dataset](http://iitlab.bit.edu.cn/mcislab/vehicledb/).
2. Unpack the dataset in the `images` directory:
    ```
     $ tree
     .
     ├── bitvehicle_test.json
     ├── bitvehicle_train.json
     ├── images
     │   ├── vehicle_0000001.jpg
     │   ├── vehicle_0000002.jpg
     |   ...
     └── README.md
    ```
3. Downscale images to increase training speed:
    ```
    python <training_toolbox_tensorflow>/tools/downscale_images.py -target_size 512 <training_toolbox_tensorflow>/data/bitvehicle/images

    ```

## Annotation Structure

The structure of JSON files is fairly straightforward. There are three
top-level arrays: `images`, `annotations` and `categories`.

**Example of arrays and their fields**

1. `images` records:  
    ```
        {
            "dataset": "BitVehicle",
            "height": 1080,
            "id": 4,
            "width": 1920,
            "file_name":
            "vehicle_0000005.jpg",
            "coco_url": null,
            "license": null,
            "flickr_url": null,
            "image": "./images/vehicle_0000005.jpg",
            "date_captured": null
        }
    ```
2. `annotation` records:
    ```
    [
        {
            "area": 199023.0,
            "id": 10,
            "iscrowd": 0,
            "category_id": 1,
            "is_occluded": false,
            "image_id": 4,
            "segmentation": null,
            "bbox": [512.0, 346.0, 407.0, 489.0],
            "attributes": {}
        },
        {
            "area": 2668.0,
            "id": 11,
            "iscrowd": 0,
            "category_id": 2,
            "is_occluded": false,
            "image_id": 4,
            "segmentation": null,
            "bbox": [638.0, 773.0, 92.0, 29.0],
            "attributes": {}
        },
        {
            "area": 5187.0,
            "id": 12,
            "iscrowd": 0,
            "category_id": 1,
            "is_occluded": true,
            "image_id": 4,
            "segmentation": null,
            "bbox": [1023.0, 0.0, 273.0, 19.0],
            "attributes": {}
        }
    ]
    ```
3. `categories` records:
    ```
    [
        {"id": 0, "name": "bg", "supercategory": ""},
        {"id": 1, "name": "vehicle", "supercategory": ""},
        {"id": 2, "name": "plate", "supercategory": ""}
    ]
    ```

In the example above, the ID of the `vehicle_0000005.jpg` image
equals to `4`, which is represented by `"image_id": 4` that has in the `annotation` array, and has three bounding boxes: two of them have the `category_id`
set to `1` and one has the `category_id` set to `2`. 
From the `categories` array, we learn that there are two vehicles and one plate in that
image (their coordinates are given in the `annotation` array in the `bbox` field with the `[x, y, width, height]` format).
