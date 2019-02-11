BIT-Vehicle Dataset: http://iitlab.bit.edu.cn/mcislab/vehicledb/

# Prepare the dataset

1. Download the dataset from link above.
2. Unpack dataset in the directory `images`.
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
3. Downscale images to increase training speed.
    ```
    python <training_toolbox_tensorflow>/tools/downscale_images.py -target_size 512 <training_toolbox_tensorflow>/data/bitvehicle/images

    ```

# Annotation structure

Json files contain annotation in a fairly straightforward structure. There’s 3
top-level arrays: "images", "annotations" and "categories"

1. “images” has records, like,
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
2. “annotation” has records, like,
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
3. “categories” has just 3 records:
```
[
    {"id": 0, "name": "bg", "supercategory": ""},
    {"id": 1, "name": "vehicle", "supercategory": ""},
    {"id": 2, "name": "plate", "supercategory": ""}
]
```

In the example above, the image with the file name `vehicle_0000005.jpg` has
`id` equal to 4. Then the records in `annotation` array say that this image
(`image_id` is equal to 4) has 3 bounding boxes – 2 of them have `category_id`
set to 1 and one has `category_id` set to 2. Now, the last array -
`categories` - tells us that it means that there’s 2 cars and 1 plate in that
image (their coordinates are given in the `bbox` field – `[x, y, width,
height]`).
