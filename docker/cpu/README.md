# Setup Docker Image for CPU Training

Steps to setup Docker* image for CPU training using Intel® optimized PyTorch*.

## Step 1. Build image

```
$ docker build -t ote/mmdetection:v0.2.0 . \
  --build-arg subproject=mmdetection
```

Select one of OTE sub-projects:
 - anomaly
 - deep-object-reid
 - mmdetection
 - mmsegmentation
 - model-preparation-algorithm

```
$ docker images
```
```
REPOSITORY                      TAG            IMAGE ID       CREATED          SIZE
ote/mmdetection                 v0.2.0         b303399ddd14   12 seconds ago   6.47GB
intel/intel-optimized-pytorch   1.11.0-conda   95e46843f5d3   3 weeks ago      2.82GB
```

## Step 2. Run container

```
$ docker run -it \
  --shm-size=8g \
  -p 8888:8888 \
  --rm ote/mmdetection:v0.2.0 /bin/bash
```
```
tami@34acbed0e8c7:~/workspace$
```

## Step 3. Activate virtual environment

Inside the container, run following command.

```
$ source /home/tami/workspace/training_extensions/mmdetection_venv/bin/activate
```

## Step 4. Test CLI

Inside the virtual environment, run following command.

```
$ ote find --root /home/tami/workspace/training_extensions/external/mmdetection/
```
```
- id: Custom_Counting_Instance_Segmentation_MaskRCNN_EfficientNetB2B
  name: MaskRCNN-EfficientNetB2B
  path: /home/tami/workspace/training_extensions/external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template.yaml
  task_type: INSTANCE_SEGMENTATION
- id: Custom_Counting_Instance_Segmentation_MaskRCNN_ResNet50
  name: MaskRCNN-ResNet50
  path: /home/tami/workspace/training_extensions/external/mmdetection/configs/custom-counting-instance-seg/resnet50_maskrcnn/template.yaml
  task_type: INSTANCE_SEGMENTATION
...
```

## Step 5. Test notebook

Run notebook server.

```
$ cd /home/tami/workspace/training_extensions/ote_cli/notebooks
```
```
$ jupyter notebook --ip=0.0.0.0
```
```
[I 06:46:12.691 NotebookApp] Serving notebooks from local directory: /home/tami/workspace/training_extensions/ote_cli/notebooks
[I 06:46:12.691 NotebookApp] Jupyter Notebook 6.4.11 is running at:
[I 06:46:12.691 NotebookApp] http://04de438f448e:8888/?token=2bd...2b7
[I 06:46:12.691 NotebookApp]  or http://127.0.0.1:8888/?token=2bd...2b7
[I 06:46:12.691 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 06:46:12.693 NotebookApp]

    To access the notebook, open this file in a browser:
        file:///home/tami/.local/share/jupyter/runtime/nbserver-43-open.html
    Or copy and paste one of these URLs:
        http://04de438f448e:8888/?token=2bd...2b7
     or http://127.0.0.1:8888/?token=2bd...2b7
```

Connect to notebook server using web browser.

## Step 6. Deactivate virtual environment

```
$ deactivate
```

## Step 7. Exit container

```
$ exit
```

---
\* Other names and brands may be claimed as the property of others.
