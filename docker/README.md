## Step 1. Install docker

Review the instructions for installation docker [here](https://docs.docker.com/engine/install/ubuntu/) and configure Docker
to use a proxy server as [here](https://docs.docker.com/network/proxy/#configure-the-docker-client).

## Step 2. Install nvidia-docker

*Skip this step if you don't have GPU.*

Review the instructions for installation docker [here](https://github.com/NVIDIA/nvidia-docker).


## Step 4. Build image

In the project folder run in terminal:
```
sudo docker image build --network=host -t ote <clonned root folder>/docker
```

By default the image will created with the  MMDetection algorithm virtual environment.

If you need other algorithm please use --build-arg install_be=[OTE Algorithm] option.

```
sudo docker image build --network=host -t ote <clonned root folder>/docker --build-arg install_be=<OTE Algorithm>
```

Available OTE Algorithms:

|OTE Algorithms| Virtual environment for |
| :--- | :--- |
| anomaly |  Anomaly Classification, Detection and Segmentation |
| deep-object-reid | Image Classification |
| mmdetection | Object Detection, Counting, Rotated Object Detection |
| mmsegmentation | Semantic Segmentation |

Use `--network` to duplicate the network settings of your localhost into context build.

## Step 5. Run container
Run in terminal:
```
sudo docker run \
-it \
--name=<CONTAINER_NAME> \
--runtime=nvidia \
--network=host \
--shm-size=1g \
--ulimit memlock=-1 \
<IMAGE_ID>
 ```

You should not use `--runtime=nvidia` if you don't have GPU.

Use `--shm-size` to increase the size of the shared memory directory.

Now you can use ote on your container