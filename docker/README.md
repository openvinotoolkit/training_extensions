## Step 1. Install docker

Review the instructions for installation docker [here](https://docs.docker.com/engine/install/ubuntu/) and configure Docker
to use a proxy server as [here](https://docs.docker.com/network/proxy/#configure-the-docker-client).

## Step 2. Install nvidia-docker

*Skip this step if you don't have GPU.*

Review the instructions for installation docker [here](https://github.com/NVIDIA/nvidia-docker).

## Step 3. Build image

In the project folder run in terminal:
```
sudo docker image build --network=host -t ote:test <PATH_TO_DIR_WITH_DOCKERFILE>
```

Use `--network` to duplicate the network settings of your localhost into context build.

## Step 4. Run container
Run in terminal:
```
sudo docker run \
-it \
--name=<CONTAINER_NAME> \
--runtime=nvidia \
--network=host \
--shm-size=1g \
--ulimit memlock=-1 \
--mount type=bind,source=<PATH_TO_OTE_HOME_ON_HOST>,target=/home/training_extention \
<IMAGE_ID>
 ```

You should not use `--runtime=nvidia` if you don't have GPU.

Use `--shm-size` to increase the size of the shared memory directory.

