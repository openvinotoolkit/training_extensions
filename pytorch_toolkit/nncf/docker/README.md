## Step 1. Install docker
Review the instructions for installation docker [here](https://docs.docker.com/engine/install/ubuntu/) and configure HTTP or HTTPS proxy behavior as [here](https://docs.docker.com/config/daemon/systemd/).

## Step 2. Install nvidia-docker

*Skip this step if you don't have GPU.*

Review the instructions for installation docker [here](https://github.com/NVIDIA/nvidia-docker)

## Step 3. Build image
In the project folder run in terminal:
 ```
 sudo docker image build --network=host --build-arg http_proxy=http://example.com:80 --build-arg https_proxy=http://example.com:81 --build-arg 
ftp_proxy=http://example.com:80 <PATH_TO_DIR_WITH_DOCKERFILE>
 ```

*Use `--http_proxy` , `--https_proxy`, `--ftp_proxy`, `--network` to duplicate the network settings of your localhost into context build*
  
## Step 4. Run container
Run in terminal:
```
sudo docker run --name <NAME_CONTAINER> --runtime=nvidia -it --network=host --mount type=bind,source=<PATH_TO_DATASETS_ON_HOST>,target=<PATH_TO_DATSETS_IN_CONTAINER>  --mount type=bind,source=<PATH_TO_NNCF_HOME_ON_HOST>,target=/home/nncf/ <ID_IMAGE>
 ```

*You should not use `--runtime=nvidia` if you want to use `--cpu-only` mode.* 

*Use `--shm-size` to increase the size of the shared memory directory.*

Now you have a working container and you can run examples.

