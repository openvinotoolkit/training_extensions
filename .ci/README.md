# Guide to Setting up the CI using the Docker images

1. Build the docker image using the Dockerfile in the .ci directory.
   Make sure you are in the root directory of `training_extensions`.

   ```bash
   sudo docker build --build-arg HTTP_PROXY="$http_proxy" --build-arg HTTPS_PROXY="$https_proxy" --build-arg NO_PROXY="$no_proxy" . -t otx-ci -f .ci/Dockerfile
   ```

   Here, `otx_ci` is the name of the image.

1. Create and start a container

   ```bash
   sudo docker run --gpus all -i -t -d --name otx-ci-container otx-ci
   ```

   Note: `--gpus all` is required for the container to have access to the GPUs.
   `-d flag ensure that the container is detached when it is created.

1. Enter the container by

   ```bash
   sudo docker exec -it  otx-ci-container /bin/bash
   ```

1. Install github actions runner in the container by navigating to [https://github.com/openvinotoolkit/training_extensions/settings/actions/runners/new](https://github.com/openvinotoolkit/training_extensions/settings/actions/runners/new)

   For example:

   ```bash
   mkdir actions-runner && cd actions-runner

   curl -o actions-runner-linux-x64-2.296.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.296.1/actions-runner-linux-x64-2.296.1.tar.gz

   tar xzf ./actions-runner-linux-x64-2.296.1.tar.gz

   rm actions-runner-linux-x64-2.296.1.tar.gz

   ./config.sh --url https://github.com/openvinotoolkit/training_extensions --token <enter-your-token-here>
   ```

   Follow the instructions on the screen to complete the installation.

1. To ensure that coverage report is uploaded to codacy, add the following environment variables to the container:

   ```bash
   export CODACY_PROJECT_TOKEN=<codacy-project-token>
   ```

1. Now the container is ready. Type `exit` to leave the container.

1. Start github actions runner in detached mode in the container by

   ```bash
   sudo docker exec -d otx-ci-container /home/validation/actions-runner/run.sh
   ```
