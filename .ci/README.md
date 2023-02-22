# Guide to Setting up the CI using the Docker images

1. Build the docker image using the `build.sh` in the .ci directory.
   Make sure you are in the root directory of `training_extensions`.

   ```bash
   training_extensions$ .ci/build.sh --help
     USAGE: .ci/build.sh <tag> [Options]
     Options
        -p|--push           Push built image(s) to registry
        -u|--url            url to get Github actions-runner package
        -c|--cuda           Specify CUDA version
        -h|--help           Print this message
   ```

   Below example builds an image using actions-runner v2.299.1 based on `NVIDIA CUDA 11.1.1` image and tag it as `2.299.1`.

   ```bash
   training_extensions$ .ci/build.sh 2.299.1 -u https://github.com/actions/runner/releases/download/v2.299.1/actions-runner-linux-x64-2.299.1.tar.gz -c 11.1.1
   ```

   > **_Note_**: While building an image, script will use your system's environment variables `http_proxy`, `https_proxy`, and `no_proxy`. If you need to use proxy to access external entity, please check those settings before using this script.

   > **_Note_**: You can get the latest version of Github actions-runner package downloading URL from here - https://github.com/actions/runner/releases

1. Create a container and start runner

   ```bash
   training_extensions$ .ci/start-runner.sh --help
     USAGE: .ci/start-runner.sh <container-name> <github-token> <codacy-token> [Options]
         Options
             -g|--gpu-ids        GPU ID or IDs (comma separated) for runner or 'all'
             -c|--cuda           Specify CUDA version
             -d|--debug          Flag to start debugging CI container
             -h|--help           Print this message
   ```

   Below example starts a runner named as `otx-ci-container` with GPU ID 0

   ```bash
   training_extensions$ .ci/start-runner.sh otx-ci-container <github-token> <codacy-token> -g 0
   ```

   Once it executed, An interactive session for the runner configuration will be started and you should input some information to each prompt.

   - `Enter the name of the runner group to add this runner to:`
     - Just press Enter key for default
   - `Enter the name of runner:`
     - Input the runner name which would be used by Github. If you want to replace an existing runner to this one, input the same runner's name.
   - [`Optional`] If you input the existing runner's name to replace in the step above, confirmation prompt will be shown.
     - Input 'y' for the replacing
     - `Enter name of work folder:`
     - Just press Enter key for default

   All configurations were configured and the runner is started successfully, you can see the messages below.

   ```
   v Settings Saved.

   Successfully started actions runner
   ```

   > **_Note_**: About to getting tokens that used in the command above, you need to have proper permission to this repository. Please contact the repo admin to discuss futher.

   > **_Note_**: If there is no docker image for the OpenVINO™ Training Extensions CI on the host machine, this script will pull it from the registry and that will take some time to complete pull operation. It can lead an error on starting runner instance because of the expiring of the given token's validity. In this case, you should re-run the start-runner script again with refreshed token.

1. Stop the runner and running container

   ```bash
   training_extensions$ .ci/stop-runner.sh
     USAGE: .ci/stop-runner.sh <container-name> <github-token> [Options]
     Options
         -h|--help           Print this message
   ```

   Below example stops a runner named as `otx-ci-container`

   ```bash
   training_extensions$ .ci/stop-runner.sh otx-ci-container <github-token>
   ```

   > **_Note_**: If there is an action in progress on the actions-runner which you want to stop, this script will be resulted with an error. To perform force stopping the runner, you can stop the docker container using `docker stop` command on the host machine.

1. Monitor the running runner
   ```bash
   training_extensions$ .ci/check-runner.sh --help
   USAGE: .ci/check-runner.sh <container-name> [Options]
   Options
       -r|--runner         Check runner's log instead of Job one
       -h|--help           Print this message
   ```
