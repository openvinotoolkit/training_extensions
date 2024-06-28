# Guide to Setting up the CI using the Docker images

1. Build the docker image using the `build.sh` in the .ci directory.
   Make sure you are in the `.ci/docker` directory of `training_extensions`.

   ```bash
   training_extensions/.ci/docker$ ./build.sh --help
      USAGE: ./build.sh <tag> [Options]
      Positional args
         <tag>               Tag name to be tagged to newly built image
      Options
         -p|--push           Push built image(s) to registry
         -u|--url            url to get Github actions-runner package
         -r|--reg            Specify docker registry URL <default: local>
         -h|--help           Print this message
   ```

   Below example builds an image using actions-runner v2.317.0 based on NVIDIA cuda image and tag it as `2.317.0`.

   ```bash
   training_extensions/.ci/docker$ ./build.sh 2.317.0 -u https://github.com/actions/runner/releases/download/v2.305.0/actions-runner-linux-x64-2.305.0.tar.gz
   ```

   > **_Note_**: While building an image, script will use your system's environment variables `http_proxy`, `https_proxy`, and `no_proxy`. If you need to use proxy to access external entity, please check those settings before using this script.

   <!-- -->

   > **_Note_**: The docker image name will be `<DOCKER_REG_ADDR>/ote/ci/cu<VER_CUDA>/runner:<TAG>`

   <!-- -->

   > **_Note_**: You can get the latest version of Github actions-runner package downloading URL from [here](https://github.com/actions/runner/releases).

1. Create a container and start runner

   ```bash
   training_extensions$ .ci/docker/start-runner.sh --help
      USAGE: .ci/docker/start-runner.sh <container-prefix> <github-token> [Options]
      Positional args
         <container-prefix>  Prefix to the ci container and actions-runner
         <github-token>      Github token string
      Options
         -g|--gpu-ids        GPU ID or IDs (comma separated) for runner or 'all'
         -c|--cuda           Specify CUDA version
         -t|--tag            Specify TAG for the CI container
         -l|--labels         Additional label string to set the actions-runner
         -m|--mount          Dataset root path to be mounted to the started container (absolute path)
         -r|--reg            Specify docker registry URL <default: local>
         -d|--debug          Flag to start debugging CI container
         -a|--attach-cache   Attach host path to the .cache on the container
         -f|--fix-cpus       Specify the number of CPUs to set for the CI container
         -h|--help           Print this message
   ```

   Below example starts a runner named as `<container-prefix>-0` with GPU ID 0 (GPU ID will be attached to both container and runner name)

   ```bash
   training_extensions$ .ci/docker/start-runner.sh <container-prefix> <github-token> -g 0
   ```

   If there exist the container named as same, it will be stopped before starting a new container.

   All configurations were configured and the runner is started successfully, you can see the messages below.

   ```
   v Settings Saved.

   Successfully started actions runner
   ```

   > **_Note_**: About to getting tokens that used in the command above, you need to have proper permission to this repository. Please contact the repo admin to discuss futher.

   <!-- -->

   > **_Note_**: If there is no docker image for the OpenVINO™ Training Extensions CI on the host machine, this script will pull it from the registry and that will take some time to complete pull operation. It can lead an error on starting runner instance because of the expiring of the given token's validity. In this case, you should re-run the start-runner script again with refreshed token.

1. Stop the runner and running container

   ```bash
   training_extensions$ .ci/docker/stop-runner.sh
     USAGE: .ci/docker/stop-runner.sh <container-name> <github-token> [Options]
     Options
         -h|--help           Print this message
   ```

   Below example stops a runner named as `otx-ci-container`

   ```bash
   training_extensions$ .ci/docker/stop-runner.sh otx-ci-container <github-token>
   ```

   > **_Note_**: If there is an action in progress on the actions-runner which you want to stop, this script will be resulted with an error. To perform force stopping the runner, you can stop the docker container using `docker stop` command on the host machine.

1. Monitor the running runner
   ```bash
   training_extensions$ .ci/docker/check-runner.sh --help
   USAGE: .ci/check-runner.sh <container-name> [Options]
   Options
       -r|--runner         Check runner's log instead of Job one
       -h|--help           Print this message
   ```
