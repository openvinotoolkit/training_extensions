
# GPU Version

## GPU Requirements

## Install Nvidia-driver

### Uninstall Old driver

```sh
sudo apt purge nvidia-* libnvidia-*
```

### Install New

```sh
sudo apt install nvidia-driver-440
```

REBOOT REQUIRED

## Install CUDA 10.2

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda-10-2
```

## Install Docker

```sh
$ sudo apt update
$ sudo apt install apt-transport-https ca-certificates curl software-properties-common
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
$ sudo apt update
$ sudo apt install docker-ce
```

### Test Installation

```sh
$ sudo systemctl status docker
● docker.service - Docker Application Container Engine
   Loaded: loaded (/lib/systemd/system/docker.service; enabled; vendor preset: enabled)
  Drop-In: /etc/systemd/system/docker.service.d
           └─https-proxy.conf, intel.conf, override.conf
   Active: active (running) since Tue 2020-07-21 13:52:39 MSK; 35min ago
     Docs: https://docs.docker.com
 Main PID: 2003 (dockerd)
    Tasks: 104
   CGroup: /system.slice/docker.service
           ├─2003 /usr/bin/dockerd --host=fd:// --add-runtime=nvidia=/usr/bin/nvidia-container-runtime-hook
           ├─3533 /usr/bin/docker-proxy -proto tcp -host-ip 0.0.0.0 -host-port 8080 -container-ip 172.21.0.4 -container-port 80
           └─4023 /usr/bin/docker-proxy -proto tcp -host-ip 0.0.0.0 -host-port 8888 -container-ip 172.20.0.6 -container-port 8888
```

### Use docker as root

```sh
$ sudo usermod -aG docker ${USER}
```

### Config Proxy

Create or edit the file `~/.docker/config.json` in the home directory of the user which starts containers.

```sh
{
 "proxies":
 {
   "default":
   {
     "httpProxy": "http://127.0.0.1:3001",
     "httpsProxy": "http://127.0.0.1:3001",
     "noProxy": "*.test.example.com,.example2.com"
   }
 }
}
```

RE-LOGIN REQUIRED after this command

## Install Docker-Compose

```sh
$ sudo curl -L "https://github.com/docker/compose/releases/download/1.26.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
$ sudo chmod +x /usr/local/bin/docker-compose
```

### Test Installation

```sh
$ docker-compose --version
docker-compose version 1.26.2, build 1110ad01
```

## Install nvidia-docker

### Add the package repositories

```sh
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker
```

### Install nvidia-runtime

```sh
$ sudo apt-get install nvidia-container-runtime
```

### Add Daemon configuration file

```bash
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
```

## GPU Setup

```sh
$ git clone --recursive https://github.com/dmitryagapov/openvino_training_extensions.git
$ cd openvino_training_extensions
$ git submodule update --init --recursive
$ IDLP_HOST=<host.name> docker-compose -f docker-compose.gpu.yml up --build -d
```

### Create CVAT root user

```sh
$ docker exec -it cvat bash -ic 'python3 ~/manage.py createsuperuser'
```

Username: django

Password: django

## Load UI

`http://localhost:8001`


## RESTART LOCAL

```sh
$ git pull
$ git git submodule update --recursive
$ IDLP_HOST=<host.name> docker-compose -f docker-compose.gpu.yml up --build -d
```
