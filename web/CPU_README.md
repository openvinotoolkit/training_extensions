  
# CPU Version

## CPU Requirements

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
RE-LOGIN REQUIRED after this command

### Test Installation

```sh
$ docker-compose --version
docker-compose version 1.26.2, build 1110ad01
```


## CPU Setup

```sh
$ git clone --recursive https://github.com/dmitryagapov/openvino_training_extensions.git
$ cd openvino_training_extensions
$ git submodule update --init --recursive
$ IDLP_HOST=localhost docker-compose -f docker-compose.cpu.yml up --build -d
```
### Create CVAT root user

```sh
$ docker exec -it cvat bash -ic 'python3 ~/manage.py createsuperuser'
```
Username: django

Password: django

## Open In Browser

`http://localhost:8001`

## CPU Restart

```sh
$ git pull
$ git git submodule update --recursive
$ IDLP_HOST=<host.name> docker-compose -f docker-compose.cpu.yml up --build -d
```
