**Ubuntu部署Docker**

部署dify平台，需要基于docker环境，而腾讯云新建的云平台上默认是没有docker的。接着，需要在腾讯云租用的服务器中部署Docker。

**什么是Docker？**

![1761532679426](assets/1761532679426.png)

Docker是一种容器化技术，相较于传统的通过虚拟机技术实现的虚拟化方案来说，Docker是⼀种更加轻量级的虚拟化解决方案。

**它可以将应用程序及其依赖项打包成一个独立的容器，并在不同的环境中运行。**通过Docker容器， 开发者可以轻松地构建、部署和运行应用程序，而无需担心环境配置和依赖问题。

按照下面的指令一步一步进行操作

```bash
#更新软件包
sudo apt update

sudo apt upgrade

#安装docker依赖
sudo apt install software-properties-common

sudo apt-get install ca-certificates curl gnupg lsb-releasesudo 
sudo apt-get install ca-certificates curl gnupg lsb-release

#添加Docker官方GPG密钥
curl -fsSL http://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -

#添加Docker软件源（输入后根据提示按Enter）
sudo add-apt-repository "deb [arch=amd64] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"

#安装docker（输入后根据提示输入 y ）
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

执行`sudo apt upgrade`的时候会出现这个界面，按回车即可

![1761532742377](assets/1761532742377.png)

之后如果在这个界面卡住，按几下回车即可

![1761532771485](assets/1761532771485.png)

安装完毕，启动docker，并查看状态

```bash
sudo systemctl start docker

sudo systemctl status docker
```

如图所示即为启动成功

![1761532791737](assets/1761532791737.png)

> 看到running状态说明docker已经正常启动

进行镜像源的配置

```bash
sudo vi /etc/docker/daemon.json
#执行sudo docker compose up -d，Docker 会自动帮你：拉取需要的镜像 → 创建容器 → 按顺序启动所有服务 → 后台运行；
#后续想停止服务，用sudo docker compose down就行，一键关停所有相关容器，干净不残留。
```

添加下面的配置

```bash
{
"registry-mirrors": [
"https://docker.unsee.tech",
"https://dockerpull.org",
"https://docker.1panel.live",
"https://dockerhub.icu",
"https://docker.m.daocloud.io",
"https://docker.nju.edu.cn",
"https://registry.docker-cn.com",
"https://docker.mirrors.ustc.edu.cn",
"https://hub-mirror.c.163.com",
"https://mirror.baidubce.com",
"https://5tqw56kt.mirror.aliyuncs.com",
"https://docker.hpcloud.cloud",
"http://mirrors.ustc.edu.cn",
"https://docker.chenby.cn",
"https://docker.ckyl.me",
"http://mirror.azure.cn",
"https://hub.rat.dev"
	]
}
```

保存，然后在终端重新启动一下docker

```bash
# 重新登陆，需要输入密码
systemctl daemon-reload

systemctl restart docker
```

重新执行

```bash
sudo docker compose up -d
```

**CentOS 7 安装说明**

```bash
# 1. 查看系统版本
cat /etc/centos-release

# 2. 更新软件包
sudo yum update -y

# 3. 卸载旧版本 Docker（如果没有会自动跳过）
sudo yum remove -y docker \
                  docker-client \
                  docker-client-latest \
                  docker-common \
                  docker-latest \
                  docker-latest-logrotate \
                  docker-logrotate \
                  docker-engine

# 4. 安装依赖包
sudo yum install -y yum-utils device-mapper-persistent-data lvm2

# 5. 添加 Docker 官方仓库 
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/
docker-ce.repo

# 6. 安装 Docker
sudo yum install -y docker-ce docker-ce-cli containerd.io

# 7. 启动 Docker 服务
sudo systemctl start docker

# 8. 设置开机自启
sudo systemctl enable docker

# 9. 查看 Docker 版本
docker --version

# 10. 运行测试容器
sudo docker run hello-world
```
文档说明:

- 执行 sudo yum update -y 时，如果更新内容较多，等待执行完成即可
- 执行 sudo yum install -y docker-ce docker-ce-cli containerd.io 时，如果提示导入 GPG key，输入 y 确认即可
- 执行 sudo systemctl start docker 后，可通过 sudo systemctl status docker 查看 Docker 是否启动成功