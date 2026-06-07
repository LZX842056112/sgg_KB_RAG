#### 1.4 部署MinIO

在`linux`部署MinIO，安装方式采用rpm离线安装，具体步骤可参考[官方文档](https://min.io/docs/minio/linux/operations/install-deploy-manage/deploy-minio-single-node-single-drive.html#minio-snsd)。

1. **获取MinIO安装包**

   下载地址如下：https://dl.min.io/server/minio/release/linux-amd64/archive/minio-20230809233022.0.0.x86_64.rpm，通过以下命令可直接将安装包下载至服务器

   本地位置: `/资料/软件资料/minio-20230809233022.0.0.x86_64.rpm` 

   ```bash
   wget https://dl.min.io/server/minio/release/linux-amd64/archive/minio-20230809233022.0.0.x86_64.rpm
   ```

   注：若下载缓慢，大家可直接使用课程资料中附带的安装包

2. **安装MinIO**

   ```bash
   rpm -ivh minio-20230809233022.0.0.x86_64.rpm
   ```

3. **集成Systemd**

   - **Systemd概述**

     `Systemd`是一个广泛应用于Linux系统的系统初始化和服务管理器，其可以管理系统中的各种服务和进程，包括启动、停止和重启服务，除此之外，其还可以监测各服务的运行状态，并在服务异常退出时，自动拉起服务，以保证服务的稳定性。系统自带的防火墙服务`firewalld`，我们自己安装的`mysqld`和`redis`均是由`Systemd`进行管理的，此处将MinIO服务也交给Systemd管理。

   - **编写MinIO服务配置文件**

     Systemd所管理的服务需要由一个配置文件进行描述，这些配置文件均位于`/etc/systemd/system/`或者`/usr/lib/systemd/system/`目录下，下面创建MinIO服务的配置文件。

     执行以下命令创建并打开`minio.service`文件

     ```bash
     sudo vim /usr/lib/systemd/system/minio.service
     ```

     内容如下，具体可参考MinIO[官方文档](https://min.io/docs/minio/linux/operations/install-deploy-manage/deploy-minio-single-node-single-drive.html#create-the-systemd-service-file)。

     ```ini
     [Unit]
     Description=MinIO
     Documentation=https://min.io/docs/minio/linux/index.html
     Wants=network-online.target
     After=network-online.target
     AssertFileIsExecutable=/usr/local/bin/minio
     
     [Service]
     WorkingDirectory=/usr/local
     ProtectProc=invisible
     EnvironmentFile=-/etc/default/minio
     ExecStartPre=/bin/bash -c "if [ -z \"${MINIO_VOLUMES}\" ]; then echo \"Variable MINIO_VOLUMES not set in /etc/default/minio\"; exit 1; fi"
     ExecStart=/usr/local/bin/minio server $MINIO_OPTS $MINIO_VOLUMES
     Restart=always
     LimitNOFILE=65536
     TasksMax=infinity
     TimeoutStopSec=infinity
     SendSIGKILL=no
     
     [Install]
     WantedBy=multi-user.target
     ```

     **注意**：

     重点关注上述文件中的以下内容即可

     - `EnvironmentFile`，该文件中可配置MinIO服务所需的各项参数
     - `ExecStart`，该参数用于配置MinIO服务的启动命令，其中`$MINIO_OPTS`、`$MINIO_VOLUMES`，均引用于`EnvironmentFile`中的变量。
       - `MINIO_OPTS`用于配置MinIO服务的启动选项，可省略不配置。
       - `MINIO_VOLUMES`用于配置MinIO服务的数据存储路径。
     - `Restart`，表示自动重启

   - **编写`EnvironmentFile`文件**

     执行以下命令创建并打开`/etc/default/minio`文件

     ```bash
     sudo vim /etc/default/minio
     ```

     内容如下，具体可参考[官方文档](https://min.io/docs/minio/linux/operations/install-deploy-manage/deploy-minio-single-node-single-drive.html#create-the-environment-variable-file)。

     ```ini
     MINIO_ROOT_USER=minioadmin
     MINIO_ROOT_PASSWORD=minioadmin
     MINIO_VOLUMES=/data
     MINIO_OPTS="--console-address :9001"
     ```

     **注意**

     - `MINIO_ROOT_USER`和`MINIO_ROOT_PASSWORD`为用于访问MinIO的用户名和密码，**密码长度至少8位**。

     - `MINIO_VOLUMES`用于指定数据存储路径，需确保指定的路径是存在的，可执行以下命令创建该路径。

       ```bash
       mkdir /data
       chmod -R 777 /data
       ```
       
     - `MINIO_OPTS`中的`console-address`,用于指定管理页面的地址。

4. **启动MinIO**

   执行以下命令启动MinIO

   ```bash
   systemctl start minio
   ```
   执行以下命令查询运行状态
   ```bash
   systemctl status minio
   ```

   设置MinIO开机自启

   ```bash
   systemctl enable minio
   ```

5. **访问MinIO管理页面**

   管理页面的访问地址为：`http://192.168.10.101:9001`

   **注意**：

   `ip`需要根据实际情况做出修改
