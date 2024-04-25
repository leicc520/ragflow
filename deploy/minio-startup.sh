#!/bin/bash

# 设置MinIO的配置参数
# 用户名
export MINIO_ROOT_USER=admin
# 密码
export MINIO_ROOT_PASSWORD=password

#设置MinIO端口

#S3-API端口
export MINIO_ADDRESS=":6900" #可按需修改

#Console控制台页面访问端口
export MINIO_CONSOLE_ADDRESS=":6901" # 可按需修改

#设置MinIO安装路径
export MINIO_PATH_DIR="/data/minio/bin"

#设置MinIO配置文件路径
export MINIO_CONFIG_DIR="/data/minio/config"

# 设置数据存储路径
export MINIO_DATA_DIR="/data/minio/data"

# 设置日志存储路径
export MINIO_LOG_DIR="/data/minio/log"

# 启动MinIO服务器
nohup $MINIO_PATH_DIR/minio server --address $MINIO_ADDRESS --console-address $MINIO_CONSOLE_ADDRESS --config-dir $MINIO_CONFIG_DIR $MINIO_DATA_DIR > $MINIO_LOG_DIR/minio.log 2>&1 &

