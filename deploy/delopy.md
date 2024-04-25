python版本要求
https://www.python.org/downloads/windows/
Requires-Python >=3.5,<3.10;

python 依赖的安装
pip install -r requirements.txt 

文件存储服务
https://min.io/docs/minio/windows/index.html
启动服务
./minio.exe server ./ --console-address :9001

向量数据库使用ES
https://www.elastic.co/cn/downloads/past-releases/elasticsearch-8-11-3

然后我们打开config文件夹中的elasticsearch.yml文件
修改其中参数：
# 关闭http访问限制
xpack.security.enabled: false


运行下载nltk
import nltk
nltk.download('punkt')
nltk.download('wordnet')

报字符加载的错误
调整对应的文档的字符集即可

linux第三方依赖服务部署：
minio文件服务
mkdir -p /data/minio/{bin,data,config,log}
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio  #添加执行权限
客服端
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc  #添加执行权限


部署elastic-search
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.12.0-linux-x86_64.tar.gz
解压
tar -xzvf elasticsearch-8.12.0-linux-x86_64.tar.gz 
useradd es-user
修改权限
chown es-user:es-user -R elasticsearch-8.12.0

修改elasticsearch.yml文件
调整数据/日志/节点配置

https://blog.csdn.net/m0_50287279/article/details/131819482