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


第三方模型的下载逻辑
https://hf-mirror.com/models

源码编译python ssl模块的依赖
https://blog.csdn.net/younger_china/article/details/131956824
python项目安装依赖
pip install -r requirements.txt 

# 调整c/c++环境变量
export C_INCLUDE_PATH=/usr/include/python:/opt/Python-3.9.16
export C_INCLUDE_PATH=XXXX:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=XXX:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=XXX:$LD_LIBRARY_PATH
export LIBRARY_PATH=XXX:$LIBRARY_PATH

依赖包报错的逻辑
yum install openmpi-devel
export CC=/usr/lib64/openmpi/bin/mpicc
pip install mpi4py

安装依赖的搜索路径
export PYTHONPATH=/usr/local/lib64/python3.9/site-packages:/usr/local/lib64/python3.9/site-packages:.


pdf文件解析报错
During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/data/ragflow/ragflow/rag/svr/task_broker.py", line 99, in dispatch
    pages = PdfParser.total_page_number(r["name"], file_bin)
  File "/data/ragflow/ragflow/deepdoc/parser/pdf_parser.py", line 924, in total_page_number
    pdf = fitz.open(fnm) if not binary else fitz.open(
  File "/usr/local/lib64/python3.9/site-packages/fitz/__init__.py", line 2630, in __init__
    raise TypeError("bad type: 'stream'")
TypeError: bad type: 'stream'
bad type: 'stream'
Traceback (most recent call last):
  File "/data/ragflow/ragflow/deepdoc/parser/pdf_parser.py", line 921, in total_page_number
    fnm) if not binary else pdfplumber.open(BytesIO(binary))
  File "/usr/local/lib/python3.9/site-packages/pdfplumber/pdf.py", line 95, in open
    return cls(
  File "/usr/local/lib/python3.9/site-packages/pdfplumber/pdf.py", line 45, in __init__
    self.doc = PDFDocument(PDFParser(stream), password=password or "")
  File "/usr/local/lib/python3.9/site-packages/pdfminer/pdfdocument.py", line 752, in __init__
    raise PDFSyntaxError("No /Root object! - Is this really a PDF?")
pdfminer.pdfparser.PDFSyntaxError: No /Root object! - Is this really a PDF?

日志安装目录的logs目录
