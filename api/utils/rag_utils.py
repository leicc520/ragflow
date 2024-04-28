import requests
import json
import os
from api.utils.file_utils import get_project_base_directory
from api.utils import get_base_config
from api.utils.log_utils import LoggerFactory, getLogger

RAG_CONFIG = get_base_config("rag_notify", {})

'''
    定义RAG服务通知的业务代码
'''
cmdDocumentParseFinished = 101

LoggerFactory.set_directory(
    os.path.join(
        get_project_base_directory(),
        "logs",
        "rag"))
# {CRITICAL: 50, FATAL:50, ERROR:40, WARNING:30, WARN:30, INFO:20, DEBUG:10, NOTSET:0}
LoggerFactory.LEVEL = 20

app_logger = getLogger("app")


def call_rag_notify(doc):
    """
    发送RAG解析完毕的数据通知业务逻辑
    """
    result = {"kb_id": doc["kb_id"], "doc_id":doc["doc_id"], "name":doc["name"]}
    return call_app_rag_api(cmdDocumentParseFinished, result)
def call_app_rag_api(cmd, result):
    link = RAG_CONFIG.get("url")
    if not link:
        return

    payload = {'cmd': cmd, 'api_key': RAG_CONFIG.get("api_key"), 'result': json.dumps(result)}
    headers = {'content-type': 'application/json'}
    sp = requests.post(link, data=json.dumps(payload), headers=headers)
    app_logger.info("param:{}, result:{}".format(result, sp.text))
    if sp.status_code == 200:
        return sp.json()
    else:
        return None


if __name__ == '__main__':
    result = call_rag_notify({"kb_id": "123", "doc_id": "456", "name": "test"})
    print(result)