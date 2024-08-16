import random
import time
import datetime
import traceback

from rag.utils.redis_conn import REDIS_CONN
from api.settings import RetCode, retrievaler
from api.db.db_models import close_connection
from api.db.db_models import DB, Knowledgebase, Document, Task
from api.db.services.document_service import DocumentService
from rag.settings import cron_logger
from rag.utils.minio_conn import MINIO
from rag.settings import ParserVersionEnum

def main():
    kb_id = "e883ad1e2ee911efb7fa5254002a1a65"
    tenant_id = "be0609562e4e11ef936c525400c442a4"
    docs = DocumentService().get_list_by_kb_id(kb_id, 0, 50000)
    for doc in docs:
        res = retrievaler.chunk_list(doc["id"], tenant_id)
        if res is None:
            continue
        for loc in res:
            MINIO.rm(kb_id, loc["chunk_id"])

if __name__ == "__main__":
    doc_id = "cc8d0c60595811ef9efe525400a5affd"
    _, doc = DocumentService.get_by_id(doc_id)
    if doc is None or (doc.chunk_num > 0 and doc.progress >= 0.999):
        print("doc is None or already processed")
    print(doc)
    # wait for 5 minutes
    n = datetime.datetime.now().hour
    if n >= 8 and n <= 21:
        time.sleep(300)
    print(n)
    main()