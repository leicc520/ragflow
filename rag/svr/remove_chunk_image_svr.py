import random
import time
import traceback

from rag.utils.redis_conn import REDIS_CONN
from api.settings import RetCode, retrievaler
from api.db.db_models import close_connection
from api.db.db_models import DB, Knowledgebase, Document, Task
from api.db.services.document_service import DocumentService
from rag.settings import cron_logger
from rag.utils.minio_conn import MINIO

def main():
    kb_id = "e883ad1e2ee911efb7fa5254002a1a65"
    tenant_id = "be0609562e4e11ef936c525400c442a4"
    MINIO.rm(kb_id, "0089b00d40eaa9622fe3aca2157b2bc1")
    return
    docs = DocumentService().get_list_by_kb_id(kb_id, 0, 50000)
    print(docs)
    for doc in docs:
        print(doc)
        res = retrievaler.chunk_list(doc["id"], tenant_id)
        if res is None:
            continue
        for loc in res:
            MINIO.rm(kb_id, loc["chunk_id"])

if __name__ == "__main__":

    DocumentService.update_progress()