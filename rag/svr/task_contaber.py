#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import time

from api.db.db_models import Document, Task
from api.db.services.file2document_service import File2DocumentService
from rag.settings import database_logger,  SVR_QUEUE_NAME_CLINICAL
from api.db.services.task_service import TaskService, queue_tasks
from api.db.services.document_service import DocumentService

def main():
    queue_name = SVR_QUEUE_NAME_CLINICAL
    kb_id = "e883ad1e2ee911efb7fa5254002a1a65"
    while True:
        docs = Document.select(Document.id).where(Document.kb_id == kb_id).\
            where(Document.chunk_num == 0).order_by(Document.create_time).limit(20000)
        if not docs or len(docs) == 0:
            time.sleep(3600*8)  # sleep for 1 hour
            continue
        for doc in docs:
            doc_id = doc.id
            tenant_id = DocumentService.get_tenant_id(doc_id)
            if not tenant_id:
                continue
            TaskService.filter_delete([Task.doc_id == doc_id])
            e, doc = DocumentService.get_by_id(doc_id)
            doc = doc.to_dict()
            doc["tenant_id"] = tenant_id
            bucket, name = File2DocumentService.get_minio_address(doc_id=doc_id)
            queue_tasks(doc, bucket, name, queue_name)
            logging.info(f"Doc {doc_id} is queued to {queue_name}")
        time.sleep(86400)

if __name__ == "__main__":
    peewee_logger = logging.getLogger('peewee')
    peewee_logger.propagate = False
    peewee_logger.addHandler(database_logger.handlers[0])
    peewee_logger.setLevel(database_logger.level)
    main()
