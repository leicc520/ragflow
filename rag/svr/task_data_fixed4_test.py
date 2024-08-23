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

from api.db.db_models import Task, File
from api.db.services.file_service import FileService
from api.db.db_models import Document, Knowledgebase
from api.db.services.file2document_service import File2DocumentService
from rag.utils.minio_conn import MINIO
from elasticsearch_dsl import Q, Search
from rag.utils.es_conn import ELASTICSEARCH
from rag.nlp import search, rag_tokenizer
from api.db import FileSource
from api.db.services.document_service import DocumentService




'''清理历史脏数据'''
def main():
    doc_id = "912d8cf25be811efa9b1525400a5affd"
    b, n = File2DocumentService.get_minio_address(doc_id=doc_id)
    print(b,n)

    result =MINIO.get(b, n)
    print(result)
if __name__ == "__main__":
    main()

