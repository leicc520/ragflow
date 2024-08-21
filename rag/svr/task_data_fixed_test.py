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



def clean_doc(doc_id):
    try:
        e, doc = DocumentService.get_by_id(doc_id)
        if not e:
            raise Exception("Document not found!")
        tenant_id = DocumentService.get_tenant_id(doc_id)
        if not tenant_id:
            raise Exception("Tenant not found!")

        b, n = File2DocumentService.get_minio_address(doc_id=doc_id)

        if not DocumentService.remove_document(doc, tenant_id):
            raise Exception("Database error (Document removal)!")

        f2d = File2DocumentService.get_by_document_id(doc_id)
        FileService.filter_delete([File.source_type == FileSource.KNOWLEDGEBASE, File.id == f2d[0].file_id])
        File2DocumentService.delete_by_document_id(doc_id)
        MINIO.rm(b, n)
        ELASTICSEARCH.deleteByQuery(
            Q("match", doc_id=doc_id), idxnm=search.index_name(tenant_id))
    except Exception as e:
        print(e)
        raise e

def main():
    docs = Document.select().where(Document.kb_id == 'e883ad1e2ee911efb7fa5254002a1a65').\
        where(Document.location.contains('(') | Document.location.contains('_'))
    docs = list(docs.dicts())
    if not docs:
        return
    for doc in docs:
        print(doc.get('id'))
        clean_doc(doc.get('id'))

if __name__ == "__main__":
    main()

