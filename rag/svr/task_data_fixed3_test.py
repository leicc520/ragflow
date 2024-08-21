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

from api.db.db_models import Document
from api.db.services.file2document_service import File2DocumentService
from rag.utils.minio_conn import MINIO


'''检测数据丢失的记录清空'''
def main():
    docs = Document.select(Document.id).where(Document.kb_id == 'e883ad1e2ee911efb7fa5254002a1a65').\
        where(Document.version != 'v1.999999')
    docs = list(docs.dicts())
    if not docs:
        return
    for doc in docs:
        print(doc.get('id'))
        b, n = File2DocumentService.get_minio_address(doc_id=doc.get('id'))
        if not MINIO.obj_exists(b, n):
            print(f"{n}/{doc.get('id')}")

if __name__ == "__main__":
    main()

