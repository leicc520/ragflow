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
import os
import re
from datetime import datetime, timedelta
from flask import request, make_response
from flask_login import login_required, current_user


from api.db import FileType, ParserType, TaskStatus, LLMType
from api.db.db_models import APIToken
from api.db.services import duplicate_name
from api.db.services.api_service import APITokenService, API4ConversationService
from api.db.services.dialog_service import DialogService, chat
from api.db.services.document_service import DocumentService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.user_service import UserTenantService
from api.settings import RetCode, retrievaler
from api.db.services.llm_service import TenantLLMService
from api.utils import get_uuid, current_timestamp, datetime_format
from api.utils.api_utils import server_error_response, get_data_error_result, get_json_result, validate_request
from itsdangerous import URLSafeTimedSerializer

from api.utils.file_utils import filename_type, thumbnail
from rag.utils import MINIO, rmSpace
from rag.nlp import search

def generate_confirmation_token(tenent_id):
    serializer = URLSafeTimedSerializer(tenent_id)
    return "ragflow-" + serializer.dumps(get_uuid(), salt=tenent_id)[2:34]


@manager.route('/new_token', methods=['POST'])
@validate_request("dialog_id")
@login_required
def new_token():
    req = request.json
    try:
        tenants = UserTenantService.query(user_id=current_user.id)
        if not tenants:
            return get_data_error_result(retmsg="Tenant not found!")

        tenant_id = tenants[0].tenant_id
        obj = {"tenant_id": tenant_id, "token": generate_confirmation_token(tenant_id),
               "dialog_id": req["dialog_id"],
               "create_time": current_timestamp(),
               "create_date": datetime_format(datetime.now()),
               "update_time": None,
               "update_date": None
               }
        if not APITokenService.save(**obj):
            return get_data_error_result(retmsg="Fail to new a dialog!")

        return get_json_result(data=obj)
    except Exception as e:
        return server_error_response(e)


@manager.route('/token_list', methods=['GET'])
@login_required
def token_list():
    try:
        tenants = UserTenantService.query(user_id=current_user.id)
        if not tenants:
            return get_data_error_result(retmsg="Tenant not found!")

        objs = APITokenService.query(tenant_id=tenants[0].tenant_id, dialog_id=request.args["dialog_id"])
        return get_json_result(data=[o.to_dict() for o in objs])
    except Exception as e:
        return server_error_response(e)


@manager.route('/rm', methods=['POST'])
@validate_request("tokens", "tenant_id")
@login_required
def rm():
    req = request.json
    try:
        for token in req["tokens"]:
            APITokenService.filter_delete(
                [APIToken.tenant_id == req["tenant_id"], APIToken.token == token])
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route('/stats', methods=['GET'])
@login_required
def stats():
    try:
        tenants = UserTenantService.query(user_id=current_user.id)
        if not tenants:
            return get_data_error_result(retmsg="Tenant not found!")
        objs = API4ConversationService.stats(
            tenants[0].tenant_id,
            request.args.get(
                "from_date",
                (datetime.now() -
                 timedelta(
                    days=7)).strftime("%Y-%m-%d 24:00:00")),
            request.args.get(
                "to_date",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        res = {
            "pv": [(o["dt"], o["pv"]) for o in objs],
            "uv": [(o["dt"], o["uv"]) for o in objs],
            "speed": [(o["dt"], float(o["tokens"])/(float(o["duration"]+0.1))) for o in objs],
            "tokens": [(o["dt"], float(o["tokens"])/1000.) for o in objs],
            "round": [(o["dt"], o["round"]) for o in objs],
            "thumb_up": [(o["dt"], o["thumb_up"]) for o in objs]
        }
        return get_json_result(data=res)
    except Exception as e:
        return server_error_response(e)


@manager.route('/new_conversation', methods=['GET'])
def set_conversation():
    token = request.headers.get('Authorization').split()[1]
    objs = APIToken.query(token=token)
    if not objs:
        return get_json_result(
            data=False, retmsg='Token is not valid!"', retcode=RetCode.AUTHENTICATION_ERROR)
    req = request.json
    try:
        e, dia = DialogService.get_by_id(objs[0].dialog_id)
        if not e:
            return get_data_error_result(retmsg="Dialog not found")
        conv = {
            "id": get_uuid(),
            "dialog_id": dia.id,
            "user_id": request.args.get("user_id", ""),
            "message": [{"role": "assistant", "content": dia.prompt_config["prologue"]}]
        }
        API4ConversationService.save(**conv)
        e, conv = API4ConversationService.get_by_id(conv["id"])
        if not e:
            return get_data_error_result(retmsg="Fail to new a conversation!")
        conv = conv.to_dict()
        return get_json_result(data=conv)
    except Exception as e:
        return server_error_response(e)


@manager.route('/completion', methods=['POST'])
@validate_request("conversation_id", "messages")
def completion():
    token = request.headers.get('Authorization').split()[1]
    if not APIToken.query(token=token):
        return get_json_result(
            data=False, retmsg='Token is not valid!"', retcode=RetCode.AUTHENTICATION_ERROR)
    req = request.json
    e, conv = API4ConversationService.get_by_id(req["conversation_id"])
    if not e:
        return get_data_error_result(retmsg="Conversation not found!")

    msg = []
    for m in req["messages"]:
        if m["role"] == "system":
            continue
        if m["role"] == "assistant" and not msg:
            continue
        msg.append({"role": m["role"], "content": m["content"]})

    try:
        conv.message.append(msg[-1])
        e, dia = DialogService.get_by_id(conv.dialog_id)
        if not e:
            return get_data_error_result(retmsg="Dialog not found!")
        del req["conversation_id"]
        del req["messages"]
        ans = chat(dia, msg, **req)
        if not conv.reference:
            conv.reference = []
        conv.reference.append(ans["reference"])
        conv.message.append({"role": "assistant", "content": ans["answer"]})
        API4ConversationService.append_message(conv.id, conv.to_dict())
        return get_json_result(data=ans)
    except Exception as e:
        return server_error_response(e)


@manager.route('/conversation/<conversation_id>', methods=['GET'])
# @login_required
def get_conversation(conversation_id):
    try:
        e, conv = API4ConversationService.get_by_id(conversation_id)
        if not e:
            return get_data_error_result(retmsg="Conversation not found!")

        return get_json_result(data=conv.to_dict())
    except Exception as e:
        return server_error_response(e)

@manager.route('/document/<doc_id>', methods=['GET'])
def get_document(doc_id):
    try:
        e, doc = DocumentService.get_by_id(doc_id)
        if not e:
            return get_data_error_result(retmsg="Conversation not found!")

        return get_json_result(data=doc.to_dict())
    except Exception as e:
        return server_error_response(e)

@manager.route('/document/image/<image_id>', methods=['GET'])
def get_document_image(image_id):
    try:
        bkt, nm = image_id.split("-")
        response = make_response(MINIO.get(bkt, nm))
        response.headers.set('Content-Type', 'image/JPEG')
        return response
    except Exception as e:
        return server_error_response(e)

@manager.route('/document/upload', methods=['POST'])
@validate_request("kb_name")
def upload():
    token = request.headers.get('Authorization').split()[1]
    objs = APIToken.query(token=token)
    if not objs:
        return get_json_result(
            data=False, retmsg='Token is not valid!"', retcode=RetCode.AUTHENTICATION_ERROR)

    kb_name = request.form.get("kb_name").strip()
    use_type = request.form.get("use_type", "document")
    tenant_id = objs[0].tenant_id

    try:
        e, kb = KnowledgebaseService.get_by_name(kb_name, tenant_id)
        if not e:
            return get_data_error_result(
                retmsg="Can't find this knowledgebase!")
        kb_id = kb.id
    except Exception as e:
        return server_error_response(e)

    if 'file' not in request.files:
        return get_json_result(
            data=False, retmsg='No file part!', retcode=RetCode.ARGUMENT_ERROR)

    file = request.files['file']
    if file.filename == '':
        return get_json_result(
            data=False, retmsg='No file selected!', retcode=RetCode.ARGUMENT_ERROR)
    try:
        '''
        不需要限制用户上传文件数量
        if DocumentService.get_doc_count(kb.tenant_id) >= int(os.environ.get('MAX_FILE_NUM_PER_USER', 8192)):
            return get_data_error_result(
                retmsg="Exceed the maximum file number of a free user!")
        '''
        filename = duplicate_name(
            DocumentService.query,
            name=file.filename,
            kb_id=kb_id)
        filetype = filename_type(filename)
        if not filetype:
            return get_data_error_result(
                retmsg="This type of file has not been supported yet!")

        location = filename
        while MINIO.obj_exist(kb_id, location):
            location += "_"
        blob = request.files['file'].read()
        MINIO.put(kb_id, location, blob)
        doc = {
            "id": get_uuid(),
            "kb_id": kb.id,
            "parser_id": kb.parser_id,
            "parser_config": kb.parser_config,
            "created_by": kb.tenant_id,
            "type": filetype,
            "use_type": use_type,
            "name": filename,
            "location": location,
            "size": len(blob),
            "thumbnail": thumbnail(filename, blob)
        }
        #设置上传之后立刻解析
        if request.form.get("run"):
            doc["run"] = TaskStatus.RUNNING.value
            doc["progress"] = 0
            doc["progress_msg"] = ""
            doc["chunk_num"] = 0
            doc["token_num"] = 0
        if doc["type"] == FileType.VISUAL:
            doc["parser_id"] = ParserType.PICTURE.value
        if re.search(r"\.(ppt|pptx|pages)$", filename):
            doc["parser_id"] = ParserType.PRESENTATION.value
        doc = DocumentService.insert(doc)
        return get_json_result(data=doc.to_json())
    except Exception as e:
        return server_error_response(e)

@manager.route('/kb/retrieval', methods=['POST'])
@validate_request("kb_id", "question")
def retrieval():
    req = request.json
    page = int(req.get("page", 1))
    size = int(req.get("size", 1024))
    question = req["question"]
    kb_id = req["kb_id"]
    doc_ids = req.get("doc_ids", [])
    similarity_threshold = float(req.get("similarity_threshold", 0.2))
    vector_similarity_weight = float(req.get("vector_similarity_weight", 0.3))
    top = int(req.get("top_k", 1024))
    try:
        e, kb = KnowledgebaseService.get_by_id(kb_id)
        if not e:
            return get_data_error_result(retmsg="Knowledgebase not found!")

        embd_mdl = TenantLLMService.model_instance(
            kb.tenant_id, LLMType.EMBEDDING.value, llm_name=kb.embd_id)
        ranks = retrievaler.retrieval(question, embd_mdl, kb.tenant_id, [kb_id], page, size, similarity_threshold,
                                      vector_similarity_weight, top, doc_ids, aggs=False)
        for c in ranks["chunks"]:
            if "vector" in c:
                del c["vector"]

        return get_json_result(data=ranks)
    except Exception as e:
        if str(e).find("not_found") > 0:
            return get_json_result(data=False, retmsg=f'No chunk found! Check the chunk status please!',
                                   retcode=RetCode.DATA_ERROR)
        return server_error_response(e)

@manager.route('/chunk/list', methods=['POST'])
@validate_request("doc_id")
def chunk_list():
    req = request.json
    doc_id = req["doc_id"]
    page = int(req.get("page", 1))
    size = int(req.get("size", 30))
    question = req.get("keywords", "")
    try:
        tenant_id = DocumentService.get_tenant_id(req["doc_id"])
        if not tenant_id:
            return get_data_error_result(retmsg="Tenant not found!")
        e, doc = DocumentService.get_by_id(doc_id)
        if not e:
            return get_data_error_result(retmsg="Document not found!")
        query = {
            "doc_ids": [doc_id], "page": page, "size": size, "question": question, "sort": True
        }
        if "available_int" in req:
            query["available_int"] = int(req["available_int"])
        sres = retrievaler.search(query, search.index_name(tenant_id))
        res = {"total": sres.total, "chunks": [], "doc": doc.to_dict()}
        for id in sres.ids:
            d = {
                "chunk_id": id,
                "content_with_weight": rmSpace(sres.highlight[id]) if question and id in sres.highlight else sres.field[id].get(
                    "content_with_weight", ""),
                "doc_id": sres.field[id]["doc_id"],
                "docnm_kwd": sres.field[id]["docnm_kwd"],
                "important_kwd": sres.field[id].get("important_kwd", []),
                "img_id": sres.field[id].get("img_id", ""),
                "available_int": sres.field[id].get("available_int", 1),
                "positions": sres.field[id].get("position_int", "").split("\t")
            }
            if len(d["positions"]) % 5 == 0:
                poss = []
                for i in range(0, len(d["positions"]), 5):
                    poss.append([float(d["positions"][i]), float(d["positions"][i + 1]), float(d["positions"][i + 2]),
                                 float(d["positions"][i + 3]), float(d["positions"][i + 4])])
                d["positions"] = poss
            res["chunks"].append(d)
        return get_json_result(data=res)
    except Exception as e:
        if str(e).find("not_found") > 0:
            return get_json_result(data=False, retmsg=f'No chunk found!',
                                   retcode=RetCode.DATA_ERROR)
        return server_error_response(e)