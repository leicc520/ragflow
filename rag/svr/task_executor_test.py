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
import datetime
import json
import logging
import os
import hashlib
import copy
import re
import sys
import time
import traceback
from functools import partial

from api.db.db_models import Document, Knowledgebase
from api.db.services.file2document_service import File2DocumentService
from api.settings import retrievaler
from rag.raptor import RecursiveAbstractiveProcessing4TreeOrganizedRetrieval as Raptor
from rag.utils.minio_conn import MINIO
from api.db.db_models import close_connection
from rag.settings import database_logger, SVR_QUEUE_NAME, SVR_QUEUE_NAME_CRAWLER, SVR_QUEUE_NAME_CLINICAL, SVR_CONSUMER_NAME, SVR_CONSUMER_GROUP_NAME
from rag.settings import cron_logger, DOC_MAXIMUM_SIZE
from multiprocessing import Pool
import numpy as np
from elasticsearch_dsl import Q, Search
from multiprocessing.context import TimeoutError
from api.db.services.task_service import TaskService
from rag.utils.es_conn import ELASTICSEARCH
from timeit import default_timer as timer
from rag.utils import rmSpace, findMaxTm, num_tokens_from_string

from rag.nlp import search, rag_tokenizer
from io import BytesIO
import pandas as pd

from rag.app import laws, paper, presentation, manual, qa, table, book, resume, picture, naive, one

from api.utils.rag_utils import call_rag_notify
from api.db import LLMType, ParserType
from api.db.services.document_service import DocumentService
from api.db.services.llm_service import LLMBundle
from api.utils.file_utils import get_project_base_directory
from rag.utils.redis_conn import REDIS_CONN

BATCH_SIZE = 64

FACTORY = {
    "general": naive,
    ParserType.NAIVE.value: naive,
    ParserType.PAPER.value: paper,
    ParserType.BOOK.value: book,
    ParserType.PRESENTATION.value: presentation,
    ParserType.MANUAL.value: manual,
    ParserType.LAWS.value: laws,
    ParserType.QA.value: qa,
    ParserType.TABLE.value: table,
    ParserType.RESUME.value: resume,
    ParserType.PICTURE.value: picture,
    ParserType.ONE.value: one,
}


def set_progress(task_id, from_page=0, to_page=-1,
                 prog=None, msg="Processing..."):
    if prog is not None and prog < 0:
        msg = "[ERROR]" + msg
    cancel = TaskService.do_cancel(task_id)
    if cancel:
        msg += " [Canceled]"
        prog = -1

    if to_page > 0:
        if msg:
            msg = f"Page({from_page + 1}~{to_page + 1}): " + msg
    d = {"progress_msg": msg}
    if prog is not None:
        d["progress"] = prog
    try:
        TaskService.update_progress(task_id, d)
    except Exception as e:
        cron_logger.error("set_progress:({}), {}".format(task_id, str(e)))

    close_connection()
    if cancel:
        sys.exit()


def collect(queue_name):
    try:
        payload = REDIS_CONN.queue_consumer(queue_name, SVR_CONSUMER_GROUP_NAME, SVR_CONSUMER_NAME)
        if not payload:
            time.sleep(1)
            return pd.DataFrame()
    except Exception as e:
        cron_logger.error("Get task event from queue exception:" + str(e))
        return pd.DataFrame()

    msg = payload.get_message()
    payload.ack()
    if not msg: return pd.DataFrame()

    if TaskService.do_cancel(msg["id"]):
        cron_logger.info("Task {} has been canceled.".format(msg["id"]))
        return pd.DataFrame()
    tasks = TaskService.get_tasks(msg["id"])
    assert tasks, "{} empty task!".format(msg["id"])
    tasks = pd.DataFrame(tasks)
    if msg.get("type", "") == "raptor":
        tasks["task_type"] = "raptor"
    return tasks

def get_minio_binary(bucket, name):
    return MINIO.get(bucket, name)


def build(row):
    if row["size"] > DOC_MAXIMUM_SIZE:
        set_progress(row["id"], prog=-1, msg="File size exceeds( <= %dMb )" %
                                             (int(DOC_MAXIMUM_SIZE / 1024 / 1024)))
        return []

    callback = partial(
        set_progress,
        row["id"],
        row["from_page"],
        row["to_page"])
    chunker = FACTORY[row["parser_id"].lower()]
    try:
        st = timer()
        bucket, name = File2DocumentService.get_minio_address(doc_id=row["doc_id"])
        binary = get_minio_binary(bucket, name)
        cron_logger.info(
            "From minio({}) {}/{}".format(timer() - st, row["location"], row["name"]))
        cks = chunker.chunk(row["name"], binary=binary, from_page=row["from_page"],
                            to_page=row["to_page"], lang=row["language"], callback=callback,
                            kb_id=row["kb_id"], parser_config=row["parser_config"], tenant_id=row["tenant_id"])
        cron_logger.info(
            "Chunkking({}) {}/{}".format(timer() - st, row["location"], row["name"]))
    except TimeoutError as e:
        callback(-1, f"Internal server error: Fetch file timeout. Could you try it again.")
        cron_logger.error(
            "Chunkking {}/{}: Fetch file timeout.".format(row["location"], row["name"]))
        return
    except Exception as e:
        if re.search("(No such file|not found)", str(e)):
            callback(-1, "Can not find file <%s>" % row["name"])
        else:
            callback(-1, f"Internal server error: %s" %
                     str(e).replace("'", ""))
        traceback.print_exc()

        cron_logger.error(
            "Chunkking {}/{}: {}".format(row["location"], row["name"], str(e)))

        return

    docs = []
    doc = {
        "doc_id": row["doc_id"],
        "kb_id": [str(row["kb_id"])]
    }
    el = 0
    for ck in cks:
        d = copy.deepcopy(doc)
        d.update(ck)
        md5 = hashlib.md5()
        md5.update((ck["content_with_weight"] +
                    str(d["doc_id"])).encode("utf-8"))
        d["_id"] = md5.hexdigest()
        d["create_time"] = str(datetime.datetime.now()).replace("T", " ")[:19]
        d["create_timestamp_flt"] = datetime.datetime.now().timestamp()
        if not d.get("image"):
            docs.append(d)
            continue

        output_buffer = BytesIO()
        if isinstance(d["image"], bytes):
            output_buffer = BytesIO(d["image"])
        else:
            d["image"].save(output_buffer, format='JPEG')

        st = timer()
        #MINIO.put(row["kb_id"], d["_id"], output_buffer.getvalue())
        #d["img_id"] = "{}-{}".format(row["kb_id"], d["_id"])
        el += timer() - st
        del d["image"]
        docs.append(d)
    cron_logger.info("MINIO PUT({}):{}".format(row["name"], el))

    return docs


def init_kb(row):
    idxnm = search.index_name(row["tenant_id"])
    if ELASTICSEARCH.indexExist(idxnm):
        return
    return ELASTICSEARCH.createIdx(idxnm, json.load(
        open(os.path.join(get_project_base_directory(), "conf", "mapping.json"), "r")))


def embedding(docs, mdl, parser_config={}, callback=None):
    batch_size = 32
    tts, cnts = [rmSpace(d["title_tks"]) for d in docs if d.get("title_tks")], [
        re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", d["content_with_weight"]) for d in docs]
    tk_count = 0
    if len(tts) == len(cnts):
        tts_ = np.array([])
        for i in range(0, len(tts), batch_size):
            vts, c = mdl.encode(tts[i: i + batch_size])
            if len(tts_) == 0:
                tts_ = vts
            else:
                tts_ = np.concatenate((tts_, vts), axis=0)
            tk_count += c
            callback(prog=0.6 + 0.1 * (i + 1) / len(tts), msg="")
        tts = tts_

    cnts_ = np.array([])
    for i in range(0, len(cnts), batch_size):
        vts, c = mdl.encode(cnts[i: i + batch_size])
        if len(cnts_) == 0:
            cnts_ = vts
        else:
            cnts_ = np.concatenate((cnts_, vts), axis=0)
        tk_count += c
        callback(prog=0.7 + 0.2 * (i + 1) / len(cnts), msg="")
    cnts = cnts_

    title_w = float(parser_config.get("filename_embd_weight", 0.1))
    vects = (title_w * tts + (1 - title_w) *
             cnts) if len(tts) == len(cnts) else cnts

    assert len(vects) == len(docs)
    for i, d in enumerate(docs):
        v = vects[i].tolist()
        d["q_%d_vec" % len(v)] = v
    return tk_count


def run_raptor(row, chat_mdl, embd_mdl, callback=None):
    vts, _ = embd_mdl.encode(["ok"])
    vctr_nm = "q_%d_vec"%len(vts[0])
    chunks = []
    for d in retrievaler.chunk_list(row["doc_id"], row["tenant_id"], fields=["content_with_weight", vctr_nm]):
        chunks.append((d["content_with_weight"], np.array(d[vctr_nm])))

    raptor = Raptor(
        row["parser_config"]["raptor"].get("max_cluster", 64),
        chat_mdl,
        embd_mdl,
        row["parser_config"]["raptor"]["prompt"],
        row["parser_config"]["raptor"]["max_token"],
        row["parser_config"]["raptor"]["threshold"]
    )
    original_length = len(chunks)
    raptor(chunks, row["parser_config"]["raptor"]["random_seed"], callback)
    doc = {
        "doc_id": row["doc_id"],
        "kb_id": [str(row["kb_id"])],
        "docnm_kwd": row["name"],
        "title_tks": rag_tokenizer.tokenize(row["name"])
    }
    res = []
    tk_count = 0
    for content, vctr in chunks[original_length:]:
        d = copy.deepcopy(doc)
        md5 = hashlib.md5()
        md5.update((content + str(d["doc_id"])).encode("utf-8"))
        d["_id"] = md5.hexdigest()
        d["create_time"] = str(datetime.datetime.now()).replace("T", " ")[:19]
        d["create_timestamp_flt"] = datetime.datetime.now().timestamp()
        d[vctr_nm] = vctr.tolist()
        d["content_with_weight"] = content
        d["content_ltks"] = rag_tokenizer.tokenize(content)
        d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
        res.append(d)
        tk_count += num_tokens_from_string(content)
    return res, tk_count


def main():
    kb = Knowledgebase.select().where(Knowledgebase.id == "700bf9f62e5611ef86e6525400c442a4").get()
    doc = Document.select().where(Document.id == "962b30865dd411efb92a525400a5affd").get()
    r = doc.to_dict()
    rb_dict = kb.to_dict()
    r["from_page"] = 0
    r["to_page"] = 15
    r["doc_id"] = r["id"]
    del r['parser_config']
    r = {**rb_dict, **r}

    cks = build(r)
    print(cks)


# 设置段落 + 句子的最长长度
def merge():
    i = 0
    chunks = [{"x0": 48.0, "x1": 223.33333333333334, "top": 34.0, "text": " frontiers  Frontiers in Psychiatry", "bottom": 49.0, "page_number": 1, "layout_type": ""}, {"x0": 222.33333333333334, "x1": 478.6666666666667, "top": 136.0, "text": "Depression with comorbid", "bottom": 153.33333333333334, "page_number": 1, "layout_type": "title", "layoutno": "title-0"}, {"x0": 50.666666666666664, "x1": 125.0, "top": 137.66666666666666, "text": " Check for updates", "bottom": 150.0, "page_number": 1, "layout_type": "text", "layoutno": "text-0"}, {"x0": 221.66666666666666, "x1": 535.3333333333334, "top": 161.33333333333334, "text": "borderline personality disorder -", "bottom": 179.0, "page_number": 1, "layout_type": "title", "layoutno": "title-0"}, {"x0": 48.666666666666664, "x1": 97.66666666666667, "top": 170.0, "text": "OPEN ACCESS", "bottom": 179.66666666666666, "page_number": 1, "layout_type": "title", "layoutno": "title-1"}, {"x0": 48.0, "x1": 80.66666666666667, "top": 184.0, "text": "EDITED BY", "bottom": 196.33333333333334, "page_number": 1, "layout_type": ""}, {"x0": 220.66666666666666, "x1": 421.0, "top": 186.0, "text": "could ketamine be a", "bottom": 203.33333333333334, "page_number": 1, "layout_type": "title", "layoutno": "title-0"}, {"x0": 48.666666666666664, "x1": 110.66666666666667, "top": 194.66666666666666, "text": "Giovanni Martinotti", "bottom": 203.33333333333334, "page_number": 1, "layout_type": ""}, {"x0": 47.666666666666664, "x1": 196.66666666666666, "top": 201.66666666666666, "text": "University of Studies G. d'Annunzio Chieti and", "bottom": 214.0, "page_number": 1, "layout_type": ""}, {"x0": 220.66666666666666, "x1": 407.6666666666667, "top": 210.33333333333334, "text": "treatment catalyst?", "bottom": 229.0, "page_number": 1, "layout_type": "title", "layoutno": "title-0"}, {"x0": 48.0, "x1": 92.33333333333333, "top": 211.33333333333334, "text": " Pescara, Italy", "bottom": 223.66666666666666, "page_number": 1, "layout_type": ""}, {"x0": 48.666666666666664, "x1": 87.0, "top": 227.0, "text": "REVIEWED BY", "bottom": 236.0, "page_number": 1, "layout_type": ""}, {"x0": 49.666666666666664, "x1": 110.66666666666667, "top": 236.0, "text": "Giacomo d'Andrea,", "bottom": 245.33333333333334, "page_number": 1, "layout_type": ""}, {"x0": 47.666666666666664, "x1": 197.66666666666666, "top": 243.66666666666666, "text": "University of Studies G. d'Annunzio Chieti and ", "bottom": 256.0, "page_number": 1, "layout_type": ""}, {"x0": 221.66666666666666, "x1": 534.6666666666666, "top": 243.66666666666666, "text": "Magdalena Wiedtocha *, Piotr Marcinowicz', Jan Komarnicki2", "bottom": 256.0, "page_number": 1, "layout_type": "text", "layoutno": "text-1"}, {"x0": 48.333333333333336, "x1": 92.33333333333333, "top": 252.33333333333334, "text": " Pescara, Italy", "bottom": 263.0, "page_number": 1, "layout_type": ""}, {"x0": 219.66666666666666, "x1": 474.3333333333333, "top": 257.6666666666667, "text": "Matgorzata Tobiaszewska?, Weronika Debowska1.", "bottom": 272.6666666666667, "page_number": 1, "layout_type": "text", "layoutno": "text-1"}, {"x0": 48.666666666666664, "x1": 89.66666666666667, "top": 262.0, "text": "Valerio Ricci,", "bottom": 271.6666666666667, "page_number": 1, "layout_type": ""}, {"x0": 48.0, "x1": 188.0, "top": 271.0, "text": "San Luigi Gonzaga University Hospital, Italy", "bottom": 282.3333333333333, "page_number": 1, "layout_type": ""}, {"x0": 220.66666666666666, "x1": 400.6666666666667, "top": 275.3333333333333, "text": "Marta Debowska and Agata Szulc1", "bottom": 287.6666666666667, "page_number": 1, "layout_type": "text", "layoutno": "text-1"}, {"x0": 49.0, "x1": 106.33333333333333, "top": 285.0, "text": "*CORRESPONDENCE", "bottom": 294.6666666666667, "page_number": 1, "layout_type": ""}, {"x0": 220.66666666666666, "x1": 526.6666666666666, "top": 293.6666666666667, "text": "Department of Psychiatry, Faculty of Health Sciences, Medical University of Warsaw, Pruszkow,", "bottom": 305.0, "page_number": 1, "layout_type": "text", "layoutno": "text-3"}, {"x0": 48.666666666666664, "x1": 122.33333333333333, "top": 294.6666666666667, "text": "Magdalena Wiedtocha", "bottom": 306.0, "page_number": 1, "layout_type": ""}, {"x0": 222.33333333333334, "x1": 532.0, "top": 304.3333333333333, "text": "Masovian, Poland, 2Leszek Giec Upper-Silesian Medical Centre of the Medical University of Silesia,", "bottom": 313.0, "page_number": 1, "layout_type": "text", "layoutno": "text-3"}, {"x0": 221.66666666666666, "x1": 464.3333333333333, "top": 312.0, "text": "Katowice, Poland, 3Medical University of Warsaw, Warsaw, Masovian, Poland", "bottom": 323.6666666666667, "page_number": 1, "layout_type": "text", "layoutno": "text-3"}, {"x0": 48.0, "x1": 127.66666666666667, "top": 319.0, "text": "RECEIVED 10 March 2024", "bottom": 330.6666666666667, "page_number": 1, "layout_type": "text", "layoutno": "text-2"}, {"x0": 48.666666666666664, "x1": 125.0, "top": 329.6666666666667, "text": "ACCEPTED 15 April 2024", "bottom": 341.0, "page_number": 1, "layout_type": "text", "layoutno": "text-2"}, {"x0": 48.0, "x1": 126.66666666666667, "top": 338.3333333333333, "text": "PUBLISHED 29 April 2024", "bottom": 349.6666666666667, "page_number": 1, "layout_type": "text", "layoutno": "text-2"}, {"x0": 220.66666666666666, "x1": 547.0, "top": 348.0, "text": "Borderline personality disorder (BPD) is diagnosed in 10-30% of patients with", "bottom": 359.3333333333333, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 49.666666666666664, "x1": 79.0, "top": 354.3333333333333, "text": "CITATION", "bottom": 364.0, "page_number": 1, "layout_type": ""}, {"x0": 219.66666666666666, "x1": 547.6666666666666, "top": 360.3333333333333, "text": "major depressive disorder (MDD), and the frequency of MDD among individuals ", "bottom": 371.6666666666667, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 48.0, "x1": 190.66666666666666, "top": 362.0, "text": "Wiedtocha M, Marcinowicz P, Komarnicki J,", "bottom": 373.3333333333333, "page_number": 1, "layout_type": ""}, {"x0": 48.666666666666664, "x1": 193.33333333333334, "top": 371.6666666666667, "text": "Tobiaszewska M, Debowska W, Debowska M", "bottom": 383.0, "page_number": 1, "layout_type": ""}, {"x0": 220.66666666666666, "x1": 547.0, "top": 372.6666666666667, "text": "with BPD reaches over 80%. The comorbidity of MDD and BPD is associated with", "bottom": 384.0, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 48.666666666666664, "x1": 197.66666666666666, "top": 381.3333333333333, "text": "and Szulc A (2024) Depression with comorbid ", "bottom": 392.6666666666667, "page_number": 1, "layout_type": ""}, {"x0": 219.66666666666666, "x1": 548.6666666666666, "top": 385.0, "text": "more severe depressive symptoms and functional impairment, higher risk of ", "bottom": 397.3333333333333, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 49.666666666666664, "x1": 173.66666666666666, "top": 391.0, "text": "borderline personality disorder - could", "bottom": 399.6666666666667, "page_number": 1, "layout_type": ""}, {"x0": 219.66666666666666, "x1": 547.6666666666666, "top": 398.0, "text": "treatment resistance and increased suicidality. The effectiveness of ketamine", "bottom": 409.3333333333333, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 49.666666666666664, "x1": 157.66666666666666, "top": 399.0, "text": "ketamine be a treatment catalyst?", "bottom": 410.3333333333333, "page_number": 1, "layout_type": ""}, {"x0": 48.666666666666664, "x1": 142.66666666666666, "top": 408.6666666666667, "text": "Front. Psychiatry 15:1398859.", "bottom": 417.3333333333333, "page_number": 1, "layout_type": ""}, {"x0": 219.66666666666666, "x1": 547.6666666666666, "top": 410.3333333333333, "text": "usage in treatment resistant depression (TRD) has been demonstrated in", "bottom": 422.6666666666667, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 48.0, "x1": 155.0, "top": 417.3333333333333, "text": "doi: 10.3389/fpsyt.2024.1398859", "bottom": 428.6666666666667, "page_number": 1, "layout_type": ""}, {"x0": 219.66666666666666, "x1": 548.6666666666666, "top": 422.6666666666667, "text": "numerous studies. In most of these studies, individuals with BPD were not ", "bottom": 434.0, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 48.666666666666664, "x1": 83.33333333333333, "top": 431.3333333333333, "text": "COPYRIGHT", "bottom": 441.0, "page_number": 1, "layout_type": ""}, {"x0": 220.66666666666666, "x1": 547.6666666666666, "top": 435.0, "text": "excluded, thus given the high co-occurrence of these disorders, it is possible", "bottom": 446.3333333333333, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 48.666666666666664, "x1": 190.66666666666666, "top": 439.3333333333333, "text": "@ 2024 Wiedtocha, Marcinowicz, Komarnicki.", "bottom": 450.6666666666667, "page_number": 1, "layout_type": ""}, {"x0": 219.66666666666666, "x1": 547.6666666666666, "top": 448.0, "text": "that the beneficial effects of ketamine also extend to the subpopulation with", "bottom": 459.3333333333333, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 48.666666666666664, "x1": 179.0, "top": 449.0, "text": "Tobiaszewska, Debowska, Debowska and ", "bottom": 460.3333333333333, "page_number": 1, "layout_type": ""}, {"x0": 48.666666666666664, "x1": 196.66666666666666, "top": 457.6666666666667, "text": "Szulc. This is an open-access article distributed ", "bottom": 469.0, "page_number": 1, "layout_type": ""}, {"x0": 220.66666666666666, "x1": 547.0, "top": 460.3333333333333, "text": "comorbid TRD and BPD. However, no protocols were developed that would", "bottom": 471.6666666666667, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 49.666666666666664, "x1": 181.66666666666666, "top": 468.3333333333333, "text": "under the terms of the Creative Commons", "bottom": 477.0, "page_number": 1, "layout_type": ""}, {"x0": 220.66666666666666, "x1": 547.6666666666666, "top": 472.6666666666667, "text": "account for comorbidity. Moreover, psychotherapeutic interventions, which may", "bottom": 485.0, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 48.0, "x1": 164.0, "top": 475.0, "text": "Attribution License (CC BY). The use.", "bottom": 486.6666666666667, "page_number": 1, "layout_type": ""}, {"x0": 48.666666666666664, "x1": 190.66666666666666, "top": 485.0, "text": "distribution or reproduction in other forums", "bottom": 496.3333333333333, "page_number": 1, "layout_type": ""}, {"x0": 219.66666666666666, "x1": 546.0, "top": 485.0, "text": "be crucial for achieving a lasting therapeutic effect in TRD and BPD comorbidity.", "bottom": 497.0, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 50.666666666666664, "x1": 188.0, "top": 495.3333333333333, "text": "s permitted, provided the original author(s)", "bottom": 504.0, "page_number": 1, "layout_type": ""}, {"x0": 219.66666666666666, "x1": 547.6666666666666, "top": 498.0, "text": "were not included. In the article, we discuss the results of a small number of", "bottom": 509.3333333333333, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 48.666666666666664, "x1": 190.66666666666666, "top": 503.3333333333333, "text": "and the copyright owner(s) are credited and", "bottom": 514.6666666666666, "page_number": 1, "layout_type": ""}, {"x0": 220.66666666666666, "x1": 547.6666666666666, "top": 510.3333333333333, "text": "existing studies and case reports on the use of ketamine in depressive disorders", "bottom": 521.6666666666666, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 48.666666666666664, "x1": 191.33333333333334, "top": 512.0, "text": "that the original publication in this journal is", "bottom": 523.3333333333334, "page_number": 1, "layout_type": ""}, {"x0": 48.666666666666664, "x1": 195.0, "top": 520.6666666666666, "text": "cited, in accordance with accepted academic", "bottom": 532.3333333333334, "page_number": 1, "layout_type": ""}, {"x0": 219.66666666666666, "x1": 547.6666666666666, "top": 522.6666666666666, "text": "with comorbid BPD. We elucidate how, at the molecular and brain network", "bottom": 534.0, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 49.666666666666664, "x1": 196.0, "top": 529.6666666666666, "text": "practice. No use, distribution or reproduction", "bottom": 541.0, "page_number": 1, "layout_type": ""}, {"x0": 219.0, "x1": 546.0, "top": 534.6666666666666, "text": "levels, ketamine can impact the neurobiology and symptoms of BPD.", "bottom": 547.0, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 48.666666666666664, "x1": 183.33333333333334, "top": 539.3333333333334, "text": "is permitted which does not comply with ", "bottom": 550.6666666666666, "page_number": 1, "layout_type": ""}, {"x0": 220.66666666666666, "x1": 547.0, "top": 548.0, "text": "Furthermore, we explore whether ketamine-induced neuroplasticity.", "bottom": 560.3333333333334, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 48.666666666666664, "x1": 88.66666666666667, "top": 549.0, "text": "these terms.", "bottom": 558.3333333333334, "page_number": 1, "layout_type": ""}, {"x0": 219.66666666666666, "x1": 547.0, "top": 560.3333333333334, "text": "augmented by psychotherapy, could be of use in alleviating core BPD-related", "bottom": 572.6666666666666, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 221.66666666666666, "x1": 546.0, "top": 571.6666666666666, "text": "symptoms such as emotional dysregulation, self-identity disturbances and self-", "bottom": 584.0, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 220.66666666666666, "x1": 547.6666666666666, "top": 584.6666666666666, "text": "harming behaviors. We also discuss the potential of ketamine-assisted", "bottom": 596.3333333333334, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 220.66666666666666, "x1": 547.6666666666666, "top": 598.0, "text": "psychotherapy (KAP) in BPD treatment. As there is no standard approach to the", "bottom": 610.3333333333334, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 220.66666666666666, "x1": 547.6666666666666, "top": 610.3333333333334, "text": "application of ketamine or KAP in individuals with comorbid TRD and BPD, we", "bottom": 621.6666666666666, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 219.66666666666666, "x1": 547.6666666666666, "top": 622.3333333333334, "text": " consider further research in the field as imperative. The priorities should include", "bottom": 634.6666666666666, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 220.66666666666666, "x1": 547.6666666666666, "top": 635.6666666666666, "text": "development of dedicated protocols, distinguishing subpopulations that may", "bottom": 647.0, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 220.66666666666666, "x1": 547.6666666666666, "top": 648.0, "text": "benefit most from such treatment and investigating factors that may influence its", "bottom": 660.3333333333334, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 220.0, "x1": 321.0, "top": 659.3333333333334, "text": "effectiveness and safety.", "bottom": 671.6666666666666, "page_number": 1, "layout_type": "text", "layoutno": "text-4"}, {"x0": 220.66666666666666, "x1": 253.66666666666666, "top": 712.0, "text": "KEYWORDS", "bottom": 720.6666666666666, "page_number": 1, "layout_type": ""}, {"x0": 220.66666666666666, "x1": 547.6666666666666, "top": 724.3333333333334, "text": "ketamine, esketamine, depression, treatment resistant depression (TRD), borderline", "bottom": 735.6666666666666, "page_number": 1, "layout_type": "text", "layoutno": "text-5"}, {"x0": 219.66666666666666, "x1": 450.3333333333333, "top": 735.6666666666666, "text": " personality disorder, ketamine-assisted psychotherapy (KAT)", "bottom": 748.0, "page_number": 1, "layout_type": "text", "layoutno": "text-5"}, {"x0": 498.6666666666667, "x1": 547.3333333333334, "top": 795.6666666666666, "text": "frontiersin.org", "bottom": 808.0, "page_number": 1, "layout_type": ""}, {"x0": 49.666666666666664, "x1": 121.33333333333333, "top": 798.6666666666666, "text": "Frontiers in Psychiatry", "bottom": 810.0, "page_number": 1, "layout_type": ""}, {"x0": 293.3333333333333, "x1": 302.3333333333333, "top": 799.6666666666666, "text": "01", "bottom": 807.3333333333334, "page_number": 1, "layout_type": ""}, {"x0": 49.666666666666664, "x1": 101.0, "top": 876.0, "text": "Wiedtocha et al.", "bottom": 884.6666666666666, "page_number": 2, "layout_type": ""}, {"x0": 304.0, "x1": 547.6666666666666, "top": 916.3333333333333, "text": "depersonalization, which, along with the desire to reduce", "bottom": 928.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 49.666666666666664, "x1": 128.66666666666666, "top": 918.0, "text": "Introduction", "bottom": 930.3333333333333, "page_number": 2, "layout_type": "title", "layoutno": "title-0"}, {"x0": 304.0, "x1": 547.6666666666666, "top": 929.3333333333333, "text": "emotional tension, are the main driving factors for self-harm in", "bottom": 940.6666666666666, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 304.0, "x1": 341.3333333333333, "top": 942.6666666666666, "text": "BPD (20).", "bottom": 952.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 63.0, "x1": 292.6666666666667, "top": 944.3333333333333, "text": "Borderline personality disorder (BPD) is diagnosed in 10-30%", "bottom": 956.6666666666666, "page_number": 2, "layout_type": "text", "layoutno": "text-0"}, {"x0": 316.3333333333333, "x1": 547.0, "top": 953.0, "text": " Self-identity disturbances in BPD manifest as an inconsistent,", "bottom": 964.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 48.666666666666664, "x1": 292.6666666666667, "top": 957.3333333333333, "text": "patients with major depressive disorder (MDD), whereas the", "bottom": 969.0, "page_number": 2, "layout_type": "text", "layoutno": "text-0"}, {"x0": 303.0, "x1": 547.6666666666666, "top": 965.3333333333333, "text": "non-integrated sense of self and unstable, usually negative sef-", "bottom": 980.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 48.0, "x1": 292.6666666666667, "top": 968.6666666666666, "text": "incidence of MDD in BPD individuals ranges from 71% to 83%", "bottom": 981.0, "page_number": 2, "layout_type": "text", "layoutno": "text-0"}, {"x0": 303.0, "x1": 547.0, "top": 979.3333333333333, "text": " esteem (20). Individuals with BPD experience high levels of self-", "bottom": 991.6666666666666, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 48.0, "x1": 291.6666666666667, "top": 982.0, "text": "(1-3). Comorbidity of BPD and MDD negatively affects prognosis", "bottom": 994.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-0"}, {"x0": 304.0, "x1": 547.6666666666666, "top": 991.6666666666666, "text": "criticism, low self-compassion, strongly impaired self-reflection and", "bottom": 1004.0, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 48.666666666666664, "x1": 292.6666666666667, "top": 995.0, "text": "of both disorders and is associated with more severe depressive", "bottom": 1006.6666666666666, "page_number": 2, "layout_type": "text", "layoutno": "text-0"}, {"x0": 304.0, "x1": 547.6666666666666, "top": 1004.0, "text": "disoriented life narratives (19, 22). These disturbances result in", "bottom": 1016.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 47.0, "x1": 293.3333333333333, "top": 1006.6666666666666, "text": "symptoms and functional impairment, delayed time to remission", "bottom": 1021.6666666666666, "page_number": 2, "layout_type": "text", "layoutno": "text-0"}, {"x0": 303.0, "x1": 547.0, "top": 1016.3333333333333, "text": " distrust in their own judgment and long-term difficulties with self-", "bottom": 1028.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 48.666666666666664, "x1": 292.6666666666667, "top": 1019.6666666666666, "text": "and shorter time to relapse (4, 5). Moreover, available treatment", "bottom": 1032.0, "page_number": 2, "layout_type": "text", "layoutno": "text-0"}, {"x0": 304.0, "x1": 548.6666666666666, "top": 1029.3333333333333, "text": "and goal-oriented behavior (20). Moreover, high self-criticism and", "bottom": 1040.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 48.666666666666664, "x1": 293.3333333333333, "top": 1033.0, "text": "options such as antidepressants, electroconvulsive therapy, and", "bottom": 1045.0, "page_number": 2, "layout_type": "text", "layoutno": "text-0"}, {"x0": 304.0, "x1": 469.0, "top": 1041.6666666666665, "text": "low self-compassion are related to NSSI (23).", "bottom": 1053.0, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 48.0, "x1": 292.6666666666667, "top": 1045.0, "text": "psychotherapy are far less efective in such individuals (6-8). In", "bottom": 1057.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-0"}, {"x0": 316.3333333333333, "x1": 545.0, "top": 1054.0, "text": " In patients with MDD and BPD, the prevalence of post-", "bottom": 1066.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 48.0, "x1": 292.6666666666667, "top": 1057.3333333333333, "text": "this article we elucidate how, at the molecular and brain network", "bottom": 1068.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-0"}, {"x0": 303.3333333333333, "x1": 547.6666666666666, "top": 1065.3333333333333, "text": "traumatic stress disorder (PTSD) is significantly higher than in", "bottom": 1077.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 48.666666666666664, "x1": 293.3333333333333, "top": 1069.6666666666665, "text": "levels, ketamine can impact the neurobiology and symptoms of", "bottom": 1082.0, "page_number": 2, "layout_type": "text", "layoutno": "text-0"}, {"x0": 303.0, "x1": 547.6666666666666, "top": 1079.3333333333333, "text": " patients without BPD diagnosis (24). It is estimated that 22-24% of", "bottom": 1091.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 47.0, "x1": 292.6666666666667, "top": 1081.0, "text": " BPD. We also discuss the results of existing studies and case reports", "bottom": 1093.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-0"}, {"x0": 304.0, "x1": 547.0, "top": 1091.6666666666665, "text": "subjects with primary diagnosis of PTSD have comorbid BPD,", "bottom": 1103.0, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 48.666666666666664, "x1": 292.6666666666667, "top": 1095.0, "text": "on the use of ketamine/esketamine in BPD or depressive disorders", "bottom": 1106.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-0"}, {"x0": 303.0, "x1": 547.0, "top": 1103.0, "text": "whereas the prevalence of PTSD in BPD population ranges from 33", "bottom": 1115.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 48.0, "x1": 290.6666666666667, "top": 1106.6666666666665, "text": "with comorbid BPD. Furthermore, we explore whether ketamine-", "bottom": 1118.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-0"}, {"x0": 304.0, "x1": 547.6666666666666, "top": 1116.3333333333333, "text": "to 79% (25, 26). Thus, the comorbidity of BPD and PTSD, as well as", "bottom": 1128.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 48.666666666666664, "x1": 293.3333333333333, "top": 1118.6666666666665, "text": "induced, psychotherapy-augmented neuroplasticity, augmented by", "bottom": 1133.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-0"}, {"x0": 303.0, "x1": 548.6666666666666, "top": 1129.3333333333333, "text": " BPD with PTSD and MDD seems to be relatively frequent. It is", "bottom": 1140.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 47.0, "x1": 292.3333333333333, "top": 1131.0, "text": "psychotherapy, could prove effective in alleviating core BPD-related", "bottom": 1146.0, "page_number": 2, "layout_type": "text", "layoutno": "text-0"}, {"x0": 303.0, "x1": 548.6666666666666, "top": 1141.6666666666665, "text": " perhaps unsurprising given that BPD is considered a potential risk", "bottom": 1154.0, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 48.666666666666664, "x1": 293.3333333333333, "top": 1145.0, "text": "symptoms. Moreover, we discuss the potential of ketamine-assisted", "bottom": 1157.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-0"}, {"x0": 303.3333333333333, "x1": 548.0, "top": 1153.0, "text": "factor for PTSD (24). In comparison with single-disorder groups,", "bottom": 1165.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 47.0, "x1": 241.0, "top": 1157.3333333333333, "text": " psychotherapy (KAP) in MDD with comorbid BPD.", "bottom": 1169.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-0"}, {"x0": 304.0, "x1": 548.6666666666666, "top": 1167.0, "text": "these patients often experienced greater exposure to trauma and", "bottom": 1179.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 303.3333333333333, "x1": 547.6666666666666, "top": 1178.3333333333333, "text": "more severe mood instability (27). Traumatic or disturbed early", "bottom": 1190.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 49.0, "x1": 142.66666666666666, "top": 1191.6666666666665, "text": "Clinical outline", "bottom": 1203.6666666666665, "page_number": 2, "layout_type": "title", "layoutno": "title-1"}, {"x0": 304.0, "x1": 547.0, "top": 1191.6666666666665, "text": "relationship experiences may result in insecure attachment patterns", "bottom": 1204.0, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 304.0, "x1": 548.0, "top": 1203.0, "text": "and impaired emotional processing (28). It is worth mentioning", "bottom": 1215.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 304.0, "x1": 547.0, "top": 1217.0, "text": "that complex PTSD (cPTSD), a diagnostic category added recently", "bottom": 1229.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 63.666666666666664, "x1": 292.6666666666667, "top": 1219.6666666666665, "text": "According to International Classification of Diseases 11th", "bottom": 1232.0, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 304.0, "x1": 547.0, "top": 1229.3333333333333, "text": "to ICD-11, in addition to PTSD symptoms, is characterized by", "bottom": 1241.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 46.0, "x1": 293.3333333333333, "top": 1230.3333333333333, "text": " Revision (ICD-11) borderline personality is a pattern specifer", "bottom": 1245.0, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 302.3333333333333, "x1": 548.6666666666666, "top": 1240.6666666666665, "text": " disturbances in self-organization, which are conceptualized", "bottom": 1253.0, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 48.0, "x1": 294.3333333333333, "top": 1243.3333333333333, "text": "used in combination with a personality disorder category or a", "bottom": 1255.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 304.0, "x1": 422.0, "top": 1254.0, "text": "similarly to BPD symptoms (9).", "bottom": 1266.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-1"}, {"x0": 48.666666666666664, "x1": 293.3333333333333, "top": 1257.3333333333333, "text": "personality difficulty. It may be applied to individuals whose", "bottom": 1269.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 48.666666666666664, "x1": 292.6666666666667, "top": 1269.6666666666665, "text": "personality disturbance is characterized by a pervasive instability", "bottom": 1282.0, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 48.666666666666664, "x1": 291.6666666666667, "top": 1283.0, "text": "of interpersonal relationships, self-image, affects and marked", "bottom": 1295.0, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 305.0, "x1": 539.6666666666666, "top": 1290.0, "text": "Potential neurobiological background", "bottom": 1302.0, "page_number": 2, "layout_type": "title", "layoutno": "title-2"}, {"x0": 48.666666666666664, "x1": 293.3333333333333, "top": 1295.0, "text": "impulsivity (9). Subjects with BPD experience profound mood", "bottom": 1307.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 304.3333333333333, "x1": 418.3333333333333, "top": 1304.6666666666665, "text": "of BPD symptoms", "bottom": 1317.0, "page_number": 2, "layout_type": "title", "layoutno": "title-2"}, {"x0": 48.666666666666664, "x1": 292.6666666666667, "top": 1307.3333333333333, "text": "disturbances, persistent negative affect and excessive emotional", "bottom": 1319.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 48.666666666666664, "x1": 292.6666666666667, "top": 1319.6666666666665, "text": "reactions especially in response to social rejection and", "bottom": 1332.0, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 316.6666666666667, "x1": 548.0, "top": 1331.0, "text": " In BPD brain dysfunction centers around hypoactive anterior", "bottom": 1342.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 48.666666666666664, "x1": 292.6666666666667, "top": 1333.0, "text": "abandonment (10, 11). Both MDD and BPD highly correlate with", "bottom": 1344.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 303.0, "x1": 548.6666666666666, "top": 1343.3333333333333, "text": "cingulate corex (ACC), hyperactive amygdala and insula, as well as", "bottom": 1358.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 48.666666666666664, "x1": 293.3333333333333, "top": 1345.0, "text": "non-suicidal self-injuries (NSSI) (12). NSSI is common in BPD", "bottom": 1356.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 48.666666666666664, "x1": 294.3333333333333, "top": 1357.3333333333333, "text": "patients (50-80% of cases) and approximately 40% of patients", "bottom": 1369.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 304.0, "x1": 547.6666666666666, "top": 1357.3333333333333, "text": "functional dysconnectivity within and between large brain networks", "bottom": 1368.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 302.3333333333333, "x1": 548.6666666666666, "top": 1368.6666666666665, "text": "(11). Although recent meta-analysis showed no consistent pattern", "bottom": 1381.0, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 48.666666666666664, "x1": 292.6666666666667, "top": 1369.6666666666665, "text": "committed more than 50 self-mutilations (13). It is estimated that", "bottom": 1381.0, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 302.3333333333333, "x1": 547.6666666666666, "top": 1381.0, "text": " of alterations in brain activity, it reported a dysfunction of amygdala", "bottom": 1393.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 48.666666666666664, "x1": 294.3333333333333, "top": 1382.6666666666665, "text": "40 to 85% of BPD individuals attempt suicide, usually multiple", "bottom": 1395.0, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 304.0, "x1": 547.6666666666666, "top": 1394.3333333333333, "text": "and ACC during processing of emotional stimuli (29). Goldstein", "bottom": 1406.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 48.666666666666664, "x1": 292.6666666666667, "top": 1395.0, "text": "times, and up to 10% die as a result (13, 14). Soloff et al. found that", "bottom": 1406.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 48.0, "x1": 292.6666666666667, "top": 1406.3333333333333, "text": " comorbidity of BPD with MDD increases the number and severity", "bottom": 1418.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 302.3333333333333, "x1": 547.0, "top": 1406.6666666666665, "text": " et al. found that BPD subjects, when exposed to repeated negative", "bottom": 1418.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 48.0, "x1": 292.6666666666667, "top": 1419.6666666666665, "text": "of suicide attempts (15). A recent study supported findings that", "bottom": 1432.0, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 304.0, "x1": 547.6666666666666, "top": 1419.6666666666665, "text": "stimuli, exhibit amplifed amygdala response. This evidences", "bottom": 1432.0, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 304.0, "x1": 548.6666666666666, "top": 1432.0, "text": "impaired amygdala habituation (30). Extensive response to", "bottom": 1444.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 48.666666666666664, "x1": 293.3333333333333, "top": 1432.6666666666665, "text": "comorbid BPD plays crucial role as a risk factor for suicide attempts", "bottom": 1445.0, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 48.0, "x1": 118.66666666666667, "top": 1445.0, "text": "in depression (16).", "bottom": 1456.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 304.0, "x1": 547.6666666666666, "top": 1445.0, "text": "negatively valenced information is associated with higher anxiety,", "bottom": 1457.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 61.333333333333336, "x1": 292.6666666666667, "top": 1456.3333333333333, "text": " Other core features of BPD include impulsivity, emotional", "bottom": 1468.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 302.3333333333333, "x1": 548.6666666666666, "top": 1456.6666666666665, "text": "aggression and affective instability levels (11). Hyperresponsiveness", "bottom": 1471.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 48.666666666666664, "x1": 294.3333333333333, "top": 1470.6666666666665, "text": "dysregulation and disturbed self-identity (17-19). Impulsive ", "bottom": 1482.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 303.0, "x1": 547.6666666666666, "top": 1470.6666666666665, "text": " of amygdala may prompt individuals to excessively process negative", "bottom": 1482.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 303.0, "x1": 547.6666666666666, "top": 1482.0, "text": " affective stimuli. For BPD subjects, painful stimuli were proven to", "bottom": 1494.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 48.666666666666664, "x1": 293.3333333333333, "top": 1482.6666666666665, "text": "behavior in BPD is closely linked to emotional suffering and low", "bottom": 1495.0, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 48.0, "x1": 293.3333333333333, "top": 1494.3333333333333, "text": "distress tolerance (20). Emotional dysregulation is related to", "bottom": 1506.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 303.0, "x1": 547.6666666666666, "top": 1495.0, "text": " normalize stress levels and amygdala activity, which may explain", "bottom": 1507.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 303.0, "x1": 389.0, "top": 1507.3333333333333, "text": "frequent NSSI (31, 32).", "bottom": 1518.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 48.666666666666664, "x1": 294.3333333333333, "top": 1508.3333333333333, "text": "heightened negative affect, sensitivity, low self-awareness and ", "bottom": 1519.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 317.3333333333333, "x1": 547.6666666666666, "top": 1519.6666666666665, "text": "Baczkowski et al. demonstrated that in BPD, an increase in", "bottom": 1532.0, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 48.666666666666664, "x1": 292.6666666666667, "top": 1520.6666666666665, "text": "deficits in applying regulation strategies (18). Instead of adaptive", "bottom": 1532.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 302.3333333333333, "x1": 548.6666666666666, "top": 1531.0, "text": "connectivity resulting from performing emotional regulation tasks", "bottom": 1546.0, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 48.666666666666664, "x1": 293.3333333333333, "top": 1532.6666666666665, "text": "regulation, maladaptive coping mechanisms are present. These", "bottom": 1545.0, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 48.666666666666664, "x1": 292.6666666666667, "top": 1545.0, "text": "include ruminations, NsSl, impulsive suicidal behaviors and", "bottom": 1557.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 303.0, "x1": 547.6666666666666, "top": 1545.0, "text": "does not occur in regions essential for effortful emotional", "bottom": 1557.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 47.0, "x1": 292.6666666666667, "top": 1555.6666666666665, "text": "substance abuse (11). Soloffet al. observed that negative affectivity", "bottom": 1569.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 304.0, "x1": 547.6666666666666, "top": 1557.3333333333333, "text": "regulation, such as prefrontal cortex (PFC). As a result, cognitive", "bottom": 1569.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 48.0, "x1": 291.6666666666667, "top": 1570.6666666666665, "text": "is linked with clinical severity of suicide attempts and reduced", "bottom": 1582.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 304.0, "x1": 547.0, "top": 1570.6666666666665, "text": "control, which enables reinterpretation of meaning of emotional", "bottom": 1582.6666666666665, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 48.0, "x1": 290.6666666666667, "top": 1582.0, "text": " inhibitory control (21). A high percentage of patients exhibit stress-", "bottom": 1594.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 305.0, "x1": 547.0, "top": 1582.0, "text": "stimuli, is impaired (33). Frontolimbic dysconnectivity hypothesis,", "bottom": 1594.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 48.0, "x1": 293.3333333333333, "top": 1594.3333333333333, "text": "related dissociative experiences such as derealization and", "bottom": 1606.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-2"}, {"x0": 305.0, "x1": 547.0, "top": 1595.0, "text": "which includes deficient top-down control and enhanced bottom-", "bottom": 1607.3333333333333, "page_number": 2, "layout_type": "text", "layoutno": "text-3"}, {"x0": 48.666666666666664, "x1": 121.33333333333333, "top": 1640.6666666666665, "text": "Frontiers in Psychiatry", "bottom": 1652.0, "page_number": 2, "layout_type": ""}, {"x0": 293.3333333333333, "x1": 302.3333333333333, "top": 1641.6666666666665, "text": "02", "bottom": 1649.3333333333333, "page_number": 2, "layout_type": ""}, {"x0": 500.0, "x1": 546.0, "top": 1641.6666666666665, "text": "frontiersin.org", "bottom": 1651.3333333333333, "page_number": 2, "layout_type": ""}]
    chunk_len = len(chunks)
    while i < chunk_len:
        chunks[i]["sort"] = 0
        if chunks[i].get('x0', 0) > 200:
            chunks[i]["sort"] = 1
        i+=1
    chunks.sort(key=lambda x: x['sort'])

    print(chunks)


if __name__ == "__main__":
    #merge()
    main()

