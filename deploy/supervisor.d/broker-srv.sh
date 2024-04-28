#!/bin/bash
cd /data/ragflow/ragflow
export PYTHONPATH=/usr/local/lib64/python3.9/site-packages:/usr/local/lib64/python3.9/site-packages:.
python3 rag/svr/task_broker.py