#!/bin/bash
cd /data/ragflow/ragflow
export PYTHONPATH=/usr/local/lib64/python3.9/site-packages:/usr/local/lib64/python3.9/site-packages:.
python3 api/ragflow_server.py