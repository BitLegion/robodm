#!/bin/bash

mkdir -p droid_10gb \
         droid_50gb \
         droid_200gb \
         droid_500gb \
         droid_1tb \
         droid_5tb 

nohup /home/pratyayp/robodm/venv/bin/python ingest_droid.py --output-dir droid_10gb --target-size-gb 10.0 > droid_10gb/ingestion.log 2>&1 &

nohup /home/pratyayp/robodm/venv/bin/python ingest_droid.py --output-dir droid_50gb --target-size-gb 50.0 > droid_50gb/ingestion.log 2>&1 &

nohup /home/pratyayp/robodm/venv/bin/python ingest_droid.py --output-dir droid_200gb --target-size-gb 200.0 > droid_200gb/ingestion.log 2>&1 &

nohup /home/pratyayp/robodm/venv/bin/python ingest_droid.py --output-dir droid_500gb --target-size-gb 500.0 > droid_500gb/ingestion.log 2>&1 &

nohup /home/pratyayp/robodm/venv/bin/python ingest_droid.py --output-dir droid_1tb --target-size-gb 1024.0 > droid_1tb/ingestion.log 2>&1 &

nohup /home/pratyayp/robodm/venv/bin/python ingest_droid.py --output-dir droid_5tb --target-size-gb 5120.0 > droid_5tb/ingestion.log 2>&1 &