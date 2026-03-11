#!/bin/bash

uvicorn app.main:app --host 0.0.0.0 --port 8081 --workers 1 \
& python -m app.consumer \
& python -m app.jd_consumer \
& python -m app.cv_del_consumer