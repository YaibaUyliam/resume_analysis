#!/bin/bash

uvicorn app.main:app --host 0.0.0.0 --port 8081 --workers 1 \
& uv run python -m app.consumer \
& uv run python -m app.jd_consumer \
& uv run python -m app.cv_del_consumer