#!/bin/bash

export OLLAMA_DEBUG=0 ollama serve & uv run uvicorn app.main:app --host 0.0.0.0 --port 8081 & uv run python -m app.consumer & uv run python -m app.jd_consumer