#!/bin/bash

ollama serve & uv run uvicorn app.main:app --host 0.0.0.0 --port 8081 $ uv run python -m app.consumer