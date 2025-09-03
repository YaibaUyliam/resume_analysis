#!/bin/bash

OLLAMA_FLASH_ATTENTION=1 ollama serve & uv run uvicorn app.main:app --host 0.0.0.0 --port 9001