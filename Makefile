.SHELL := /usr/bin/env bash

.PHONY: all help setup dev build clean

all: help

help:
	@echo "Usage: make [target]"
	@echo "Available targets:"
	@echo "  setup        Run the setup script to configure the project"
	@echo "  run-dev      Start dev server"

setup:
	@echo "🔧 Running setup.sh…"
	@bash setup.sh

run-dev:
	@echo "🚀 Starting development server…"
	@bash -c 'trap "echo "\n🛑 Development server stopped"; exit 0" SIGINT; \
	uv run uvicorn app.main:app --reload --port 8000 --timeout-keep-alive 300 --timeout-graceful-shutdown 300'

run:
	@echo "🚀 Starting development server…"
	@bash -c 'uv run uvicorn app.main:app --port 8000'