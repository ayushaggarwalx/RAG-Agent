#!/bin/bash
set -e

# Start Streamlit frontend on HF's $PORT
/app/.venv/bin/streamlit run frontend/app.py \
    --server.address=0.0.0.0 \
    --server.port=7860  &

# Start FastAPI backend on port 8000
/app/.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000