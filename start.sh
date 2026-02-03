#!/bin/bash
set -e

cd comsigns-backend

echo "Installing dependencies..."
python3 -m pip install -r requirements.txt

echo "Starting ComSigns Backend API..."
echo "  Port: ${PORT:-8000}"
echo "  Device: ${COMSIGNS_DEVICE:-cpu}"

exec python3 -m uvicorn backend.api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
