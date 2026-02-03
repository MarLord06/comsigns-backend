#!/bin/bash
set -e

cd comsigns-backend

echo "Starting ComSigns Backend API..."
echo "  Port: ${PORT:-8000}"
echo "  Device: ${COMSIGNS_DEVICE:-cpu}"

exec uvicorn backend.api.app:app --host 0.0.0.0 --port ${PORT:-8000}
