#!/bin/bash
# Start a worker node
# Usage: ./start_worker.sh [worker_id]
export WORKER_ID=${1:-worker_$(date +%s)}
echo "Starting worker ${WORKER_ID}..."
python -m node.worker
