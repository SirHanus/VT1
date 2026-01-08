#!/bin/bash
set -e

echo "Starting YOLO ML Backend..."
echo "Model: $YOLO_MODEL"
echo "Confidence Threshold: $CONF_THRESHOLD"

# Start the label-studio-ml server
exec label-studio-ml start /app --host 0.0.0.0 --port 9090 --log-level ${LOG_LEVEL:-INFO}

