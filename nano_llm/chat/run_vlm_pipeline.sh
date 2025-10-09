#!/bin/bash
# ============================================
# 🚀 VLM Real-Time Pipeline Launcher
# Author: Shir
# ============================================

# CONFIGURATION
JETSON_CONTAINER="nano_llm_custom"
VLM_PORT=8080
DATA_DIR="/home/user/jetson-containers/data"
VLM_ENDPOINT="http://172.16.17.12:8080/describe"
CAPTURE_DIR="$DATA_DIR/images"
POSES_FILE="/opt/missions/poses.json"

echo "============================================"
echo "🚀 Starting VILA model container..."
echo "============================================"

# Start Jetson container and run VILA server inside
gnome-terminal -- bash -c "
  jetson-containers run -it \
    --publish ${VLM_PORT}:${VLM_PORT} \
    --volume ${DATA_DIR}:/mnt/VLM/jetson-data \
    ${JETSON_CONTAINER} /bin/bash -c '
      echo \"✅ Container launched. Starting VILA model...\";
      python3 -m nano_llm.chat \
        --api=mlc \
        --model Efficient-Large-Model/VILA1.5-3b \
        --max-context-len 256 \
        --max-new-tokens 32 \
        --save-json-by-image \
        --server --port ${VLM_PORT};
      exec bash'
"

sleep 5
echo ""
echo "============================================"
echo "📷 Launching capture_frames.py..."
echo "============================================"

# Open another terminal and start frame capture
gnome-terminal -- bash -c "
  cd ${CAPTURE_DIR};
  echo \"✅ Starting capture script...\";
  python3 capture_frames.py \
    --source /dev/video0 \
    --poses ${POSES_FILE} \
    --vlm ${VLM_ENDPOINT};
  exec bash"
