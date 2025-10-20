#!/bin/bash
# ========================================
# Run all VLM processes on Jetson
# ========================================

echo "[RUN] Starting all VLM services..."

# ---------- 1. VILA API Server ----------
tmux new-session -d -s vila "
echo '[VILA] Starting container...';
jetson-containers run -it \
  --publish 8080:8080 \
  --volume /home/user/jetson-containers/data:/mnt/VLM/jetson-data \
  nano_llm_custom /bin/bash -c '
    python3 -m nano_llm.chat \
      --api=mlc \
      --model Efficient-Large-Model/VILA1.5-3b \
      --max-context-len 256 \
      --max-new-tokens 32 \
      --save-json-by-image \
      --server --port 8080 \
      --notify-url http://172.16.17.12:5050/from_vila
  '
"

# ---------- 2. NanoOWL Object Detector ----------
tmux new-session -d -s nanoowl "
echo '[NanoOWL] Starting detector...';
sudo docker run -it --network host nanoowl_new:v1.4 /bin/bash -c '
  cd examples/jetson_server/ &&
  python3 nanoowl_service.py \
    --engine /opt/nanoowl/data/owl_image_encoder_patch32.engine \
    --host 0.0.0.0 --port 5060
'
"

# ---------- 3. Display Server ----------
tmux new-session -d -s display "
cd /home/user/shir &&
echo '[Display] Starting web viewer...';
python3 display_server.py \
  --root /home/user/jetson-containers/data/images/captures \
  --host 0.0.0.0 --port 8090 --latest-only
"

# ---------- 4. comm_manager.py ----------
tmux new-session -d -s comm "
cd /home/user/shir &&
echo '[Comm Manager] Starting communication server...';
python3 comm_manager.py \
  --host 0.0.0.0 --port 5050 \
  --jetson2-endpoint http://172.16.17.11:5050/prompts \
  --captures-root /home/user/jetson-containers/data/images/captures \
  --nanoowl-endpoint http://172.16.17.12:5060/infer \
  --forward-timeout 25 --forward-retries 7 \
  --nanoowl-timeout 70 --nanoowl-annotate 0 \
  --forward-json-url http://172.16.17.9:9090/ingest
"

echo "✅ Core services started (VILA, NanoOWL, Display, Comm Manager)."
echo "Use 'tmux ls' to view sessions."

# ---------- 5. Capture Frames (manual start) ----------
read -p "🖼️  Do you want to start Capture Frames now? (y/n): " answer
if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    tmux new-session -d -s capture "
    cd /home/user/jetson-containers/data/images &&
    echo '[Capture] Starting frame capture...';
    python3 capture_frames.py \
      --source /dev/video0 \
      --vlm http://172.16.17.12:8080/describe \
      --sleep 10
    "
    echo "🎥 Capture Frames started in tmux session 'capture'."
else
    echo "🚫 Skipped Capture Frames — you can start it later with:"
    echo "   tmux new -s capture 'cd /home/user/jetson-containers/data/images && python3 capture_frames.py --source /dev/video0 --vlm http://172.16.17.12:8080/describe --sleep 10'"
fi

echo "✅ All done. You can check running sessions with 'tmux ls'."
