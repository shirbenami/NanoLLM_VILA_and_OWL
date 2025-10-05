# NanoLLM Custom Project

This repository contains multiple entry points for running **NanoLLM** models with different behaviors and output formats. The scripts are designed to simplify testing and interaction with Visual Language Models (VLMs) such as **VILA1.5-3b**.

Each main file supports a slightly different workflow — from standard image chat to automatic JSON logging or object list extraction for OWL-style processing.

---

## 🧠 Environment Setup

**Docker run command:**
```bash
jetson-containers run nano_llm_custom /bin/bash
```

Then navigate to the chat directory:
```bash
cd /opt/NanoLLM/nano_llm/chat/
```

---

## ⚙️ Files Overview

### 1. `__main__.py`
**Purpose:** Base entry point for standard chat or image inference.

**Run example:**
```bash
python3 -m nano_llm.chat --api=mlc \
  --model Efficient-Large-Model/VILA1.5-3b \
  --max-context-len 256 \
  --max-new-tokens 32
```
**Description:**
- Standard interactive loop for text/image prompts.
- Prints model output directly to the terminal.
- No JSON or additional automation.

---

### 2. `main_with_time_and_json.py`
**Purpose:** Run NanoLLM with timing metrics and automatic JSON logging per image.

**Run example:**
```bash
python3 -m nano_llm.chat \
  --api=mlc \
  --model Efficient-Large-Model/VILA1.5-3b \
  --save-json-by-image
```
**Description:**
- Saves each conversation to `<image>.json` next to the image file.
- Records: timestamp, prompt, and response.
- Includes timing logs such as **TTFT** (Time To First Token), total generation time, and token throughput.
- Supports both streaming and non-streaming modes.

**Output JSON structure:**
```json
{
  "image_path": "/data/images/01.jpg",
  "model": "VILA1.5-3b",
  "api": "mlc",
  "entries": [
    {
      "timestamp": 1730000000,
      "prompt": "/data/images/01.jpg",
      "response": "A black drone with four propellers."
    }
  ]
}
```

---

### 3. `main_list_for_owl.py`
**Purpose:** Extract a **comma-separated list of objects** (nouns) from an image — used later for OWL-style pipelines.

**Run example:**
```bash
python3 -m nano_llm.chat \
  --api=mlc \
  --model Efficient-Large-Model/VILA1.5-3b \
  --max-context-len 256 \
  --max-new-tokens 32 \
  --objects-from-image \
  --save-json-by-image
```

**Description:**
- When given an image path, the model lists **all objects it detects**.
- Output example:
  ```
  drone, propeller, floor, wall
  ```
- Saves responses to `<image>.json`.
- Adds an *auto-boost* loop — if too few objects are found, the script prompts the model again for more distinct objects.

**Parameters:**
- `--min-objects` → Minimum expected object count before retry (default: 3)
- `--max-rounds` → Max retry attempts to request more objects (default: 2)

---

## 📦 Folder Summary

| File | Functionality |
|------|----------------|
| `__main__.py` | Basic image/text chat runner |
| `main_with_time_and_json.py` | Adds timing & JSON saving per image |
| `main_list_for_owl.py` | Extracts a clean object list (for OWL) |

---

## 🧩 Notes

- All scripts depend on **NanoLLM** and **termcolor**.
- Run them from within the Docker container path: `/opt/NanoLLM/nano_llm/chat/`.
- Ensure your model (`VILA1.5-3b`) and API (`mlc`) are available.

---

## 💡 Example Workflow

---

## ⚠️ Important Setup Notes

### 1. NVMe data storage (optional but recommended)
If your data is stored on an external NVMe drive, you can move the Jetson container data there and create a symbolic link:
```bash
sudo rsync -aHAX --info=progress2 /home/user/jetson-containers/data/ /mnt/VLM/jetson-data/
sudo rm -rf /home/user/jetson-containers/data
ln -s /mnt/VLM/jetson-data /home/user/jetson-containers/data
```
This ensures that large model files and datasets are stored on the NVMe drive for faster I/O.

### 2. Custom container with model timing
The main edits for measuring model performance time were done on the **CUSTOM** container, specifically inside:
```
/opt/NanoLLM/nano_llm/chat/__main__.py
```
This version includes detailed timing logs for model inference speed.

### 3. Enabling GPU graphics acceleration inside Docker
Before running scripts that use camera or visualization features, you must install and configure the necessary GStreamer and Avahi packages **inside the container**.

Instead of running the script directly, open a bash session first:
```bash
jetson-containers run -it $(autotag nano_llm) /bin/bash
```
Then inside the container, run:
```bash
apt update
apt install -y gstreamer1.0-nice
apt-get install -y avahi-utils libnss-mdns
service avahi-daemon stop
```
After performing these steps, you can **commit** and **save** the Docker image to preserve the setup and re-run it later with full graphics support.


1. Launch container:
   ```bash
   jetson-containers run nano_llm_custom /bin/bash
   ```
2. Enter the chat directory and open one of the scripts for editing if needed:
   ```bash
   nano /opt/NanoLLM/nano_llm/chat/__main__.py
   ```
3. Run one of the modes:
   - **Basic chat:** `python3 -m nano_llm.chat ...`
   - **With JSON saving:** `python3 -m nano_llm.chat --save-json-by-image`
   - **Object list mode:** `python3 -m nano_llm.chat --objects-from-image --save-json-by-image`

---

## 🧾 License

This project follows the NanoLLM repository license terms. See the original NanoLLM repo for details.
