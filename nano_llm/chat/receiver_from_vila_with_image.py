# receiver.py
from flask import Flask, request, jsonify
import os, json, time, base64

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "ingested")
os.makedirs(SAVE_DIR, exist_ok=True)

def _safe_basename_from_image_path(img_path: str) -> str:
    try:
        img_path = (img_path or "").strip()
        base = os.path.splitext(os.path.basename(img_path))[0]
        return base if base else f"image_{int(time.time())}"
    except Exception:
        return f"image_{int(time.time())}"

@app.post("/ingest")
def ingest():
    """
    Accepts a JSON payload of the per-image document.
    Optional transient fields from sender:
      _image_b64, _image_basename, _image_ext
    Saves:
      <basename>.json  (document without transient fields)
      <basename>.jpg   (or original extension if not jpeg)
    """
    doc = request.get_json(force=True, silent=False)
    if not isinstance(doc, dict):
        return jsonify({"ok": False, "error": "invalid JSON"}), 400

    # Pull transient image fields (if present)
    img_b64 = doc.pop("_image_b64", None)
    img_base = doc.pop("_image_basename", None)
    img_ext  = doc.pop("_image_ext", ".jpg") or ".jpg"

    # Determine base filename
    if not img_base:
        img_base = _safe_basename_from_image_path(doc.get("image_path") or "")

    # 1) Save JSON
    json_path = os.path.join(SAVE_DIR, f"{img_base}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    saved = {"json": os.path.abspath(json_path)}

    # 2) Save image if provided
    if isinstance(img_b64, str) and img_b64.strip():
        try:
            raw = base64.b64decode(img_b64, validate=True)
            ext_lower = (img_ext or ".jpg").lower()
            if ext_lower in (".jpg", ".jpeg"):
                img_path = os.path.join(SAVE_DIR, f"{img_base}.jpg")
            else:
                img_path = os.path.join(SAVE_DIR, f"{img_base}{ext_lower}")
            with open(img_path, "wb") as f:
                f.write(raw)
            saved["image"] = os.path.abspath(img_path)
        except Exception as e:
            saved["image_error"] = f"failed to decode/write image: {e}"

    print(f"[ingest] saved: {saved}")
    return jsonify({"ok": True, "saved": saved})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

