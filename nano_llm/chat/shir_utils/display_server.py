#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM Ingest DISPLAY Server — Cute Web GUI for viewing ingested images + captions

What it does
------------
- Serves a lightweight web UI that shows every <image + description> pair received
  into an "ingested" directory (from your receiver `/ingest`).
- Auto-refreshes every 2 seconds to pick up new arrivals.
- Click a card to open a modal with a large preview and the raw JSON metadata.

Assumptions about files in ROOT_DIR
-----------------------------------
- For each image there is a matching JSON file with the SAME BASENAME.
  Example:
    ROOT_DIR/
      2025-10-12_13-05-11_image.jpg
      2025-10-12_13-05-11_image.json
- Image extension can vary (.jpg/.png/.jpeg/...).
- JSON schema is flexible. We attempt to extract a human description from common keys.

Usage
-----
python3 display_server.py --root /path/to/ingested --host 0.0.0.0 --port 8090

Dependencies
------------
- Flask only (no sockets or DB needed)
  pip install flask

Notes
-----
- Designed to run on the same machine that stores the ingested folder, 
  but can also run on a different host if it mounts that folder.
- If your receiver saves with different names, you can adapt the MATCH_GLOB below.
"""

import argparse
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template_string, send_from_directory, request, abort

# -----------------
# Config & parsing
# -----------------

def parse_args():
    p = argparse.ArgumentParser(description="VLM Ingest DISPLAY Server")
    p.add_argument("--root", required=True, help="Directory holding ingested files (images + json)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--scan-interval", type=float, default=2.0, help="Seconds between UI auto-refreshes")
    return p.parse_args()

# -----------------
# Model
# -----------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
JSON_EXT = ".json"

@dataclass
class Item:
    basename: str              # file base w/o extension
    image_rel: str             # relative path from ROOT
    json_rel: Optional[str]    # relative path from ROOT (may be None)
    mtime: float               # latest mtime among image/json
    text: str                  # extracted caption/answer (best-effort)


# -----------------
# Utilities
# -----------------

def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def _best_json_for_image(img_path: Path) -> Optional[Path]:
    """Return JSON that sits next to the image (same basename + .json)."""
    cand = img_path.with_suffix(JSON_EXT)
    return cand if cand.exists() else None

def _extract_text(doc: dict) -> str:
    """
    Extract clean human-readable text for display.
    Priority: entries[0].response → response_describe → fallback keys.
    """
    try:
        # ✅ אם יש entries — נציג רק את הראשון
        entries = doc.get("entries")
        if isinstance(entries, list) and len(entries) > 0:
            v = entries[0].get("response")
            if isinstance(v, str) and v.strip():
                return v.strip().replace("</s>", "").strip()
    except Exception:
        pass

    # fallback – response_describe (אם אין entries)
    v = doc.get("response_describe")
    if isinstance(v, str) and v.strip():
        return v.strip().replace("</s>", "").strip()

    # fallback – תיאור כללי
    for key in ("description", "caption", "summary", "output", "response"):
        v = doc.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip().replace("</s>", "").strip()

    return "(no textual description found in JSON)"
def _collect_items(root: Path) -> List[Item]:
    items: List[Item] = []

    # Find all images; for each, try to find a JSON with the same basename
    for img_path in root.glob("**/*"):
        if not img_path.is_file():
            continue
        if not _is_image(img_path):

            continue

        basename = img_path.stem
        #json_path = _best_json_for_basename(root, basename)
        json_path = _best_json_for_image(img_path)

        text = ""
        mtime_list = [img_path.stat().st_mtime]
        if json_path and json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                text = _extract_text(doc)
                mtime_list.append(json_path.stat().st_mtime)
            except Exception:
                text = "(failed to read/parse JSON)"

        items.append(
            Item(
                basename=basename,
                image_rel=str(img_path.relative_to(root)),
                json_rel=str(json_path.relative_to(root)) if json_path else None,
                mtime=max(mtime_list),
                text=text,
            )
        )

    # newest first
    items.sort(key=lambda it: it.mtime, reverse=True)
    return items


# -----------------
# Flask app
# -----------------

def create_app(root_dir: Path, scan_interval: float) -> Flask:
    app = Flask(__name__)
    app.config["ROOT_DIR"] = root_dir
    app.config["SCAN_INTERVAL"] = scan_interval

    @app.get("/")
    def index():
        return render_template_string(INDEX_HTML,
            scan_interval=app.config["SCAN_INTERVAL"]
        )

    @app.get("/api/items")
    def api_items():
        root: Path = app.config["ROOT_DIR"]
        items = _collect_items(root)
        payload = [
            {
                "basename": it.basename,
                "image": f"/img/{it.image_rel}",
                "json": (f"/meta/{it.json_rel}" if it.json_rel else None),
                "mtime": it.mtime,
                "text": it.text,
            }
            for it in items
        ]
        return jsonify({"ok": True, "count": len(payload), "items": payload})

    @app.get("/img/<path:rel>")
    def serve_image(rel: str):
        root: Path = app.config["ROOT_DIR"]
        full = (root / rel).resolve()
        if not str(full).startswith(str(root.resolve())):
            abort(403)
        if not full.exists() or not full.is_file():
            abort(404)
        directory = str(full.parent)
        filename = full.name
        return send_from_directory(directory, filename)

    @app.get("/meta/<path:rel>")
    def serve_json(rel: str):
        root: Path = app.config["ROOT_DIR"]
        full = (root / rel).resolve()
        if not str(full).startswith(str(root.resolve())):
            abort(403)
        if not full.exists() or not full.is_file():
            abort(404)
        directory = str(full.parent)
        filename = full.name
        return send_from_directory(directory, filename, mimetype="application/json")

    return app


# -----------------
# HTML template (inline)
# -----------------

INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VLM Ingest Viewer</title>
  <style>
    :root { --bg:#0f172a; --card:#111827; --ink:#e2e8f0; --muted:#9ca3af; --accent:#22d3ee; }
    html,body { margin:0; padding:0; background:var(--bg); color:var(--ink); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
    header { display:flex; gap:12px; align-items:center; padding:14px 18px; border-bottom:1px solid #1f2937; position:sticky; top:0; background:linear-gradient(180deg, rgba(15,23,42,.95), rgba(15,23,42,.75)); backdrop-filter: blur(6px); }
    h1 { margin:0; font-size:18px; letter-spacing:0.3px; }
    .badge { background:#0ea5b7; color:#002227; padding:2px 8px; border-radius:999px; font-weight:700; font-size:12px; }
    .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap:14px; padding:18px; }
    .card { background:var(--card); border:1px solid #1f2937; border-radius:16px; overflow:hidden; box-shadow:0 8px 24px rgba(0,0,0,.25); transition: transform .15s ease, box-shadow .15s ease; }
    .card:hover { transform: translateY(-2px); box-shadow:0 12px 28px rgba(0,0,0,.35); }
    .thumb { width:100%; height:180px; object-fit:cover; display:block; background:#0b1220; }
    .body { padding:12px 14px 14px; display:flex; flex-direction:column; gap:8px; }
    .title { display:flex; align-items:center; gap:8px; justify-content:space-between; }
    .basename { font-size:13px; color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:70%; }
    .mtime { font-size:12px; color:#7dd3fc; }
    .text { font-size:14px; line-height:1.35; color:#e5e7eb; max-height:72px; overflow:hidden; mask-image: linear-gradient(to bottom, black 70%, transparent 100%); }
    .row { display:flex; gap:8px; align-items:center; }
    .btn { cursor:pointer; border:1px solid #1f2937; background:#0b1325; color:#e2e8f0; padding:8px 10px; border-radius:12px; font-size:13px; }
    .btn:hover { background:#0e162f; }

    .footer { color:var(--muted); font-size:12px; padding:6px 18px 14px; text-align:center; }

    /* Modal */
    .modal { position:fixed; inset:0; display:none; background:rgba(0,0,0,.5); align-items:center; justify-content:center; padding:20px; }
    .modal.open { display:flex; }
    .modal-card { max-width:1100px; width:100%; max-height:90vh; background:#0b1220; border:1px solid #253044; border-radius:18px; overflow:hidden; display:grid; grid-template-columns: 1.1fr 0.9fr; }
    .modal-left { background:#0a0f1b; border-right:1px solid #1e293b; display:flex; align-items:center; justify-content:center; }
    .modal-left img { width:100%; height:100%; object-fit:contain; }
    .modal-right { padding:14px; display:flex; flex-direction:column; gap:10px; }
    .modal-head { display:flex; justify-content:space-between; align-items:center; gap:12px; }
    .modal-title { font-size:14px; color:#93c5fd; }
    .close { cursor:pointer; font-size:22px; line-height:22px; padding:4px 10px; color:#93c5fd; border:1px solid #1f2937; border-radius:10px; }
    pre { margin:0; padding:12px; background:#0b1629; border:1px solid #1f2937; border-radius:12px; color:#c7d2fe; font-size:12px; overflow:auto; max-height:60vh; }

  </style>
</head>
<body>
  <header>
    <h1>📸 VLM Ingest Viewer</h1>
    <span class="badge" id="count">0</span>
    <div style="margin-left:auto; display:flex; gap:10px; align-items:center;">
      <small style="color:var(--muted)">Auto-refresh <b id="interval"></b> sec</small>
      <button class="btn" id="refreshBtn">Refresh now</button>
    </div>
  </header>

  <main class="grid" id="grid"></main>
  <div class="footer">Serving images + metadata from your ingested folder • VLM on Jetson ♥</div>

  <div class="modal" id="modal">
    <div class="modal-card">
      <div class="modal-left"><img id="modalImg" alt="preview"/></div>
      <div class="modal-right">
        <div class="modal-head">
          <div class="modal-title" id="modalTitle"></div>
          <button class="close" id="closeBtn">×</button>
        </div>
        <div class="row"><small id="modalText" style="color:#e5e7eb"></small></div>
        <pre id="modalJson">{}</pre>
      </div>
    </div>
  </div>

  <script>
    const SCAN_INTERVAL = {{ scan_interval|tojson }};
    let timer = null;

    function fmtTime(ts){
      try { return new Date(ts*1000).toLocaleString(); } catch(e){ return String(ts); }
    }

    function render(items){
      const grid = document.getElementById('grid');
      document.getElementById('count').textContent = items.length;
      grid.innerHTML = items.map(it => `
        <div class="card" onclick="openModal(${encodeURIComponent(JSON.stringify(JSON.stringify(it)))})">
          <img class="thumb" src="${it.image}" alt="${it.basename}" />
          <div class="body">
            <div class="title">
              <div class="basename" title="${it.basename}">${it.basename}</div>
              <div class="mtime">${fmtTime(it.mtime)}</div>
            </div>
            <div class="text">${(it.text||'').replace(/[\n\r]+/g,' ')}</div>
            <div class="row">
              ${it.json ? `<a class="btn" href="${it.json}" target="_blank">Open JSON</a>` : ''}
              <a class="btn" href="${it.image}" target="_blank">Open Image</a>
            </div>
          </div>
        </div>
      `).join('');
    }

    async function load(){
      try{
        const r = await fetch('/api/items');
        const js = await r.json();
        if(js && js.ok) render(js.items);
      }catch(e){ console.error(e); }
    }

    function start(){
      document.getElementById('interval').textContent = SCAN_INTERVAL;
      load();
      timer = setInterval(load, SCAN_INTERVAL*1000);
    }

    function stop(){ if(timer){ clearInterval(timer); timer = null; } }

    document.getElementById('refreshBtn').addEventListener('click', ()=>{ stop(); load(); start(); });

    async function openModal(serialized){
      const it = JSON.parse(JSON.parse(decodeURIComponent(serialized)));
      document.getElementById('modalImg').src = it.image;
      document.getElementById('modalTitle').textContent = it.basename + ' — ' + fmtTime(it.mtime);
      document.getElementById('modalText').textContent = (it.text||'').replace(/[\n\r]+/g,' ');
      let pretty = '{}';
      try{
        if(it.json){
          const r = await fetch(it.json);
          const js = await r.json();
          pretty = JSON.stringify(js, null, 2);
        }
      }catch(e){ pretty = '(failed to load json)'; }
      document.getElementById('modalJson').textContent = pretty;
      document.getElementById('modal').classList.add('open');
    }

    document.getElementById('closeBtn').addEventListener('click', ()=>{
      document.getElementById('modal').classList.remove('open');
    });
    document.getElementById('modal').addEventListener('click', (e)=>{
      if(e.target.id === 'modal') document.getElementById('modal').classList.remove('open');
    });

    start();
  </script>
</body>
</html>
"""


# -----------------
# Entrypoint
# -----------------

def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"ROOT dir not found: {root}")

    app = create_app(root, args.scan_interval)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
