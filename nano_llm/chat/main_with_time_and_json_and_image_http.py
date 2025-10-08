#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified script that supports:
1) CLI mode (interactive prompts as before).
2) HTTP API server mode (--server) to accept image_path (and optional question),
   run VILA, and append to the same JSON-by-image file.

Additions:
- After writing the per-image JSON, the document is forwarded via HTTP POST
  to a remote receiver (default: http://172.16.17.11:5000/ingest),
  configurable with --forward-url.
"""

import os
import sys
import time
import json
import signal
import logging
import threading

import base64
import urllib.request, urllib.error
from urllib.parse import urlparse

from termcolor import cprint
import numpy as np

# NanoLLM stack (assumes your existing environment)
from nano_llm import NanoLLM, ChatHistory, ChatTemplates, BotFunctions
from nano_llm.utils import ImageExtensions, ArgParser, KeyboardInterrupt, load_prompts, print_table

# ---------------------------
# Lightweight HTTP client (stdlib)
# ---------------------------
import urllib.request
import urllib.error

def _http_post_json(url: str, payload: dict, timeout: float = 6.0) -> tuple[int, str]:
    """
    POST 'payload' as JSON to 'url' using Python stdlib (no 'requests' dependency).
    Returns: (status_code, response_text) or raises on malformed URL.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            body = resp.read().decode("utf-8", errors="replace")
            return status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
        return e.code, body
    except Exception as e:
        # Network/timeout/DNS errors end up here; caller will log a warning.
        return -1, str(e)


# ---------------------------
# Helpers for JSON by image
# ---------------------------


def _read_image_bytes(path_or_url: str, max_bytes: int = 25*1024*1024):
    """
    Read image bytes from a local path or URL.
    Returns (data_bytes, base_name, ext) or (b"", "image", ".jpg") on failure.
    """
    p = (path_or_url or "").strip().strip("'").strip('"')
    try:
        if p.lower().startswith(("http://", "https://")):
            with urllib.request.urlopen(p, timeout=6.0) as resp:
                data = resp.read(max_bytes + 1)
            parsed = urlparse(p)
            base = os.path.splitext(os.path.basename(parsed.path) or "image")[0]
        else:
            with open(p, "rb") as f:
                data = f.read(max_bytes + 1)
            base = os.path.splitext(os.path.basename(p) or "image")[0]
        if len(data) > max_bytes:
            raise ValueError(f"image too large (> {max_bytes} bytes)")
        ext = _ext_of(p) or ".jpg"
        return data, base, ext
    except Exception:
        return b"", "image", ".jpg"



def _ext_of(path: str) -> str:
    """Return the lowercase extension of a local path or URL."""
    p = path.strip().strip("'").strip('"')
    if p.lower().startswith(("http://", "https://")):
        parsed = urlparse(p)
        _, ext = os.path.splitext(parsed.path)
        return ext.lower()
    _, ext = os.path.splitext(p)
    return ext.lower()

def _is_image_path_or_url(user_text: str) -> bool:
    """Detect if the user_text looks like an image path or image URL."""
    if not user_text:
        return False
    ext = _ext_of(user_text)
    normalized = {e if e.startswith(".") else f".{e}" for e in (ImageExtensions if isinstance(ImageExtensions, (list, set, tuple)) else [])}
    if not normalized:
        # Fallback if ImageExtensions is not provided
        normalized = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
    return ext in normalized

def _json_path_for_image(image_path_or_url: str) -> str:
    """Return the JSON filename that corresponds to the image (next to it or derived from URL)."""
    p = image_path_or_url.strip().strip("'").strip('"')
    if p.lower().startswith(("http://", "https://")):
        parsed = urlparse(p)
        base = os.path.basename(parsed.path) or "image"
        name, _ = os.path.splitext(base)
        return f"{name}.json"
    name, _ = os.path.splitext(p)
    return f"{name}.json"


# ---------------------------
# Arguments
# ---------------------------

parser = ArgParser()

# Colors and features
parser.add_argument("--prompt-color", type=str, default="blue", help="termcolor name for user prompts")
parser.add_argument("--reply-color", type=str, default="green", help="termcolor name for model replies")
parser.add_argument("--enable-tools", action="store_true", help="allow tool/function calls")

# Streaming and stats
parser.add_argument("--disable-automatic-generation", action="store_false", dest="automatic_generation", help="wait for 'generate' command")
parser.add_argument("--disable-streaming", action="store_true", help="disable token streaming output")
parser.add_argument("--disable-stats", action="store_true", help="suppress generation performance stats")

# Save JSON by image toggles
parser.add_argument("--save-json-by-image", action="store_true",
                    help="After each bot reply, append JSON bound to the last image path/URL provided in chat. JSON filename is <image_path>.json")
parser.add_argument("--json-indent", type=int, default=2, help="Indentation for JSON (0 to minify)")

# HTTP server mode
parser.add_argument("--server", action="store_true",
                    help="Run as HTTP server that accepts image_path/question and triggers VILA, saving JSON per image like CLI.")
parser.add_argument("--port", type=int, default=8080, help="Port for --server mode (default: 8080)")

# NEW: forwarding of the per-image JSON to remote receiver
parser.add_argument(
    "--forward-url",
    type=str,
    default="http://172.16.17.11:5000/ingest",
    help="POST the per-image JSON document to this URL after saving (default: http://172.16.17.11:5000/ingest)"
)

args = parser.parse_args()

prompts = load_prompts(args.prompt)
interrupt = KeyboardInterrupt()
tool_response = None

# Track the most recent image the user provided (used for JSON filename)
last_image_path = None

# ---------------------------
# Load Model
# ---------------------------

model = NanoLLM.from_pretrained(
    args.model,
    api=args.api,
    quantization=args.quantization,
    max_context_len=args.max_context_len,
    vision_api=args.vision_api,
    vision_model=args.vision_model,
    vision_scaling=args.vision_scaling,
)

# ---------------------------
# Chat history
# ---------------------------

chat_history = ChatHistory(model, args.chat_template, args.system_prompt)

# ---------------------------
# Append & Forward
# ---------------------------

def _append_entry_to_json(
    json_path: str,
    image_path_or_url: str,
    model,
    prompt_text: str,
    reply_text: str,
    indent: int = 2
):
    """
    Append a single {timestamp, prompt, response} record into the per-image JSON file.
    Then forward the entire JSON document to args.forward_url (if set).
    """
    record = {
        "timestamp": int(time.time()),
        "prompt": prompt_text,
        "response": reply_text,
    }

    doc = None
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            doc = None

    if not isinstance(doc, dict):
        doc = {
            "image_path": image_path_or_url.strip().strip("'").strip('"'),
            "model": getattr(model, "repo_id", None) or getattr(model, "name", None),
            "api": getattr(model, "api", None),
            "entries": []
        }

    doc.setdefault("entries", [])
    doc["entries"].append(record)

    # Persist locally
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=indent)

    cprint(f"[saved] {json_path}", "cyan")

    # Forward to remote receiver if configured
    forward_url = (args.forward_url or "").strip()
    if forward_url:
        # Shallow copy so we don't persist transient fields locally
        out_doc = dict(doc)

        # Attach image bytes (if available)
        img_bytes, base_name, ext = _read_image_bytes(image_path_or_url)
        if img_bytes:
            out_doc["_image_basename"] = base_name
            out_doc["_image_ext"] = ext
            out_doc["_image_b64"] = base64.b64encode(img_bytes).decode("ascii")

        status, body = _http_post_json(forward_url, out_doc, timeout=10.0)
        if status in (200, 201):
            cprint(f"[forward] posted JSON+image to {forward_url} (status {status})", "cyan")
        else:
            cprint(f"[forward][warn] failed to POST to {forward_url} (status {status}): {body}", "yellow")



# ---------------------------
# Single "run cycle"
# ---------------------------

_run_lock = threading.Lock()

def process_user_prompt(user_prompt: str, *, generate: bool = True) -> str:
    """
    Execute one cycle:
    - Detect if prompt is an image path/URL and update last_image_path.
    - Append user prompt.
    - Optionally embed_chat + generate with the model.
    - Append bot reply.
    - Optionally save JSON bound to last_image_path.
    Returns the textual reply produced by the model (or "" if generate=False).
    """
    global last_image_path

    # Detect image path/URL
    if _is_image_path_or_url(user_prompt):
        last_image_path = user_prompt.strip().strip("'").strip('"')

    # Append user message into chat history
    chat_history.append("user", user_prompt)

    # If we only want to append (no generation), exit early
    if not generate:
        return ""

    # Embed step
    t0 = time.perf_counter()
    embedding, position = chat_history.embed_chat(
        max_tokens=model.config.max_length - args.max_new_tokens,
        wrap_tokens=args.wrap_tokens,
        use_cache=model.has_embed and chat_history.kv_cache,
    )
    t1 = time.perf_counter()
    print(f"[TICTOK] embed_chat: {(t1 - t0)*1000:.2f} ms")

    # Generate step (exception-safe)
    gen_start = time.perf_counter()
    try:
        reply = model.generate(
            embedding,
            streaming=not args.disable_streaming,
            kv_cache=chat_history.kv_cache,
            cache_position=position,
            stop_tokens=chat_history.template.stop,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    except Exception as e:
        cprint(f"[error] generate() failed: {e}", "red")
        chat_history.append("bot", f"[error] generation failed: {e}")
        return f"[error] generation failed: {e}"

    reply_text = ""
    if args.disable_streaming:
        # Non-streaming mode: reply is a single string
        reply_text = reply
        cprint(reply_text, args.reply_color)
        gen_end = time.perf_counter()
        print(f"[TICTOK] generate_total: {(gen_end - gen_start):.3f}s")
    else:
        # Streaming mode: reply yields tokens
        first_token_time = None
        token_count = 0
        for token in reply:
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
                print(f"[TICTOK] TTFT: {(first_token_time - gen_start)*1000:.2f} ms")
            cprint(token, args.reply_color, end="", flush=True)
            reply_text += token
            token_count += 1
            if interrupt:
                try:
                    reply.stop()
                except Exception:
                    pass
                interrupt.reset()
                break

        gen_end = time.perf_counter()
        total_time = gen_end - gen_start
        if token_count > 0:
            throughput = token_count / (gen_end - (first_token_time or gen_start))
            print(f"\n[TICTOK] generate_total: {total_time:.3f}s | tokens: {token_count} | throughput: {throughput:.2f} tok/s")

    print("")  # newline after generation

    if not args.disable_stats:
        print_table(model.stats)
        print("")

    # Append bot reply to chat history
    chat_history.append("bot", reply_text)

    # Save JSON per image if enabled (this also forwards the JSON)
    if args.save_json_by_image:
        if last_image_path:
            json_path = _json_path_for_image(last_image_path)
            _append_entry_to_json(
                json_path=json_path,
                image_path_or_url=last_image_path,
                model=model,
                prompt_text=user_prompt,
                reply_text=reply_text,
                indent=(None if args.json_indent == 0 else args.json_indent),
            )
        else:
            cprint("[warn] --save-json-by-image is enabled, but no image path/URL was provided yet.", "red")

    return reply_text


# ---------------------------
# HTTP Server mode (Flask)
# ---------------------------

if args.server:
    # Lazy import so Flask is only required in --server mode.
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route("/describe", methods=["POST"])
    def describe():
        """
        JSON in:
          {
            "image_path": "/data/images/01.jpg",   # required
            "question": "optional follow-up"       # optional
          }

        Behavior:
          - Hard reset of chat context for every request (prevents leakage).
          - Append the image (no generation).
          - Auto-inject "Describe the image" and generate.
          - If 'question' provided, ask it as a second turn and generate.
          - JSON-by-image saving remains intact via process_user_prompt
            (which also forwards the JSON to args.forward_url).
        """
        body = request.get_json(force=True, silent=False) or {}
        image_path = (body.get("image_path") or "").strip()
        question   = (body.get("question")   or "").strip()

        if not image_path:
            return jsonify({"error": "image_path is required"}), 400

        with _run_lock:
            # RESET between requests
            chat_history.reset()
            globals()['last_image_path'] = None

            # 1) add the image to history (no generation)
            process_user_prompt(image_path, generate=False)

            # 2) auto prompt
            auto_prompt = "Describe the image"
            resp_describe = process_user_prompt(auto_prompt, generate=True)

            # 3) optional follow-up
            resp_question = None
            if question:
                resp_question = process_user_prompt(question, generate=True)

        return jsonify({
            "ok": True,
            "image_path": image_path,
            "auto_prompt": auto_prompt,
            "response_describe": resp_describe,
            "response_question": resp_question
        })

    @app.get("/health")
    def health():
        """Simple health endpoint to verify server is up."""
        return jsonify({"ok": True, "time": int(time.time())})

    # Start the server and exit the CLI flow
    app.run(host="0.0.0.0", port=args.port)
    sys.exit(0)


# ---------------------------
# CLI mode (unchanged behavior)
# ---------------------------

while True:
    if chat_history.turn("user"):
        # Fetch next prompt
        if isinstance(prompts, list):
            if len(prompts) > 0:
                user_prompt = prompts.pop(0)
                cprint(f">> PROMPT: {user_prompt}", args.prompt_color)
            else:
                break
        else:
            cprint(">> PROMPT: ", args.prompt_color, end="", flush=True)
            user_prompt = sys.stdin.readline().strip()

        print("")

        # Load from file or reset if needed
        if user_prompt.lower().endswith((".txt", ".json")):
            user_prompt = " ".join(load_prompts(user_prompt))
        elif user_prompt.lower() in ("reset", "clear"):
            logging.info("resetting chat history")
            chat_history.reset()
            last_image_path = None
            continue

        # Process one cycle with the given prompt
        process_user_prompt(user_prompt)

    # Optional tool functions
    if args.enable_tools:
        tool_response = BotFunctions.run(
            chat_history.messages[-1]["content"] if chat_history.messages else "",
            template=chat_history.template
        )
        if tool_response:
            chat_history.append("tool_response", tool_response)
            cprint(tool_response, "yellow")
