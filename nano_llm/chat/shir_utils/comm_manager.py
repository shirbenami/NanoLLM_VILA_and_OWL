#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import urllib.request, urllib.error, json, time, argparse
from collections import deque

app = Flask(__name__)

JETSON2_ENDPOINT = None  # e.g., http://172.16.17.11:5050/prompts
HISTORY = deque(maxlen=200)
LAST = {
    "vila_caption": None,
    "jetson2_prompts": None,
    "last_forward_status": None,
}

FORWARD_TIMEOUT = None
FORWARD_RETRIES = None

def _http_post_json(url: str, payload: dict, timeout: float = 6.0):
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
        return -1, str(e)

@app.post("/from_vila")
def from_vila():
    caption = request.get_data(as_text=True, parse_form_data=False).strip()
    if not caption:
        return jsonify({"ok": False, "error": "empty caption"}), 400

    ts = int(time.time())
    print(f"[from_vila][{ts}] {caption}")
    LAST["vila_caption"] = {"ts": ts, "text": caption}
    HISTORY.appendleft({"src": "vila", "ts": ts, "text": caption})

    f_status, f_body, prompts = None, None, None
    if JETSON2_ENDPOINT:
        f_status, f_body = _http_post_json(JETSON2_ENDPOINT, {"sentence": caption}, timeout=10.0)
        print(f"[forward->jetson2] status={f_status} body={f_body[:180] if isinstance(f_body, str) else f_body}")

        try:
            data = json.loads(f_body) if isinstance(f_body, str) else {}
            if isinstance(data, dict) and isinstance(data.get("prompts"), list):
                prompts = [str(x) for x in data["prompts"]]
        except Exception:
            prompts = None

        LAST["last_forward_status"] = {"status": f_status, "body": f_body}
        if prompts:
            LAST["jetson2_prompts"] = {"ts": int(time.time()), "prompts": prompts}
            HISTORY.appendleft({"src": "jetson2", "ts": int(time.time()), "prompts": prompts})
            print(f"[jetson2][prompts] {prompts}")

    return jsonify({"ok": True, "forward_status": f_status, "prompts": prompts})

@app.get("/latest")
def latest():
    return jsonify({"ok": True, "last": LAST})

@app.get("/health")
def health():
    return jsonify({"ok": True, "time": int(time.time())})

def main():
    global JETSON2_ENDPOINT
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--jetson2-endpoint", required=True,
                    help="Full URL to Jetson-2 prompts endpoint, e.g. http://172.16.17.11:5050/prompts")

    ap.add_argument("--forward-timeout", type=float, default=20.0,
                    help="Timeout (seconds) for POST to Jetson-2 (default 20s)")
    ap.add_argument("--forward-retries", type=int, default=3,
                    help="Retries for POST to Jetson-2 on failure/timeout (default 3)")


    args = ap.parse_args()
    global JETSON2_ENDPOINT, FORWARD_TIMEOUT, FORWARD_RETRIES
    JETSON2_ENDPOINT = args.jetson2_endpoint.strip()
    FORWARD_TIMEOUT = args.forward_timeout
    FORWARD_RETRIES = args.forward_retries

    JETSON2_ENDPOINT = args.jetson2_endpoint.strip()


    print(f"[comm_manager] listening on {args.host}:{args.port}, jetson2_endpoint={JETSON2_ENDPOINT}")
    app.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()

