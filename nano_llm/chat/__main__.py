#!/usr/bin/env python3
import os
import sys
import time
import json
import re
import logging
import traceback
from urllib.parse import urlparse
from termcolor import cprint

from nano_llm import NanoLLM, ChatHistory, BotFunctions
from nano_llm.utils import ArgParser, KeyboardInterrupt, load_prompts, print_table

# ===================== Helpers =====================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}

def _is_image_path_or_url(text: str) -> bool:
    if not text:
        return False
    s = text.strip().strip("'").strip('"')
    if s.lower().startswith(("http://", "https://")):
        _, ext = os.path.splitext(urlparse(s).path)
        return ext.lower() in IMAGE_EXTS
    _, ext = os.path.splitext(s)
    return ext.lower() in IMAGE_EXTS and os.path.exists(s)

def _json_path_for_image(image_path_or_url: str) -> str:
    s = image_path_or_url.strip().strip("'").strip('"')
    if s.lower().startswith(("http://", "https://")):
        base = os.path.basename(urlparse(s).path) or "image"
        name, _ = os.path.splitext(base)
        return f"{name}.json"
    name, _ = os.path.splitext(s)
    return f"{name}.json"

def _ensure_parent_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _clean_llm_text(t: str) -> str:
    if t is None:
        return ""
    s = str(t).replace("</s>", "").strip()
    # strip list-numbering at line starts like "1. ", "2) "
    s = "\n".join(re.sub(r"^\s*\d+[\.\)]\s*", "", ln) for ln in s.splitlines())
    return s.strip()

def _parse_owl_raw(owl_raw: str):
    """
    Expect one line like: [[a drone, a propeller, a floor]]
    Returns list of items (strings). On failure -> [].
    """
    if not owl_raw:
        return []
    s = _clean_llm_text(owl_raw)
    start = s.find("[[")
    end = s.find("]]", start + 2)
    if start == -1 or end == -1:
        return []
    inner = s[start + 2:end]
    items = [x.strip() for x in inner.split(",") if x.strip()]
    return items

def _append_json(json_path, img_path, model, prompt, reply, owl_raw=None, owl_list=None, indent=2):
    rec = {
        "timestamp": int(time.time()),
        "prompt": prompt,
        "response": reply
    }
    if owl_raw is not None:
        rec["owl_raw"] = owl_raw
    if owl_list is not None:
        rec["owl_list"] = owl_list

    data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

    if "entries" not in data or not isinstance(data.get("entries"), list):
        data = {
            "image_path": img_path,
            "model": getattr(model, "repo_id", None) or getattr(model, "name", None),
            "api": getattr(model, "api", None),
            "entries": []
        }

    data["entries"].append(rec)
    _ensure_parent_dir(json_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=(None if indent == 0 else indent))
    cprint(f"[saved] {json_path}", "cyan")

def _run_one_off(model, chat_template, system_prompt, user_text, max_new_tokens=96, temperature=0.0, top_p=1.0):
    """
    Run a short, isolated one-off prompt (no pollution to main history).
    """
    tmp = ChatHistory(model, chat_template, system_prompt)
    tmp.append("user", user_text)
    emb, pos = tmp.embed_chat()
    out = model.generate(
        emb,
        streaming=False,
        stop_tokens=tmp.template.stop,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    return out if isinstance(out, str) else str(out)

def _run_owl_from_image(model, chat_template, system_prompt, image_path_or_url: str, owl_template: str):
    """
    Build a temporary chat with:
      user: <image_path_or_url>
      user: <owl_template>
    Generate once (non-streaming). Return string.
    """
    tmp = ChatHistory(model, chat_template, system_prompt)
    tmp.append("user", image_path_or_url)     # attach image
    tmp.append("user", owl_template)          # ask for Nano-OWL list only
    emb, pos = tmp.embed_chat()
    out = model.generate(
        emb,
        streaming=False,
        stop_tokens=tmp.template.stop,
        max_new_tokens=64,
        temperature=0.0,
        top_p=1.0
    )
    return out if isinstance(out, str) else str(out)

# ===================== OWL Template =====================

OWL_FROM_IMAGE_TEMPLATE = (
    "find all the object in the image. Use singular nouns (1 word for each object). write a list of all the object"
)

# ===================== Args =====================

parser = ArgParser()

# colors & UX
parser.add_argument("--prompt-color", type=str, default="blue", help="termcolor for prompts")
parser.add_argument("--reply-color", type=str, default="green", help="termcolor for replies")

# behavior
parser.add_argument("--enable-tools", action="store_true", help="enable tool-calls post-reply")
parser.add_argument("--disable-streaming", action="store_true", help="print reply only when done (no token streaming)")
parser.add_argument("--disable-stats", action="store_true", help="hide performance stats table")

# JSON/OWL automation
parser.add_argument("--save-json-by-image", action="store_true",
                    help="After each reply, save/append JSON bound to the last image path/URL. JSON filename = <image>.json")
parser.add_argument("--json-indent", type=int, default=2, help="Indentation for JSON (0=minify)")
parser.add_argument("--owl-from-image", action="store_true",
                    help="When the input is an image path/URL, run a dedicated one-off to output ONLY Nano-OWL list (no description).")

args = parser.parse_args()

# ===================== Setup =====================

prompts = load_prompts(args.prompt)
interrupt = KeyboardInterrupt()
tool_response = None

model = NanoLLM.from_pretrained(
    args.model,
    api=args.api,
    quantization=args.quantization,
    max_context_len=args.max_context_len,
    vision_api=args.vision_api,
    vision_model=args.vision_model,
    vision_scaling=args.vision_scaling
)

chat = ChatHistory(model, args.chat_template, args.system_prompt)

last_image = None
last_input_was_image = False

# ===================== Main Loop =====================

while True:
    # input
    cprint(">> PROMPT: ", args.prompt_color, end="", flush=True)
    user_prompt = sys.stdin.readline().strip()

    # exit/reset
    if user_prompt.lower() in ("quit", "exit", "q"):
        break
    if user_prompt.lower() in ("reset", "clear"):
        logging.info("resetting chat")
        chat.reset()
        last_image = None
        last_input_was_image = False
        cprint("[info] chat reset.", "cyan")
        continue

    # classify input
    last_input_was_image = _is_image_path_or_url(user_prompt)
    if last_input_was_image:
        last_image = user_prompt.strip().strip("'").strip('"')
        cprint(f"[info] detected image: {last_image}", "cyan")

    # always keep transcript coherent: store user's message
    chat.append("user", user_prompt)

    # ===== Fast-path for image → OWL =====
    if args.owl_from_image and last_input_was_image:
        try:
            owl_out = _run_owl_from_image(
                model=model,
                chat_template=args.chat_template,
                system_prompt=args.system_prompt,
                image_path_or_url=last_image,
                owl_template=OWL_FROM_IMAGE_TEMPLATE
            )
            reply_text = _clean_llm_text(owl_out)
            cprint(reply_text, args.reply_color)
            chat.append("bot", reply_text)  # keep in transcript

            # parse / save JSON
            owl_raw = reply_text
            owl_list = _parse_owl_raw(owl_raw)
            cprint(f"[owl] {owl_raw}", "yellow")

            if args.save_json_by_image and last_image:
                json_path = _json_path_for_image(last_image)
                cprint(f"[info] saving json → {json_path}", "cyan")
                if not last_image.lower().startswith(("http://", "https://")):
                    _ensure_parent_dir(json_path)
                _append_json(
                    json_path=json_path,
                    img_path=last_image,
                    model=model,
                    prompt=user_prompt,
                    reply=reply_text,
                    owl_raw=owl_raw,
                    owl_list=owl_list,
                    indent=args.json_indent,
                )

            # optional stats/tools and continue (skip normal generate)
            if not args.disable_stats:
                print_table(model.stats)
                print("")
            if args.enable_tools:
                try:
                    tool_resp = BotFunctions.run(reply_text, template=chat.template)
                    if tool_resp:
                        chat.append("tool_response", tool_resp)
                        cprint(tool_resp, "yellow")
                except Exception as e:
                    cprint(f"[warn] tool execution failed: {e}", "red")

            last_input_was_image = False
            continue

        except Exception as e:
            cprint(f"[error] OWL one-off failed, falling back to regular generate: {e}", "red")
            traceback.print_exc()
            # fall through to normal generate below

    # ===== embed =====
    t0 = time.perf_counter()
    emb, pos = chat.embed_chat(
        max_tokens=model.config.max_length - args.max_new_tokens,
        wrap_tokens=args.wrap_tokens,
        use_cache=model.has_embed and chat.kv_cache,
    )
    t1 = time.perf_counter()
    print(f"[TICTOK] embed_chat: {(t1 - t0)*1000:.2f} ms")

    # ===== generate (regular text prompts) =====
    gen_start = time.perf_counter()
    reply_text = ""
    if args.disable_streaming:
        out = model.generate(
            emb,
            streaming=False,
            kv_cache=chat.kv_cache,
            cache_position=pos,
            stop_tokens=chat.template.stop,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        reply_text = _clean_llm_text(out if isinstance(out, str) else str(out))
        cprint(reply_text, args.reply_color)
        gen_end = time.perf_counter()
        print(f"[TICTOK] generate_total: {(gen_end - gen_start):.3f}s")
    else:
        first_tok = None
        tok_count = 0
        stream = model.generate(
            emb,
            streaming=True,
            kv_cache=chat.kv_cache,
            cache_position=pos,
            stop_tokens=chat.template.stop,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        for tok in stream:
            now = time.perf_counter()
            if first_tok is None:
                first_tok = now
                print(f"[TICTOK] TTFT: {(first_tok - gen_start)*1000:.2f} ms")
            s = str(tok)
            reply_text += s
            cprint(s, args.reply_color, end="", flush=True)
            tok_count += 1
            if interrupt:
                try:
                    stream.stop()
                except Exception:
                    pass
                interrupt.reset()
                break
        gen_end = time.perf_counter()
        total = gen_end - gen_start
        if tok_count > 0:
            tp = tok_count / (gen_end - (first_tok or gen_start))
            print(f"\n[TICTOK] generate_total: {total:.3f}s | tokens: {tok_count} | throughput: {tp:.2f} tok/s")

    print("")  # newline after reply

    # stats
    if not args.disable_stats:
        print_table(model.stats)
        print("")

    # persist in history
    reply_text = _clean_llm_text(reply_text)
    chat.append("bot", reply_text)

    # save JSON for non-image text prompts as well (into the last image's file)
    if args.save_json_by_image:
        try:
            if last_image:
                json_path = _json_path_for_image(last_image)
                cprint(f"[info] saving json → {json_path}", "cyan")
                if not last_image.lower().startswith(("http://", "https://")):
                    _ensure_parent_dir(json_path)
                _append_json(
                    json_path=json_path,
                    img_path=last_image,
                    model=model,
                    prompt=user_prompt,
                    reply=reply_text,
                    indent=args.json_indent,
                )
            else:
                cprint("[warn] --save-json-by-image is enabled, but no image path/URL was provided yet.", "red")
        except Exception as e:
            cprint(f"[error] failed to save JSON: {e}", "red")
            traceback.print_exc()

    # optional: tool-calls
    if args.enable_tools:
        try:
            tool_resp = BotFunctions.run(reply_text, template=chat.template)
            if tool_resp:
                chat.append("tool_response", tool_resp)
                cprint(tool_resp, "yellow")
        except Exception as e:
            cprint(f"[warn] tool execution failed: {e}", "red")

    # the image-trigger applies only on the exact turn of image input
    last_input_was_image = False
