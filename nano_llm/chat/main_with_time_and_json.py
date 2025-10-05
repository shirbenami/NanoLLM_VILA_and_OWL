#!/usr/bin/env python3
import os
import sys
import time
import signal
import logging
import json
from urllib.parse import urlparse
from termcolor import cprint

import numpy as np

from nano_llm import NanoLLM, ChatHistory, ChatTemplates, BotFunctions
from nano_llm.utils import ImageExtensions, ArgParser, KeyboardInterrupt, load_prompts, print_table 

# =============== New helpers for JSON saving ===============

def _ext_of(path: str) -> str:
    """Return the lowercase extension (with leading '.') of a path/URL."""
    p = path.strip().strip("'").strip('"')
    if p.lower().startswith(("http://", "https://")):
        parsed = urlparse(p)
        _, ext = os.path.splitext(parsed.path)
        return ext.lower()
    _, ext = os.path.splitext(p)
    return ext.lower()

def _is_image_path_or_url(user_text: str) -> bool:
    """Detect if the input looks like an image path/URL by its extension."""
    if not user_text:
        return False
    ext = _ext_of(user_text)
    # ImageExtensions may be like [".jpg", ".png", ...] or ["jpg","png",...]
    # Normalize both cases:
    normalized = {e if e.startswith(".") else f".{e}" for e in (ImageExtensions if isinstance(ImageExtensions, (list, set, tuple)) else [])}
    if not normalized:
        # fallback if ImageExtensions is not iterable/available
        normalized = {".jpg",".jpeg",".png",".webp",".bmp",".tiff",".tif",".gif"}
    return ext in normalized

def _json_path_for_image(image_path_or_url: str) -> str:
    """
    Map an image path/URL to its JSON filename:
      - Local file: replace extension with .json (same directory)
      - URL: use URL basename (without query) and put <basename>.json in CWD
    """
    p = image_path_or_url.strip().strip("'").strip('"')
    if p.lower().startswith(("http://", "https://")):
        parsed = urlparse(p)
        base = os.path.basename(parsed.path) or "image"
        name, _ = os.path.splitext(base)
        return f"{name}.json"
    # local path → same directory
    name, _ = os.path.splitext(p)
    return f"{name}.json"

def _append_entry_to_json(json_path: str, image_path_or_url: str, model, prompt_text: str, reply_text: str, indent: int = 2):
    """
    Create/append an entry {timestamp, prompt, response} into a JSON file:
    {
      "image_path": ...,
      "model": ...,
      "api": ...,
      "entries": [ {timestamp, prompt, response}, ... ]
    }
    """
    record = {
        "timestamp": int(time.time()),
        "prompt": prompt_text,
        "response": reply_text,
    }

    # load existing
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

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=indent)

    cprint(f"[saved] {json_path}", "cyan")


# =============== Args ===============

# see utils/args.py for options
parser = ArgParser()

parser.add_argument("--prompt-color", type=str, default='blue', help="color to print user prompts (see https://github.com/termcolor/termcolor)")
parser.add_argument("--reply-color", type=str, default='green', help="color to print user prompts (see https://github.com/termcolor/termcolor)")
parser.add_argument("--enable-tools", action="store_true", help="enable the model to call tool functions")

parser.add_argument("--disable-automatic-generation", action="store_false", dest="automatic_generation", help="wait for 'generate' command before bot output")
parser.add_argument("--disable-streaming", action="store_true", help="wait to output entire reply instead of token by token")
parser.add_argument("--disable-stats", action="store_true", help="suppress the printing of generation performance stats")

# New: turn JSON saving on/off and control indentation
parser.add_argument("--save-json-by-image", action="store_true",
                    help="After each bot reply, save/append JSON bound to the last image path/URL provided in chat. The JSON filename is <image_path>.json")
parser.add_argument("--json-indent", type=int, default=2, help="Indentation for JSON output (0 to minify)")

args = parser.parse_args()

prompts = load_prompts(args.prompt)
interrupt = KeyboardInterrupt()
tool_response = None

# Track the most recent image the user provided
last_image_path = None

# =============== Model ===============

model = NanoLLM.from_pretrained(
    args.model, 
    api=args.api,
    quantization=args.quantization, 
    max_context_len=args.max_context_len,
    vision_api=args.vision_api,
    vision_model=args.vision_model,
    vision_scaling=args.vision_scaling, 
)

# =============== Chat History ===============

chat_history = ChatHistory(model, args.chat_template, args.system_prompt)

# =============== Main Loop ===============

while True: 
    if chat_history.turn('user'):
        # when it's the user's turn to prompt, get the next input
        if isinstance(prompts, list):
            if len(prompts) > 0:
                user_prompt = prompts.pop(0)
                cprint(f'>> PROMPT: {user_prompt}', args.prompt_color)
            else:
                break
        else:
            cprint('>> PROMPT: ', args.prompt_color, end='', flush=True)
            user_prompt = sys.stdin.readline().strip()

        print('')

        # special commands:  load prompts from file
        # 'reset' or 'clear' resets the chat history
        if user_prompt.lower().endswith(('.txt', '.json')):
            user_prompt = ' '.join(load_prompts(user_prompt))
        elif user_prompt.lower() in ('reset', 'clear'):
            logging.info("resetting chat history")
            chat_history.reset()
            last_image_path = None
            continue

        # Detect if the user just supplied an image path/URL
        if _is_image_path_or_url(user_prompt):
            last_image_path = user_prompt.strip().strip("'").strip('"')

        # add user prompt and embed
        chat_history.append('user', user_prompt)

    # ----------------------------
    # TICTOK:  embed_chat
    # ----------------------------
    t0 = time.perf_counter()
    embedding, position = chat_history.embed_chat(
        max_tokens=model.config.max_length - args.max_new_tokens,
        wrap_tokens=args.wrap_tokens,
        use_cache=model.has_embed and chat_history.kv_cache,
    )
    t1 = time.perf_counter()
    print(f"[TICTOK] embed_chat: {(t1 - t0)*1000:.2f} ms")

    # ----------------------------
    # TICTOK: generate
    # ----------------------------
    gen_start = time.perf_counter()
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

    # stream or print full output; always collect the final text
    reply_text = ""
    if args.disable_streaming:
        # non-streaming returns full string
        reply_text = reply
        cprint(reply_text, args.reply_color)
        gen_end = time.perf_counter()
        print(f"[TICTOK] generate_total: {(gen_end - gen_start):.3f}s")
    else:
        first_token_time = None
        token_count = 0
        for token in reply:
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
                print(f"[TICTOK] TTFT: {(first_token_time - gen_start)*1000:.2f} ms")
            cprint(token, args.reply_color, end='', flush=True)
            reply_text += token  # collect for JSON/logging
            token_count += 1
            if interrupt:
                # reply is a generator-like object with .stop()
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

    print('\n')
    
    if not args.disable_stats:
        print_table(model.stats)
        print('')
    
    # save the output and kv cache
    chat_history.append('bot', reply_text)

    # After each answer: save/append JSON bound to the last image seen
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
            cprint("[warn] --save-json-by-image is enabled, but no image path/URL has been provided yet in this session.", "red")

    # run tools
    if args.enable_tools:
        tool_response = BotFunctions.run(reply_text, template=chat_history.template)
        if tool_response:
            chat_history.append('tool_response', tool_response)
            cprint(tool_response, 'yellow')
