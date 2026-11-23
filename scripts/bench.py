#!/usr/bin/env python3
"""Reproducible speed & parity benchmark for v100fast-qwen38-27b.

Measures short Chat, Code, and long-context (12K / 50K / 100K) prompt-read and decode tok/s
plus deterministic temp=0 output SHA-256 hashes and draft acceptance rate.

Usage:
  python3 scripts/bench.py --url http://127.0.0.1:8080
  python3 scripts/bench.py --url http://127.0.0.1:8080 --long-ctx 50000 100000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request


def post(url: str, path: str, body: dict, api_key: str = "", timeout: int = 1800) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url.rstrip("/") + path, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_prompt(url: str, api_key: str, label: str, messages: list[dict], max_tokens: int = 256,
               temperature: float = 0.0, tools: list | None = None, cache_prompt: bool = True) -> dict:
    body: dict = {
        "model": "qwen3.8-27b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": 42,
        "stream": False,
        "cache_prompt": cache_prompt,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        body["tools"] = tools
    t0 = time.perf_counter()
    resp = post(url, "/v1/chat/completions", body, api_key=api_key)
    wall_s = time.perf_counter() - t0
    msg = resp["choices"][0]["message"]
    content = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    t = resp.get("timings") or {}
    sha = hashlib.sha256(content.encode()).hexdigest()[:12]
    draft_n = t.get("draft_n") or 0
    draft_ok = t.get("draft_n_accepted") or 0
    acc = (draft_ok / draft_n) if draft_n else 0.0
    row = {
        "label": label,
        "prompt_n": t.get("prompt_n", 0),
        "prompt_tps": round(t.get("prompt_per_second", 0.0), 1),
        "gen_n": t.get("predicted_n", 0),
        "gen_tps": round(t.get("predicted_per_second", 0.0), 1),
        "accept": round(acc, 3),
        "draft": f"{draft_ok}/{draft_n}",
        "wall_s": round(wall_s, 2),
        "sha": sha,
    }
    print(
        f"  {label:16s} | pp={row['prompt_n']:6d} @ {row['prompt_tps']:7.1f} tok/s "
        f"| tg={row['gen_n']:3d} @ {row['gen_tps']:6.1f} tok/s | acc={row['accept']:.2f} ({row['draft']}) | sha={sha}",
        flush=True,
    )
    return row


def make_synthetic_codebase(url: str, api_key: str, target_tokens: int) -> str:
    chunk = (
        "// Module: kv_cache_allocator.cpp — deterministic memory pool & Volta sm_70 attention planner\n"
        "struct KvBlock { int32_t seq_id; int32_t pos; uint32_t slot; float scale; };\n"
        "static inline int64_t align_up(int64_t x, int64_t a) { return (x + a - 1) & ~(a - 1); }\n"
        "void plan_layer_buffers(int n_layers, int n_kv_heads, int head_dim, int max_ctx) {\n"
        "    const int64_t row_bytes = (head_dim / 32) * 34LL * n_kv_heads;\n"
        "    const int64_t layer_bytes = row_bytes * max_ctx * 2;\n"
        "    (void) align_up(layer_bytes * n_layers, 256);\n"
        "}\n\n"
    )
    # Repeat and binary-search to hit target_tokens accurately
    est_chars = max(4096, int(target_tokens * 4.5))
    text = chunk * ((est_chars // len(chunk)) + 2)
    lo, hi = 0, len(text)
    while hi - lo > 2000:
        mid = (lo + hi) // 2
        n = len(post(url, "/tokenize", {"content": text[:mid]}, api_key=api_key)["tokens"])
        lo, hi = (mid, hi) if n < target_tokens else (lo, mid)
    return text[:lo]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080", help="Server base URL")
    ap.add_argument("--api-key", default="", help="Optional Bearer API key")
    ap.add_argument("--long-ctx", type=int, nargs="*", default=[12000],
                    help="Long-context lengths to benchmark (e.g. --long-ctx 12000 50000 100000)")
    args = ap.parse_args()

    req = urllib.request.Request(args.url.rstrip("/") + "/health")
    with urllib.request.urlopen(req, timeout=10) as r:
        print("Health:", r.read().decode().strip())

    # Warmup (populates CUDA graphs)
    run_prompt(args.url, args.api_key, "warmup", [{"role": "user", "content": "Say ok."}], max_tokens=16)

    print("\n=== Short Chat & Code ===")
    run_prompt(args.url, args.api_key, "chat_explain",
               [{"role": "user", "content": "Explain speculative decoding and KV cache quantization in 6 concise bullet points."}], 256)
    run_prompt(args.url, args.api_key, "code_lis",
               [{"role": "user", "content": "Write a clean O(n log n) Python function for longest increasing subsequence with path reconstruction and trace [10,9,2,5,3,7,101,18]."}], 256)
    run_prompt(args.url, args.api_key, "code_spsc",
               [{"role": "user", "content": "Write a concise C++20 lock-free SPSC ring buffer using std::atomic with acquire/release memory ordering."}], 256)

    for ctx_n in args.long_ctx:
        print(f"\n=== Long Context ({ctx_n // 1000}K tokens, 2 turns) ===")
        blob = make_synthetic_codebase(args.url, args.api_key, ctx_n)
        msgs = [{"role": "user", "content": f"Analyze this code and write a C++ helper computing exact VRAM for 16 attention layers:\n\n{blob}"}]
        run_prompt(args.url, args.api_key, f"ctx_{ctx_n//1000}k_t1", msgs, 200, cache_prompt=True)
        msgs.append({"role": "assistant", "content": "Here is the VRAM calculation helper."})
        msgs.append({"role": "user", "content": "Now add a second helper estimating Q8_0 KV cache bytes per token for 4 KV heads of dim 256."})
        run_prompt(args.url, args.api_key, f"ctx_{ctx_n//1000}k_t2", msgs, 200, cache_prompt=True)


if __name__ == "__main__":
    main()
