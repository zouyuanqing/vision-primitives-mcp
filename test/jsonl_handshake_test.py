#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JSONL 协议握手测试：验证修改后的 vision_primitives_mcp.py 支持 Hana 风格的换行分隔 MCP 消息。"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "vision_primitives_mcp.py"

env = os.environ.copy()
env["VISION_API_KEY"] = env.get("VISION_API_KEY", "sk-test")
env["VISION_OUTPUT_DIR"] = str(ROOT / "generated")

proc = subprocess.Popen(
    [sys.executable, str(SERVER)],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    env=env,
    cwd=str(ROOT),
)


def _read_loop(out, q):
    try:
        while True:
            line = out.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            q.put(json.loads(line.decode("utf-8")))
    except Exception:
        pass


q = queue.Queue()
threading.Thread(target=_read_loop, args=(proc.stdout, q), daemon=True).start()


def send(obj):
    proc.stdin.write(json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n")
    proc.stdin.flush()


def read_msg(timeout=15):
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


t0 = time.time()
send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
      "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                 "clientInfo": {"name": "jsonl-test", "version": "0.1"}}})
resp = read_msg()
print(f"[{time.time()-t0:.1f}s] initialize ->", json.dumps(resp, ensure_ascii=False)[:300] if resp else "TIMEOUT/None")

ok = True
if resp and "result" in resp:
    pv = resp["result"].get("protocolVersion")
    print(f"  protocolVersion echoed: {pv} (expect 2025-11-25)")
    if pv != "2025-11-25":
        ok = False

    send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = read_msg()
    n = len(tools["result"]["tools"]) if tools and "result" in tools else -1
    print(f"[{time.time()-t0:.1f}s] tools/list -> {n} tools")
    if n != 21:
        ok = False

    send({"jsonrpc": "2.0", "id": 3, "method": "ping"})
    pong = read_msg()
    print(f"[{time.time()-t0:.1f}s] ping ->", json.dumps(pong, ensure_ascii=False)[:120] if pong else "TIMEOUT/None")
    if not pong or "result" not in pong:
        ok = False

    # 连续多帧混发（验证 JSONL 流式解析）
    send({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
          "params": {"name": "vision_health", "arguments": {}}})
    health = read_msg(40)
    if health and "result" in health:
        text = health["result"]["content"][0]["text"]
        print(f"[{time.time()-t0:.1f}s] vision_health ok, problems: {text[:200]}")
    else:
        print("vision_health ->", json.dumps(health, ensure_ascii=False)[:300] if health else "TIMEOUT/None")
        ok = False

print("\nRESULT:", "PASS" if ok else "FAIL")
proc.terminate()
try:
    proc.wait(timeout=5)
except Exception:
    proc.kill()
