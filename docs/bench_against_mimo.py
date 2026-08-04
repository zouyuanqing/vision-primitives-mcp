#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark Wjl1224734792/visual-primitives-mcp against MiMo on sample.png.
Goal: run their visual_describe + visual_locate pipeline, extract the located
box for the blue submit button, and compare against pixel-level ground truth
from cv_locate (color-cc): box=[121,181,299,239], center=(210,210), 800x500.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SERVER_DIR = Path(r"C:\Users\Adfhj\Desktop\OH-WorkSpace\visual-primitives-mcp-wjl")
SAMPLE = Path(r"C:\Users\Adfhj\Documents\Codex\codex-vision-bridge\sample.png")
NODE = "node"

# Ground truth (pixel, from vision-bridge cv_locate color-cc, conf 0.97)
GT = {"box": [121, 181, 299, 239], "center": [210, 210], "size": [800, 500]}


def center(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class Client:
    def __init__(self, env_extra):
        env = dict(os.environ)
        env.update(env_extra)
        self.proc = subprocess.Popen(
            [NODE, str(SERVER_DIR / "dist" / "server.js")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, cwd=str(SERVER_DIR),
        )
        self.id = 0

    def _frame(self, obj):
        # MCP SDK v1: stdio uses newline-delimited JSON (JSON Lines)
        return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

    def send(self, obj):
        self.id += 1
        obj["id"] = self.id
        self.proc.stdin.write(self._frame(obj))
        self.proc.stdin.flush()

    def notify(self, method, params=None):
        self.proc.stdin.write(self._frame({"jsonrpc": "2.0", "method": method, "params": params or {}}))
        self.proc.stdin.flush()

    def recv(self, timeout=240):
        deadline = time.time() + timeout
        while True:
            if time.time() > deadline:
                raise TimeoutError("recv timeout")
            line = self.proc.stdout.readline()
            if not line:
                raise TimeoutError("EOF from server")
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
            except Exception:
                # 非 JSON 行：对方 stdout 日志污染，跳过（他们 pino 默认写 stdout）
                sys.stderr.write(f"[polluted-stdout] {line.decode('utf-8', 'replace')[:200]}\n")
                continue
            if "id" not in obj and "method" not in obj:
                # pino 日志也是 JSON 但无 id/method（有 level/time/msg），跳过
                continue
            return obj

    def call(self, method, params=None):
        self.send({"jsonrpc": "2.0", "method": method, "params": params or {}})
        return self.recv()

    def close(self):
        self.proc.kill()


def main():
    env_extra = {
        "VISION_API_BASE_URL": "https://api.xiaomimimo.com/v1",
        "VISION_API_KEY": os.environ.get("VISION_API_KEY", ""),
        "VISION_MODEL_NAME": "mimo-v2.5",
        "VISION_OCR_MODEL": "mimo-v2.5",
        "TIMEOUT_MS": "180000",
        "LOG_LEVEL": "warn",
    }
    if not env_extra["VISION_API_KEY"]:
        print("ERROR: set VISION_API_KEY env var (MiMo key)")
        sys.exit(1)

    c = Client(env_extra)
    r = c.call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                               "clientInfo": {"name": "bench", "version": "1"}})
    print("initialize:", r["result"]["serverInfo"], "| protocol:", r["result"]["protocolVersion"])
    c.notify("notifications/initialized")
    time.sleep(0.3)
    r = c.call("tools/list")
    tools = [t["name"] for t in r["result"]["tools"]]
    print("tools:", len(tools), "->", ", ".join(tools))

    # ---- Step 1: visual_describe ----
    t0 = time.time()
    r = c.call("tools/call", {"name": "visual_describe",
                              "arguments": {"image_path": str(SAMPLE), "task": "general"}})
    dt = time.time() - t0
    text = r["result"]["content"][0]["text"]
    print(f"\n[visual_describe] {dt:.1f}s, isError={r['result'].get('isError')}")
    print(text[:2000])
    desc = {}
    try:
        desc = json.loads(text)
    except Exception:
        pass
    session_id = desc.get("session_id", "")
    print("session_id:", session_id)

    # ---- Step 2: visual_locate ----
    t0 = time.time()
    r = c.call("tools/call", {"name": "visual_locate",
                              "arguments": {"question": "找到蓝色提交按钮的精确坐标，输出其 bbox",
                                            "session_id": session_id}})
    dt = time.time() - t0
    text = r["result"]["content"][0]["text"]
    print(f"\n[visual_locate] {dt:.1f}s, isError={r['result'].get('isError')}")
    print(text[:2500])
    try:
        loc = json.loads(text)
    except Exception:
        loc = {}

    # ---- Evaluate ----
    print("\n===== EVALUATION =====")
    print("ground truth box:", GT["box"], "center:", GT["center"], "(pixel, 800x500)")
    for obj in (loc.get("raw_visual_analysis") or {}).get("objects", []) or []:
        bbox = obj.get("bbox")
        if not bbox:
            continue
        # normalized 0-1000 -> pixel
        bx = [v / 1000 * GT["size"][0] if i % 2 == 0 else v / 1000 * GT["size"][1] for i, v in enumerate(bbox)]
        bc = center(bx)
        dc = dist(bc, GT["center"])
        print(f"object: label={obj.get('label')} bbox_norm={bbox} -> pixel~[{int(bx[0])},{int(bx[1])},{int(bx[2])},{int(bx[3])}]")
        print(f"  center_dist={dc:.1f}px  (their center {[round(v) for v in bc]} vs GT {GT['center']})")
    c.close()


if __name__ == "__main__":
    main()
