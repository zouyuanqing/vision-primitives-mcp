#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live verification: connect to vision-bridge MCP server over stdio (like Codex does)."""
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

CFG = Path.home() / ".codex" / "config.toml"
SCRIPT = Path(__file__).resolve().parent.parent / "vision_primitives_mcp.py"


def main():
    with open(CFG, "rb") as f:
        env = tomllib.load(f)["mcp_servers"]["vision-bridge"]["env"]
    proc_env = dict(os.environ)
    proc_env.update(env)
    proc_env["VISION_CACHE"] = "1"
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=proc_env,
    )

    def send(obj):
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        proc.stdin.write(f"Content-Length: {len(payload)}\r\n\r\n".encode())
        proc.stdin.write(payload)
        proc.stdin.flush()

    def recv():
        headers = {}
        while True:
            line = proc.stdout.readline()
            if line in (b"\r\n", b"\n"):
                break
            if b":" in line:
                k, v = line.split(b":", 1)
                headers[k.strip().lower()] = v.strip()
        n = int(headers.get(b"content-length", b"0"))
        return json.loads(proc.stdout.read(n).decode("utf-8"))

    send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "verify", "version": "1"}}})
    r = recv()
    print("initialize:", r["result"]["serverInfo"], "| protocol:", r["result"]["protocolVersion"])

    send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    r = recv()
    print("tools:", len(r["result"]["tools"]), "->", ", ".join(t["name"] for t in r["result"]["tools"]))

    send({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "vision_health", "arguments": {}}})
    r = recv()
    print("vision_health:", r["result"]["content"][0]["text"].replace("\n", " ")[:300])

    send({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "describe_image", "arguments": {"image": str(SCRIPT.parent / "sample.png"), "detail": "brief"}}})
    r = recv()
    text = r["result"]["content"][0]["text"]
    print("describe_image (brief):", text[:300].replace("\n", " "))
    print("isError:", r["result"]["isError"])

    proc.stdin.close()
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    print("LIVE VERIFICATION DONE")


if __name__ == "__main__":
    main()