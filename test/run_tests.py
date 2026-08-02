#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vision Bridge MCP mock 测试套件（不依赖真实 API key）。"""
import base64
import http.server
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="vb-test-"))
os.environ["VISION_CACHE_DIR"] = str(TMP / "cache")
os.environ["VISION_OUTPUT_DIR"] = str(TMP / "out")

# ---------- mock 视觉后端 ----------
class MockVisionHandler(http.server.BaseHTTPRequestHandler):
    requests = []
    responses = []
    models = [{"id": "mimo-v2.5", "object": "model", "owned_by": "xiaomi"}]
    vision_calls = 0

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._send(200, {"object": "list", "data": self.models})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).requests.append(body)
        type(self).vision_calls += 1
        if type(self).responses:
            resp = type(self).responses.pop(0)
            if isinstance(resp, Exception):
                code, payload = 500, {"error": {"message": str(resp)}}
            else:
                code, payload = 200, {
                    "id": "mock", "object": "chat.completion", "model": body.get("model"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": resp}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                }
        else:
            code, payload = 200, {
                "id": "mock", "object": "chat.completion", "model": body.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "mock 默认响应"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        self._send(code, payload)

def start_mock():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockVisionHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv

def reset_mock():
    MockVisionHandler.requests.clear()
    MockVisionHandler.responses.clear()
    MockVisionHandler.vision_calls = 0

# ---------- 小工具 ----------
PASS = []
FAIL = []

def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")

def make_img(w=200, h=100, color=(255, 0, 0)):
    from PIL import Image
    img = Image.new("RGB", (w, h), color)
    return img

def tmp_png(name, img):
    p = TMP / name
    img.save(p, "PNG")
    return str(p)

# ---------- 测试 ----------
def test_protocol_over_stdio(api_base):
    env = dict(os.environ)
    env.update({
        "VISION_API_BASE": api_base,
        "VISION_API_KEY": "test-key",
        "VISION_MODEL": "mimo-v2.5",
        "VISION_CACHE": "1",
        "VISION_TIMEOUT_S": "10",
    })
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "vision_primitives_mcp.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env,
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

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}})
        r = recv()
        check("initialize", r.get("result", {}).get("protocolVersion") == "2025-06-18" and r["result"]["serverInfo"]["name"] == "vision-primitives-mcp", str(r)[:200])

        send({"jsonrpc": "2.0", "id": 2, "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
        r = recv()
        names = [t["name"] for t in r["result"]["tools"]]
        check("tools/list 24 tools", names == ["describe_image", "analyze_image", "locate_object", "som_locate", "cursor_locate", "ocr_image", "annotate_image", "crop_image", "zoom_region", "vision_health", "annotate_infer", "screen_capture", "screen_info", "screen_click", "screen_move", "screen_drag", "screen_scroll", "screen_type", "screen_key", "cv_locate", "compare_infer", "reason_graph", "compare_images", "scan_anomalies"], str(names))

        reset_mock()
        MockVisionHandler.responses.append("这是一张测试图片。")
        send({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "describe_image", "arguments": {"image": str(ROOT / "sample.png")}}})
        r = recv()
        text = r["result"]["content"][0]["text"]
        check("tools/call describe", r["result"]["isError"] is False and "测试图片" in text, text[:120])

        send({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "describe_image", "arguments": {}}})
        r = recv()
        check("missing param -> -32602", r.get("error", {}).get("code") == -32602, str(r)[:200])

        send({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "no_such_tool", "arguments": {}}})
        r = recv()
        check("unknown tool -> -32601", r.get("error", {}).get("code") == -32601, str(r)[:200])

        send({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "describe_image", "arguments": {"image": str(TMP / "not-exist.png")}}})
        r = recv()
        check("missing file -> isError", r["result"]["isError"] is True and "文件不存在" in r["result"]["content"][0]["text"], str(r)[:200])
    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

def test_request_body(api_base):
    import vision_primitives_mcp as vb
    reset_mock()
    MockVisionHandler.responses.append("ok")
    img = make_img(200, 100)
    p = tmp_png("body.png", img)
    vb.tool_describe_image({"image": p, "question": "有什么？"})
    req = MockVisionHandler.requests[-1]
    check("req model", req["model"] == "mimo-v2.5", str(req.get("model")))
    check("req max_tokens 4096", req["max_tokens"] == 4096, str(req.get("max_tokens")))
    check("req temperature 0", req["temperature"] == 0, str(req.get("temperature")))
    msg = req["messages"][1]
    check("req image data url", msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))
    check("req prompt has size", "200px" in msg["content"][0]["text"] and "100px" in msg["content"][0]["text"], msg["content"][0]["text"][:100])

def test_cache_hit(api_base):
    import vision_primitives_mcp as vb
    reset_mock()
    MockVisionHandler.responses.append("第一次")
    p = tmp_png("cache.png", make_img(64, 64))
    vb.tool_describe_image({"image": p})
    calls_after_first = MockVisionHandler.vision_calls
    r1 = vb.tool_describe_image({"image": p})
    check("cache hit (no second call)", MockVisionHandler.vision_calls == calls_after_first, f"{calls_after_first} -> {MockVisionHandler.vision_calls}")
    check("cache content", r1 == "第一次", r1)

def test_analyze_primitives(api_base):
    import vision_primitives_mcp as vb
    reset_mock()
    MockVisionHandler.responses.append(json.dumps({
        "description": "有按钮",
        "visual_primitives": [
            {"id": "v1", "label": "按钮", "type": "box", "box": [20, 10, 180, 90], "confidence": 0.9},
            {"id": "v2", "label": "越界", "type": "box", "box": [1500, -5, 9999, 200], "confidence": 0.5},
        ],
    }))
    img = make_img(200, 100)
    p = tmp_png("analyze.png", img)
    res = vb.tool_analyze_image({"image": p})
    prims = res["visual_primitives"]
    check("analyze desc", res["description"] == "有按钮", str(res)[:200])
    check("analyze prim count", len(prims) == 2, str(prims))
    check("analyze box pixel", prims[0]["box_pixel"] == [20, 10, 180, 90], str(prims[0]))
    check("analyze box norm", prims[0]["box_norm"] == [100, 100, 900, 900], str(prims[0]))
    check("analyze clamp", prims[1]["box_pixel"] == [200, 0, 200, 100] and prims[1]["clamped"] is True, str(prims[1]))


def test_locate_refine(api_base):
    import vision_primitives_mcp as vb
    reset_mock()
    img = make_img(400, 300, (240, 240, 240))
    p = tmp_png("refine.png", img)
    coarse = json.dumps({"visual_primitives": [{"id": "v1", "label": "目标", "type": "box", "box": [100, 100, 200, 150], "confidence": 0.9}]})
    fine = json.dumps({"visual_primitives": [{"id": "v1", "label": "目标", "type": "box", "box": [40, 20, 120, 60], "confidence": 0.95}]})
    MockVisionHandler.responses.append(coarse)
    MockVisionHandler.responses.append(fine)
    res = vb.tool_locate_object({"image": p, "target": "目标", "refine": True})
    pr = res["primitives"][0]
    check("refine flag", pr.get("refined") is True, str(pr))
    check("refine box converted", pr["box_pixel"] == [100, 90, 140, 110], str(pr))
    check("refine two calls", MockVisionHandler.vision_calls == 2, str(MockVisionHandler.vision_calls))

def test_locate_coords(api_base):
    import vision_primitives_mcp as vb
    img = make_img(200, 100)
    p = tmp_png("locate.png", img)
    reset_mock()
    MockVisionHandler.responses.append(json.dumps({"visual_primitives": [{"id": "v1", "label": "目标", "type": "box", "box": [50, 40, 150, 60], "confidence": 0.8}]}))
    res = vb.tool_locate_object({"image": p, "target": "目标"})
    check("locate pixel", res["primitives"][0]["box"] == [50, 40, 150, 60] and res["coords"] == "pixel", str(res))
    reset_mock()
    MockVisionHandler.responses.append(json.dumps({"visual_primitives": [{"id": "v1", "label": "目标", "type": "box", "box": [50, 40, 150, 60], "confidence": 0.8}]}))
    res = vb.tool_locate_object({"image": p, "target": "目标", "coords": "norm"})
    check("locate norm", res["primitives"][0]["box"] == [250, 400, 750, 600], str(res))
    reset_mock()
    MockVisionHandler.responses.append(json.dumps({"visual_primitives": []}))
    res = vb.tool_locate_object({"image": p, "target": "没有的东西"})
    check("locate empty", res["count"] == 0 and "未找到" in res.get("note", ""), str(res))

def test_ocr(api_base):
    import vision_primitives_mcp as vb
    reset_mock()
    MockVisionHandler.responses.append(json.dumps([
        {"text": "提交", "box": [120, 180, 300, 240]},
        {"text": "Cancel", "box": [340, 180, 520, 240]},
        {"text": "bad"},
    ]))
    img = make_img(800, 500)
    p = tmp_png("ocr.png", img)
    res = vb.tool_ocr_image({"image": p})
    check("ocr count", res["count"] == 2, str(res))
    check("ocr box", res["items"][0]["box_pixel"] == [120, 180, 300, 240], str(res["items"]))

def test_annotate(api_base):
    import vision_primitives_mcp as vb
    from PIL import Image
    img = make_img(200, 100, (255, 255, 255))
    p = tmp_png("ann.png", img)
    out = Path(os.environ["VISION_OUTPUT_DIR"]) / "ann-out.png"
    res = vb.tool_annotate_image({"image": p, "items": [{"label": "目标", "box": [10, 10, 90, 60]}], "out_path": str(out)})
    check("annotate file exists", out.is_file(), str(res))
    check("annotate count", res["annotations"] == 1, str(res))
    im2 = Image.open(out)
    check("annotate size", im2.size == (200, 100), str(im2.size))
    px = im2.convert("RGB").getpixel((12, 12))
    check("annotate red line", px == (255, 59, 48), str(px))

def test_crop(api_base):
    import vision_primitives_mcp as vb
    img = make_img(200, 100)
    p = tmp_png("crop.png", img)
    out = Path(os.environ["VISION_OUTPUT_DIR"]) / "crop-out.png"
    res = vb.tool_crop_image({"image": p, "box": [10, 10, 110, 60], "out_path": str(out)})
    check("crop size", res["size"] == [100, 50], str(res))
    res2 = vb.tool_crop_image({"image": p, "box": [10, 10, 110, 60], "expand_px": 5, "out_path": str(Path(os.environ["VISION_OUTPUT_DIR"]) / "crop-out2.png")})
    check("crop expand", res2["size"] == [110, 60] and res2["box_used"] == [5, 5, 115, 65], str(res2))
    res3 = vb.tool_crop_image({"image": p, "box": [-50, -20, 100, 200]})
    check("crop clamp", res3["clamped"] is True and res3["box_used"] == [0, 0, 100, 100], str(res3))
    check("crop default path in out dir", str(Path(res3["path"]).parent) == str(Path(os.environ["VISION_OUTPUT_DIR"])), str(res3["path"]))

def test_zoom(api_base):
    import vision_primitives_mcp as vb
    img = make_img(200, 100)
    p = tmp_png("zoom.png", img)
    res = vb.tool_zoom_region({"image": p, "box": [0, 0, 100, 50], "scale": 2})
    check("zoom size", res["size"] == [200, 100] and res["scale"] == 2, str(res))
    res2 = vb.tool_zoom_region({"image": p, "scale": 3})
    check("zoom full", res2["size"] == [600, 300], str(res2))


def test_extract_json_robust():
    import vision_primitives_mcp as vb
    r1 = vb.extract_json('{"a":1} 多余文字')
    check("ej tail", r1 == {"a": 1}, str(r1))
    r2 = vb.extract_json('{"visual_primitives":[{"box":[200,100,300,900]}]}\n最终答案是 boxed')
    check("ej latex tail", r2["visual_primitives"][0]["box"] == [200, 100, 300, 900], str(r2))
    r3 = vb.extract_json('前缀文本 {"b": [1,2]} 尾巴')
    check("ej prefix", r3 == {"b": [1, 2]}, str(r3))
    r4 = vb.extract_json('```json\n{"c": 3}\n```')
    check("ej fence", r4 == {"c": 3}, str(r4))

def test_coord_utils():
    import vision_primitives_mcp as vb
    pts, clamped = vb.to_pixel([500, 500, 1000, 1000], 800, 500, "norm")
    check("norm->pixel", pts == [400, 250, 800, 500] and clamped is False, str(pts))
    pts, clamped = vb.to_pixel([-5, 0, 9999, 100], 800, 500, "pixel")
    check("clamp", pts == [0, 0, 800, 100] and clamped is True, str(pts))
    check("to_norm", vb.to_norm([400, 250, 800, 500], 800, 500) == [500, 500, 1000, 1000], str(vb.to_norm([400, 250, 800, 500], 800, 500)))

def test_validation(api_base):
    import vision_primitives_mcp as vb
    try:
        vb.tool_describe_image({"image": str(TMP / "nope.png")})
        check("validation missing file", False, "should raise")
    except vb.VisionError as e:
        check("validation missing file", "文件不存在" in str(e), str(e))
    big = TMP / "big.png"
    big.write_bytes(b"x" * (2 * 1024 * 1024))
    try:
        vb.tool_describe_image({"image": str(big)})
        check("validation big file", False, "should raise")
    except vb.VisionError as e:
        check("validation big file", "大小限制" in str(e), str(e))
    bad = TMP / "bad.txt"
    bad.write_text("hello")
    try:
        vb.tool_describe_image({"image": str(bad)})
        check("validation bad ext", False, "should raise")
    except vb.VisionError as e:
        check("validation bad ext", "不支持的图片格式" in str(e), str(e))
    try:
        vb.tool_crop_image({"image": str(tmp_png("v.png", make_img(10, 10))), "box": [0, 0, 5]})
        check("validation bad box", False, "should raise")
    except vb.VisionError as e:
        check("validation bad box", "box 必须是" in str(e), str(e))



def test_fit_model_norm():
    import vision_primitives_mcp as vb
    from PIL import Image
    img = Image.new("RGB", (800, 500))
    old_norm, old_max = vb.NORM_SIZE, vb.MAX_MODEL_SIDE
    try:
        vb.NORM_SIZE = 400
        mimg, sx, sy = vb._fit_model(img)
        check("norm size exact", mimg.size == (400, 250) and abs(sx - 2.0) < 0.01, f"{mimg.size} {sx}")
        vb.NORM_SIZE = 0
        mimg2, sx2, sy2 = vb._fit_model(img)
        check("norm off keeps small", mimg2.size == (800, 500) and sx2 == 1.0, f"{mimg2.size}")
    finally:
        vb.NORM_SIZE, vb.MAX_MODEL_SIDE = old_norm, old_max

def test_median_aggregation():
    import vision_primitives_mcp as vb
    # 两次采样：同一目标（不同 label/轻微坐标抖动）+ 一个独立目标
    b1 = [
        {"label": "按钮", "type": "box", "box_pixel": [100, 200, 300, 240], "confidence": 0.9},
        {"label": "圆形", "type": "box", "box_pixel": [500, 300, 600, 400], "confidence": 0.8},
    ]
    b2 = [
        {"label": "submit button", "type": "box", "box_pixel": [102, 198, 298, 242], "confidence": 0.85},
        {"label": "圆形", "type": "box", "box_pixel": [502, 302, 598, 398], "confidence": 0.75},
    ]
    out = vb.median_primitives([b1, b2])
    check("median cluster count", len(out) == 2, str(out))
    btn = next(p for p in out if "按钮" in p["label"] or "button" in p["label"])
    check("median button box", btn["box_pixel"] == [101, 199, 299, 241], str(btn))
    check("median label majority", btn["label"] == "按钮", str(btn))
    check("median confidence", btn["confidence"] == 0.875, str(btn))


def test_prompt_upgrade(api_base):
    import vision_primitives_mcp as vb
    reset_mock()
    MockVisionHandler.responses.append(json.dumps({"visual_primitives": []}))
    img = make_img(200, 100)
    p = tmp_png("prompt.png", img)
    vb.tool_locate_object({"image": p, "target": "歪斜元件"})
    req = MockVisionHandler.requests[-1]
    text = req["messages"][1]["content"][0]["text"]
    check("locate prompt full box", "完整边界框" in text, text[:200])
    check("locate prompt rotation", "rotation" in text, text[:200])
    check("locate prompt all targets", "所有可疑目标" in text, text[:200])

def test_normalize_rotation(api_base):
    import vision_primitives_mcp as vb
    prims = vb.normalize_primitives([{"label": "x", "type": "box", "box": [0, 0, 10, 10], "rotation": 12.5}], 100, 100)
    check("normalize keeps rotation", prims[0]["rotation"] == 12.5, str(prims))

def test_scan_anomalies(api_base):
    import vision_primitives_mcp as vb
    reset_mock()
    img = make_img(800, 500, (240, 240, 240))
    p = tmp_png("scan.png", img)
    locate_resp = json.dumps({"visual_primitives": [
        {"id": "v1", "label": "目标A", "type": "box", "box": [100, 100, 200, 160], "confidence": 0.9, "rotation": 15},
        {"id": "v2", "label": "目标B", "type": "box", "box": [300, 100, 400, 160], "confidence": 0.7, "rotation": 0},
    ]})
    MockVisionHandler.responses.append(locate_resp)
    MockVisionHandler.responses.append("存在明显歪斜元件，旋转约15度，丝印 A1142C，元件类型 LDO稳压器")
    MockVisionHandler.responses.append("没有明显歪斜元件")
    res = vb.tool_scan_anomalies({"image": p, "target": "歪斜元件", "region": [0, 0, 800, 500], "max_tiles": 1, "verify": True})
    check("scan candidates merged", res["candidates_merged"] == 2, str(res))
    c0 = res["candidates"][0]
    check("scan skew ranked first", c0["verified"]["verdict"] == "skewed", str(c0))
    check("scan rotation", c0["verified"]["rotation"] == 15, str(c0["verified"]))
    check("scan silkscreen", c0["verified"]["silkscreen"] == "A1142C", str(c0["verified"]))
    check("scan box orig", c0["box"] == [100, 100, 200, 160], str(c0["box"]))





def test_annotate_infer_v16(api_base):
    import vision_primitives_mcp as vb
    from PIL import Image
    reset_mock()
    MockVisionHandler.responses.append("ok")
    img = make_img(300, 200, (230, 230, 230))
    p = tmp_png("v16.png", img)
    # polygon / bubble 解析与文本
    aid, typ, lbl, color, geo = vb._parse_annot_item({"type": "polygon", "points": [[10, 10], [50, 10], [30, 60]]}, 300, 200)
    check("v16 polygon parse", typ == "polygon" and len(geo["points"]) == 3, str(geo))
    txt = vb._annot_to_text(aid, typ, "P1", color, geo)
    check("v16 polygon text", "多边形" in txt and "(10,10)" in txt, txt)
    aid, typ, lbl, color, geo = vb._parse_annot_item({"type": "bubble", "point": [100, 100], "text": "这是按钮", "direction": "up"}, 300, 200)
    check("v16 bubble parse", typ == "bubble" and geo["text"] == "这是按钮", str(geo))
    txt = vb._annot_to_text(aid, typ, "", color, geo)
    check("v16 bubble text", "气泡标注" in txt and "这是按钮" in txt, txt)
    # overlay 绘制 polygon + bubble + applied 返回
    reset_mock()
    MockVisionHandler.responses.append("ok2")
    items = [
        {"type": "polygon", "label": "P1", "points": [[20, 20], [120, 20], [70, 100]]},
        {"type": "bubble", "text": "目标区", "point": [250, 50], "direction": "auto"},
    ]
    res = vb.tool_annotate_infer({"image": p, "items": items, "question": "多边形与气泡指的区域是什么？", "mode": "overlay", "alpha": 0.5})
    check("v16 applied count", len(res["applied"]) == 2, str(res.get("applied")))
    check("v16 auto ids", res["applied"][0]["id"] == "a1" and res["applied"][1]["id"] == "a2", str(res.get("applied")))
    check("v16 overlay file", os.path.exists(res["overlay_path"]))
    ov = Image.open(res["overlay_path"]).convert("RGB")
    check("v16 polygon drawn", ov.getpixel((25, 25)) != (230, 230, 230), "polygon 未绘制")
    # corrections: move/add
    reset_mock()
    MockVisionHandler.responses.append("修正后 ok")
    items2 = [{"id": "b1", "type": "box", "box": [10, 10, 90, 60]}]
    corr = [
        {"op": "move", "id": "b1", "delta": [20, 30]},
        {"op": "add", "item": {"type": "point", "label": "新点", "point": [200, 100]}},
    ]
    res2 = vb.tool_annotate_infer({"image": p, "items": items2, "corrections": corr, "question": "修正后是什么"})
    check("v16 corrections flag", res2["corrections_applied"] is True, str(res2))
    b1 = next((a for a in res2["applied"] if a["id"] == "b1"), None)
    check("v16 move box", b1 is not None and b1["box"] == [30, 40, 110, 90], str(b1))
    check("v16 add item", any(a.get("label") == "新点" for a in res2["applied"]), str(res2.get("applied")))
    # corrections: remove
    res3 = vb.tool_annotate_infer({"image": p, "items": items2, "corrections": [{"op": "remove", "id": "b1"}], "question": "删掉后"})
    check("v16 remove", res3["applied"] == [], str(res3.get("applied")))
    # auto_boxes
    reset_mock()
    MockVisionHandler.responses.append(json.dumps({"visual_primitives": [{"id": "v1", "label": "按钮", "type": "box", "box": [50, 50, 150, 100], "confidence": 0.9}]}))
    MockVisionHandler.responses.append("自动框选 ok")
    res4 = vb.tool_annotate_infer({"image": p, "items": [{"type": "point", "point": [10, 10]}], "auto_boxes": ["按钮"], "question": "自动框了什么"})
    check("v16 auto_boxes", res4["auto_boxes_applied"] is not None and res4["auto_boxes_applied"][0]["box"] == [50, 50, 150, 100], str(res4.get("auto_boxes_applied")))
    check("v16 auto in applied", any(a.get("label") == "按钮" for a in res4["applied"]), str(res4.get("applied")))

def test_annotate_infer(api_base):
    import vision_primitives_mcp as vb
    from PIL import Image
    reset_mock()
    # virtual 模式
    MockVisionHandler.responses.append("框A中是蓝色按钮，A到B的连线表示信号连接关系。")
    img = make_img(200, 100, (240, 240, 240))
    p = tmp_png("ai.png", img)
    orig_bytes = open(p, "rb").read()
    items = [
        {"type": "box", "label": "A", "box": [10, 10, 90, 60], "color": "#ff3b30"},
        {"type": "arrow", "label": "连线1", "from": [90, 35], "to": [150, 35], "color": "#00b0f0"},
    ]
    res = vb.tool_annotate_infer({"image": p, "items": items, "question": "A 和 B 的关系？", "mode": "virtual"})
    check("ai virtual result", "关系" in res["answer"] and res["mode"] == "virtual", str(res)[:200])
    req = MockVisionHandler.requests[-1]
    text = req["messages"][1]["content"][0]["text"]
    check("ai virtual prompt has annot", "虚拟标注" in text and "A" in text and "箭头连线" in text, text[:300])
    check("ai virtual original untouched", open(p, "rb").read() == orig_bytes, "原图被修改!")
    # overlay 模式
    reset_mock()
    MockVisionHandler.responses.append("叠加层中框A区域有元件。")
    res2 = vb.tool_annotate_infer({"image": p, "items": items, "question": "框内是什么", "mode": "overlay", "alpha": 0.4})
    check("ai overlay result", res2["mode"] == "overlay" and os.path.exists(res2["overlay_path"]), str(res2)[:200])
    ov = Image.open(res2["overlay_path"]).convert("RGB")
    check("ai overlay size", ov.size == (200, 100), str(ov.size))
    check("ai overlay frame pixel", ov.getpixel((12, 12)) != (240, 240, 240), "框线未绘制")
    check("ai overlay original visible", ov.getpixel((195, 95)) == (240, 240, 240), "原图被遮挡")
    # 校验分支
    try:
        vb.tool_annotate_infer({"image": p, "items": items})
        check("ai missing question", False, "should raise")
    except vb.VisionError as e:
        check("ai missing question", "question" in str(e), str(e))
    try:
        vb.tool_annotate_infer({"image": p, "items": [{"type": "hexagon", "points": [[0, 0], [1, 1]]}], "question": "x"})
        check("ai bad type", False, "should raise")
    except vb.VisionError as e:
        check("ai bad type", "不支持的标注类型" in str(e), str(e))



def test_screen_use(api_base):
    import vision_primitives_mcp as vb
    # 安全开关默认关闭：所有控制类工具必须拒绝
    for tool, args in [
        ("screen_click", {"x": 10, "y": 10}),
        ("screen_move", {"x": 10, "y": 10}),
        ("screen_drag", {"x1": 0, "y1": 0, "x2": 10, "y2": 10}),
        ("screen_scroll", {"delta": 3}),
        ("screen_type", {"text": "hello"}),
        ("screen_key", {"key": "enter"}),
    ]:
        try:
            vb.HANDLERS[tool](args)
            check(f"screen guard {tool}", False, "should be blocked")
        except vb.VisionError as e:
            check(f"screen guard {tool}", "未启用" in str(e), str(e))
    # screen_info
    info = vb.tool_screen_info()
    check("screen info", isinstance(info.get("screen_size"), list) and info["control_allowed"] is False, str(info))
    # screen_capture 真截屏（Windows 有屏幕）
    try:
        cap = vb.tool_screen_capture({})
        check("screen capture file", os.path.exists(cap["path"]) and cap["size"][0] > 0, str(cap))
    except Exception as e:
        check("screen capture file", False, str(e)[:200])

def test_compare_infer(api_base):
    import vision_primitives_mcp as vb
    reset_mock()
    MockVisionHandler.responses.append("图1是输入级，图2是负载级，整体构成电源链路。")
    img = make_img(200, 100)
    p1 = tmp_png("ci1.png", img)
    p2 = tmp_png("ci2.png", img)
    res = vb.tool_compare_infer({
        "images": [p1, p2],
        "items_per_image": {0: [{"type": "box", "label": "输入", "box": [10, 10, 90, 60]}]},
        "question": "两张图是什么关系？",
    })
    check("ci answer", "电源" in res["answer"] and res["images"] == 2, str(res)[:200])
    req = MockVisionHandler.requests[-1]
    urls = [b for b in req["messages"][1]["content"] if b.get("type") == "image_url"]
    check("ci two images sent", len(urls) == 2, str(len(urls)))
    text = req["messages"][1]["content"][0]["text"]
    check("ci annot text", "【图1】" in text and "【图2】" in text and "输入" in text, text[:300])
    check("ci applied", len(res["applied_per_image"]) == 2 and len(res["applied_per_image"][0]) == 1, str(res.get("applied_per_image")))

def test_reason_graph(api_base):
    import vision_primitives_mcp as vb
    reset_mock()
    img = make_img(400, 300, (240, 240, 240))
    p = tmp_png("rg.png", img)
    # 1) locate step
    MockVisionHandler.responses.append(json.dumps({"visual_primitives": [{"id": "v1", "label": "芯片A", "type": "box", "box": [100, 100, 200, 150], "confidence": 0.9}]}))
    r1 = vb.tool_reason_graph({"image": p, "step": {"type": "locate", "target": "芯片A"}})
    check("rg locate", len(r1["session"]["primitives"]) == 1 and r1["session"]["primitives"][0]["geometry"] == [100, 100, 200, 150], str(r1))
    # 2) 再 locate 第二个 + measure
    MockVisionHandler.responses.append(json.dumps({"visual_primitives": [{"id": "v1", "label": "芯片B", "type": "box", "box": [300, 100, 380, 150], "confidence": 0.9}]}))
    r2 = vb.tool_reason_graph({"image": p, "session": r1["session"], "step": {"type": "locate", "target": "芯片B"}})
    check("rg locate2", len(r2["session"]["primitives"]) == 2, str(r2))
    r3 = vb.tool_reason_graph({"image": p, "session": r2["session"], "step": {"type": "measure", "measure": "distance", "refs": ["p1", "p2"]}})
    check("rg distance", r3["results"]["distance_px"] == 190.0, str(r3["results"]))
    r3b = vb.tool_reason_graph({"image": p, "session": r2["session"], "step": {"type": "measure", "measure": "angle", "refs": [[0, 0], [100, 100], [200, 100]]}})
    check("rg angle", r3b["results"]["angle_deg"] == 135.0, str(r3b["results"]))
    r3c = vb.tool_reason_graph({"image": p, "session": r2["session"], "step": {"type": "measure", "measure": "area", "refs": [[0, 0], [0, 100], [100, 100], [100, 0]]}})
    check("rg area", r3c["results"]["area_px2"] == 10000.0, str(r3c["results"]))
    # 3) semantic + hypothesis
    r4 = vb.tool_reason_graph({"image": p, "session": r2["session"], "step": {"type": "semantic", "text": "两芯片间距约190px"}})
    check("rg semantic", len(r4["session"]["semantics"]) == 1, str(r4))
    r5 = vb.tool_reason_graph({"image": p, "session": r4["session"], "step": {"type": "hypothesis", "text": "A和B通过SPI连接"}})
    check("rg hypothesis", r5["session"]["hypotheses"][0]["status"] == "proposed", str(r5))
    # 4) verify（走 annotate_infer mock）
    reset_mock()
    MockVisionHandler.responses.append("验证结果：A与B的走线符合SPI信号特征。")
    r6 = vb.tool_reason_graph({"image": p, "session": r5["session"], "step": {"type": "verify", "id": "h1"}})
    check("rg verify", "SPI" in r6["results"]["verdict"] and r6["session"]["hypotheses"][0]["status"] == "verified", str(r6)[:300])
    # 5) next
    reset_mock()
    MockVisionHandler.responses.append("下一步建议测量A与B的角度以确认连接方式。")
    r7 = vb.tool_reason_graph({"image": p, "session": r5["session"], "step": {"type": "next"}})
    check("rg next", "下一步" in r7["results"]["next_step"], str(r7)[:200])
    # 6) 校验
    try:
        vb.tool_reason_graph({"image": p, "step": {"type": "fly"}})
        check("rg bad step", False, "should raise")
    except vb.VisionError as e:
        check("rg bad step", "不支持的 step" in str(e), str(e))

def test_compare_images(api_base):
    import vision_primitives_mcp as vb
    reset_mock()
    MockVisionHandler.responses.append("图1和图2的主要差异：背景色不同，按钮位置不同。")
    img = make_img(200, 100)
    p1 = tmp_png("cmp1.png", img)
    p2 = tmp_png("cmp2.png", img)
    res = vb.tool_compare_images({"images": [p1, p2], "question": "有什么变化"})
    check("compare result", "差异" in res and "图1" in res, res)
    req = MockVisionHandler.requests[-1]
    check("compare two images", len(req["messages"][1]["content"]) == 3, str(req["messages"][1]["content"][:2]))
    check("compare prompt", "对比分析" in req["messages"][1]["content"][0]["text"], req["messages"][1]["content"][0]["text"][:80])
    # 缓存
    calls = MockVisionHandler.vision_calls
    res2 = vb.tool_compare_images({"images": [p1, p2], "question": "有什么变化"})
    check("compare cache", MockVisionHandler.vision_calls == calls and res2 == res, res2)
    # 参数校验
    try:
        vb.tool_compare_images({"images": [p1]})
        check("compare min images", False, "should raise")
    except vb.VisionError as e:
        check("compare min images", "2-4 张" in str(e), str(e))

def test_parse_verdict():
    import vision_primitives_mcp as vb
    v1 = vb._parse_verdict("1) 是\n2) 约30度\n3) 5C\n4) 电阻")
    check("verdict numbered format", v1["verdict"] == "skewed" and v1["rotation"] == 30.0 and v1["silkscreen"] == "5C", str(v1))
    v2 = vb._parse_verdict("没有明显歪斜元件。")
    check("verdict negative", v2["verdict"] == "not_skewed", str(v2))
    v3 = vb._parse_verdict("存在明显歪斜元件，旋转约15度，丝印 A1142C，元件类型 LDO稳压器")
    check("verdict positive", v3["verdict"] == "skewed" and v3["rotation"] == 15.0 and v3["silkscreen"] == "A1142C", str(v3))

def test_som_locate(api_base):
    import vision_primitives_mcp as vb
    reset_mock()
    # 全部轮次选编号（final=number，旧行为）
    MockVisionHandler.responses.append("5")
    MockVisionHandler.responses.append("1")
    img = make_img(300, 300, (240, 240, 240))
    p = tmp_png("som.png", img)
    res = vb.tool_som_locate({"image": p, "target": "红色圆形", "grid": [3, 3], "rounds": 2, "final": "number"})
    check("som count", res["count"] == 1, str(res))
    check("som path numbers", res["path_numbers"] == [5, 1], str(res))
    b = res["primitives"][0]["box_pixel"]
    check("som box in orig bounds", 0 <= b[0] < b[2] <= 300 and 0 <= b[1] < b[3] <= 300, str(b))


def test_som_locate_final_box(api_base):
    import vision_primitives_mcp as vb
    reset_mock()
    # 第一轮选 5 号格；末轮 box：局部图(260x260)上定位框 [50,50,150,150]
    MockVisionHandler.responses.append("5")
    MockVisionHandler.responses.append(json.dumps({"visual_primitives": [{"label": "x", "type": "box", "box": [50, 50, 150, 150]}]}))
    img = make_img(300, 300, (240, 240, 240))
    p = tmp_png("som_box.png", img)
    res = vb.tool_som_locate({"image": p, "target": "红色圆形", "grid": [3, 3], "rounds": 2, "final": "box"})
    check("som box mode count", res["count"] == 1, str(res))
    check("som box mode final", res["final_mode"] == "box", str(res))
    # 5 号格 [100,100,200,200] expand 0.15 -> cb(85,85,215,215) 130px -> 2x 260px
    # box [50,50,150,150] / scale 2 + origin(85,85) -> [110,110,160,160]
    b = res["primitives"][0]["box_pixel"]
    check("som box mapped to orig", b == [110, 110, 160, 160], str(b))
    check("som box path numbers", res["path_numbers"] == [5], str(res))


def test_som_locate_final_box_round1(api_base):
    import vision_primitives_mcp as vb
    reset_mock()
    # rounds=1 时末轮即第一轮：直接在原图输出框
    MockVisionHandler.responses.append(json.dumps({"visual_primitives": [{"label": "x", "type": "box", "box": [10, 20, 90, 80]}]}))
    img = make_img(200, 100, (240, 240, 240))
    p = tmp_png("som_box1.png", img)
    res = vb.tool_som_locate({"image": p, "target": "目标", "rounds": 1, "final": "box"})
    check("som box r1 count", res["count"] == 1, str(res))
    b = res["primitives"][0]["box_pixel"]
    check("som box r1 identity", b == [10, 20, 90, 80], str(b))


def test_som_locate_final_box_fallback(api_base):
    import vision_primitives_mcp as vb
    reset_mock()
    # 末轮 box 解析失败 -> 回退选编号
    MockVisionHandler.responses.append("5")
    MockVisionHandler.responses.append("非JSON内容")
    img = make_img(300, 300, (200, 205, 210))  # 不同内容避免命中前序测试缓存
    p = tmp_png("som_fb.png", img)
    res = vb.tool_som_locate({"image": p, "target": "红色圆形", "grid": [3, 3], "rounds": 2, "final": "box"})
    check("som fallback count", res["count"] == 1, str(res))
    check("som fallback keeps round1", res["path_numbers"] == [5], str(res))
    check("som fallback note", "回退" in (res.get("note") or ""), str(res.get("note")))


def test_som_locate_bad_number(api_base):
    import vision_primitives_mcp as vb
    reset_mock()
    MockVisionHandler.responses.append("99")  # 越界编号
    img = make_img(200, 200, (240, 240, 240))
    p = tmp_png("som_bad.png", img)
    res = vb.tool_som_locate({"image": p, "target": "目标", "grid": [2, 2], "rounds": 1, "final": "number"})
    check("som bad number -> count 0", res["count"] == 0, str(res))


def test_cursor_locate(api_base):
    import vision_primitives_mcp as vb
    reset_mock()
    # 第一步：向右下移动 30,20；第二步 done
    MockVisionHandler.responses.append(json.dumps({"dx": 30, "dy": 20}))
    MockVisionHandler.responses.append(json.dumps({"done": True}))
    img = make_img(400, 300, (240, 240, 240))
    p = tmp_png("cursor.png", img)
    res = vb.tool_cursor_locate({"image": p, "target": "红色圆形", "start": [0.5, 0.5], "max_steps": 4})
    check("cursor count", res["count"] == 1, str(res))
    check("cursor final pos", res["cursor"]["final"] == [230, 170], str(res["cursor"]))
    check("cursor done", res["cursor"]["done"] is True, str(res["cursor"]))
    check("cursor steps", res["cursor"]["steps_used"] == 2, str(res["cursor"]))


def test_cursor_clamp(api_base):
    import vision_primitives_mcp as vb
    reset_mock()
    # 一步移动超出边界 -> 步长钳制
    MockVisionHandler.responses.append(json.dumps({"dx": 9999, "dy": -9999}))
    MockVisionHandler.responses.append(json.dumps({"done": True}))
    img = make_img(100, 100, (240, 240, 240))
    p = tmp_png("cursor_clamp.png", img)
    res = vb.tool_cursor_locate({"image": p, "target": "x", "step_ratio": 0.2, "max_steps": 3})
    # 起始 (50,50)，step_ratio 0.2 -> max_dx=20 -> (70,30)
    check("cursor clamp", res["cursor"]["final"] == [70, 30], str(res["cursor"]))




def test_cv_locate_color(api_base):
    import vision_primitives_mcp as vb
    # 纯本地：不依赖 mock API，确定性测试
    img = make_img(300, 300, (240, 240, 240))
    from PIL import Image, ImageDraw
    d = ImageDraw.Draw(img)
    d.ellipse([100, 100, 200, 200], fill=(220, 50, 50))
    p = tmp_png("cv_red.png", img)
    r = vb.tool_cv_locate({"image": p, "target": "红色圆形", "color": "red", "coords": "pixel"})
    check("cv color count", r["count"] == 1, str(r))
    c = r["primitives"][0]["point_pixel"]
    check("cv color centroid", c == [150, 150], str(c))
    check("cv method", r["method"] == "color-cc", str(r))


def test_cv_locate_no_match(api_base):
    import vision_primitives_mcp as vb
    img = make_img(200, 200, (240, 240, 240))  # 无任何颜色目标
    p = tmp_png("cv_none.png", img)
    r = vb.tool_cv_locate({"image": p, "target": "x", "color": "red", "coords": "pixel"})
    check("cv no match count 0", r["count"] == 0, str(r))


def test_cv_locate_requires_hint(api_base):
    import vision_primitives_mcp as vb
    img = make_img(100, 100, (240, 240, 240))
    p = tmp_png("cv_hint.png", img)
    try:
        vb.tool_cv_locate({"image": p, "target": "x", "coords": "pixel"})
        check("cv requires color/template", False, "应抛出 VisionError")
    except vb.VisionError as e:
        check("cv requires color/template", "color" in str(e), str(e))


def test_som_final_cv(api_base):
    import vision_primitives_mcp as vb
    reset_mock()
    # 第一轮选 5 号格（中心），末轮 cv 颜色分割（本地，不消耗 mock 响应）
    MockVisionHandler.responses.append("5")
    img = make_img(300, 300, (240, 240, 240))
    from PIL import Image, ImageDraw
    d = ImageDraw.Draw(img)
    d.ellipse([120, 120, 180, 180], fill=(220, 50, 50))  # 红圆在中心格
    p = tmp_png("som_cv.png", img)
    r = vb.tool_som_locate({"image": p, "target": "红色圆形", "grid": [3, 3], "rounds": 2, "final": "cv", "color": "red"})
    check("som cv count", r["count"] == 1, str(r))
    b = r["primitives"][0]["box_pixel"]
    # 红圆中心 (150,150)，cv 质心应接近
    c = [(b[0] + b[2]) // 2, (b[1] + b[3]) // 2]
    check("som cv centroid", abs(c[0] - 150) <= 2 and abs(c[1] - 150) <= 2, str(c))
    check("som cv final_mode", r["final_mode"] == "cv", str(r))


def test_health(api_base):
    import vision_primitives_mcp as vb
    res = vb.tool_vision_health()
    check("health ok", res["ok"] is True and res["model"] == "mimo-v2.5", str(res))

def main():
    srv = start_mock()
    api_base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    os.environ["VISION_API_BASE"] = api_base
    os.environ["VISION_API_KEY"] = "test-key"
    os.environ["VISION_MODEL"] = "mimo-v2.5"
    os.environ["VISION_MAX_IMAGE_MB"] = "1"
    import vision_primitives_mcp as vb
    globals()["vb"] = vb
    print(f"mock server: {api_base}")
    print("== 协议（stdio 子进程） ==")
    test_protocol_over_stdio(api_base)
    print("== 请求体断言 ==")
    test_request_body(api_base)
    print("== 缓存 ==")
    test_cache_hit(api_base)
    print("== analyze/locate/ocr ==")
    test_analyze_primitives(api_base)
    test_locate_coords(api_base)
    test_locate_refine(api_base)
    test_ocr(api_base)
    print("== PIL 图像操作 ==")
    test_annotate(api_base)
    test_crop(api_base)
    test_zoom(api_base)
    print("== extract_json 稳健性 ==")
    test_extract_json_robust()
    print("== 坐标工具 ==")
    test_coord_utils()
    print("== 校验分支 ==")
    test_validation(api_base)
    print("== 归一化 ==")
    test_fit_model_norm()
    print("== 多取样聚合 ==")
    test_median_aggregation()
    print("== 工具增强 ==")
    test_prompt_upgrade(api_base)
    test_normalize_rotation(api_base)
    test_scan_anomalies(api_base)
    print("== annotate_infer v1.6 ==")
    test_annotate_infer_v16(api_base)
    print("== annotate_infer ==")
    test_annotate_infer(api_base)
    print("== computer use ==")
    test_screen_use(api_base)
    print("== compare_infer / reason_graph ==")
    test_compare_infer(api_base)
    test_reason_graph(api_base)
    print("== compare_images ==")
    test_compare_images(api_base)
    print("== verdict 解析 ==")
    test_parse_verdict()
    print("== SoM 编号定位 (v1.10) ==")
    test_som_locate(api_base)
    test_som_locate_final_box(api_base)
    test_som_locate_final_box_round1(api_base)
    test_som_locate_final_box_fallback(api_base)
    test_som_locate_bad_number(api_base)
    print("== Cursor 循环定位 (v1.10) ==")
    test_cursor_locate(api_base)
    test_cursor_clamp(api_base)
    print("== CV 精定位备选方案 (v1.10) ==")
    test_cv_locate_color(api_base)
    test_cv_locate_no_match(api_base)
    test_cv_locate_requires_hint(api_base)
    test_som_final_cv(api_base)
    print("== health ==")
    test_health(api_base)
    srv.shutdown()
    print(f"\n结果: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("失败项:", FAIL)
        sys.exit(1)
    print("全部通过 OK")

if __name__ == "__main__":
    main()