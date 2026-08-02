#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision Primitives MCP - 交互式视觉原语 MCP Server
让纯文本模型（如 deepseek-v4-flash）通过 MCP 工具获得"看图 + 视觉原语"能力：
  describe / analyze / locate(输出坐标) / OCR(带坐标) / annotate(圈画) / crop(裁切) / zoom(放大) / health
视觉后端：小米 MiMo V2.5（OpenAI 兼容 /v1/chat/completions，图片走 data URL）
依赖：Python 3.10+、Pillow（本机已装 12.2.0）。零第三方运行时依赖。
环境变量：
  VISION_API_BASE     默认 https://api.xiaomimimo.com/v1
  VISION_API_KEY      必填
  VISION_MODEL        默认 mimo-v2.5（grounding 要求高可切 mimo-v2.5-pro）
  VISION_MAX_TOKENS   默认 4096（MiMo 推理型耗 token）
  VISION_TIMEOUT_S    默认 120
  VISION_MAX_IMAGE_MB 默认 20
  VISION_CACHE        默认 1；设 0 关闭缓存
  VISION_SAMPLES      默认 1；>1 时 locate/analyze 多次取样取坐标中位数
  VISION_OUTPUT_DIR   生成图片输出目录（默认本目录 generated/）
  VISION_DEBUG        设 1 输出日志到 stderr
"""
import base64
import hashlib
import io
import json
import math
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:  # pragma: no cover
    HAS_PIL = False

# ----------------------------- 配置 -----------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

def _env(name, default=""):
    return os.environ.get(name, default).strip()

def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

API_BASE = _env("VISION_API_BASE", "https://api.xiaomimimo.com/v1").rstrip("/")
API_KEY = _env("VISION_API_KEY")
MODEL = _env("VISION_MODEL", "mimo-v2.5")
MAX_TOKENS = _env_int("VISION_MAX_TOKENS", 4096)
TIMEOUT_S = _env_int("VISION_TIMEOUT_S", 120)
MAX_IMAGE_BYTES = _env_int("VISION_MAX_IMAGE_MB", 20) * 1024 * 1024
CACHE_ENABLED = _env("VISION_CACHE", "1") != "0"
CACHE_MAX_ENTRIES = _env_int("VISION_CACHE_MAX", 256)
CACHE_TTL_S = _env_int("VISION_CACHE_TTL_S", 7 * 24 * 3600)
SAMPLES = max(1, _env_int("VISION_SAMPLES", 1))
DEBUG = _env("VISION_DEBUG", "0") == "1"
OUTPUT_DIR = Path(_env("VISION_OUTPUT_DIR", str(SCRIPT_DIR / "generated")))
CACHE_DIR = Path(_env("VISION_CACHE_DIR", str(SCRIPT_DIR / ".cache")))
CACHE_FILE = CACHE_DIR / "vision-cache.json"
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

AUX_VISION_SYSTEM = (
    "你是辅助视觉分析模型。请仔细分析用户提供的图像，并严格按要求的输出格式作答。"
    "不要提及隐藏的推理过程、内部工具，也不要提到会有另一个模型阅读你的回答。"
    "直接给出分析与结论。"
)

# ----------------------------- 基础工具 -----------------------------

class VisionError(Exception):
    """用户可见的工具错误。"""

class McpParamError(Exception):
    """参数校验错误 -> JSON-RPC -32602。"""

def log(*args):
    if DEBUG:
        print("[vision-bridge]", *args, file=sys.stderr, flush=True)

def load_image(src):
    """返回 (PIL.Image(RGB), 原始字节, 来源标签)。支持本地路径或 http(s) URL。"""
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"User-Agent": "vision-primitives-mcp/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
        except urllib.error.URLError as e:
            raise VisionError(f"无法下载图片: {e.reason}")
        if len(data) > MAX_IMAGE_BYTES:
            raise VisionError(f"图片超过大小限制（{MAX_IMAGE_BYTES // (1024*1024)}MB）")
        label = src
    else:
        p = Path(src).expanduser()
        if not p.is_file():
            raise VisionError(f"图片文件不存在: {p}")
        if p.suffix.lower() not in ALLOWED_EXTS:
            raise VisionError(f"不支持的图片格式: {p.suffix}（支持: {', '.join(sorted(ALLOWED_EXTS))}）")
        size = p.stat().st_size
        if size > MAX_IMAGE_BYTES:
            raise VisionError(f"图片 {size // (1024*1024)}MB 超过大小限制（{MAX_IMAGE_BYTES // (1024*1024)}MB）")
        data = p.read_bytes()
        label = str(p.resolve())
    try:
        # 支持超大图（如 200MP PCB 照片）；大小保护由 MAX_IMAGE_BYTES 负责
        Image.MAX_IMAGE_PIXELS = None
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        raise VisionError(f"无法解析图片: {e}")
    return img.convert("RGB"), data, label

def encode_png(img):
    """统一转 PNG data URL（兼容各种输入格式）。返回 (data_url, bytes)。"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii"), raw

def _clamp_list(pts, w, h):
    """钳制 2 或 4 元素坐标到图像边界。返回 (钳制后列表, 是否发生钳制)。"""
    n = len(pts)
    changed = False
    out = []
    for i, v in enumerate(pts):
        limit = w if i % 2 == 0 else h
        cv = max(0, min(int(round(v)), limit))
        if cv != v:
            changed = True
        out.append(cv)
    if n == 4:
        x1, y1, x2, y2 = out
        if x1 > x2:
            x1, x2, changed = x2, x1, True
        if y1 > y2:
            y1, y2, changed = y2, y1, True
        out = [x1, y1, x2, y2]
    return out, changed

def to_pixel(value, w, h, coords):
    """把 box(4)/point(2) 从 pixel 或 norm(0-1000) 转像素并钳制。返回 (list, clamped)。"""
    if not isinstance(value, (list, tuple)) or len(value) not in (2, 4):
        raise VisionError(f"坐标必须是长度为 2 或 4 的数组，收到: {value!r}")
    coords = (coords or "pixel").lower()
    if coords == "norm":
        pts = [v * (w / 1000.0 if i % 2 == 0 else h / 1000.0) for i, v in enumerate(value)]
    elif coords == "pixel":
        pts = list(value)
    else:
        raise VisionError(f"coords 必须是 'pixel' 或 'norm'，收到: {coords!r}")
    return _clamp_list(pts, w, h)

def to_norm(value, w, h):
    """像素坐标 -> 0-1000 归一化。"""
    return [round(v * 1000.0 / (w if i % 2 == 0 else h)) for i, v in enumerate(value)]

def clamp_box(box, w, h):
    return _clamp_list(box, w, h)

def extract_json(text):
    """从模型输出中稳健提取 JSON 对象或数组（容忍代码块、多余尾巴、LaTeX 等）。"""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    # 逐位置 raw_decode：处理"JSON + 尾巴"或"文本 + JSON"混合输出
    dec = json.JSONDecoder()
    for i in range(len(text)):
        ch = text[i]
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(text[i:])
                return obj
            except Exception:
                continue
    raise VisionError(f"视觉模型未返回有效 JSON: {text[:400]}")

def _font(size):
    for p in [
        "C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/PingFang.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()
# ----------------------------- 视觉后端调用 -----------------------------

def call_chat(messages, max_tokens=None, retries=1):
    """调用 OpenAI 兼容 chat/completions。返回文本 content。"""
    if not API_KEY:
        raise VisionError("未配置 VISION_API_KEY（视觉后端密钥）")
    if _env("VISION_NO_SYSTEM", "0") == "1":
        # 兼容本地小模型（如 minicpm-v-4_5）：system 消息可能导致空响应
        messages = [m for m in messages if m.get("role") != "system"]
    url = f"{API_BASE}/chat/completions"
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens or MAX_TOKENS,
        "temperature": 0,
        "stream": False,
    }
    if _env("VISION_DISABLE_THINKING", "0") == "1":
        # LM Studio 思考型视觉模型（如 minicpm-v-4_5 / qwen3.5）禁用思考，避免 content 为空
        body["thinking"] = False
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                resp = json.loads(r.read().decode("utf-8"))
            try:
                return resp["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError):
                raise VisionError(f"视觉后端响应异常: {json.dumps(resp, ensure_ascii=False)[:500]}")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "ignore")[:500]
            except Exception:
                pass
            if e.code == 429 or 500 <= e.code < 600:
                if attempt < retries:
                    time.sleep(2 * (attempt + 1))
                    continue
            raise VisionError(f"视觉后端 HTTP {e.code}: {detail}")
        except urllib.error.URLError as e:
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise VisionError(f"无法连接视觉后端（已重试 {retries} 次）: {e.reason}")
        except TimeoutError:
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise VisionError(f"视觉后端请求超时（已重试 {retries} 次）")
        except VisionError:
            raise
        except Exception as e:
            last = e
    raise VisionError(f"视觉后端请求失败: {last}")

def image_message(text, img):
    """构造带图片的 user 消息；统一转 PNG data URL。"""
    data_url, _ = encode_png(img)
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }

# ----------------------------- 缓存 -----------------------------

_lock = threading.Lock()

def cache_get(key):
    if not CACHE_ENABLED:
        return None
    try:
        with _lock:
            data = json.loads(CACHE_FILE.read_text("utf-8")) if CACHE_FILE.exists() else {}
        entry = data.get("entries", {}).get(key)
        if not entry:
            return None
        if time.time() - entry.get("ts", 0) > CACHE_TTL_S:
            return None
        return entry.get("value")
    except Exception:
        return None

def cache_set(key, value):
    if not CACHE_ENABLED:
        return
    try:
        with _lock:
            data = json.loads(CACHE_FILE.read_text("utf-8")) if CACHE_FILE.exists() else {}
            entries = data.setdefault("entries", {})
            entries[key] = {"ts": time.time(), "value": value}
            if len(entries) > CACHE_MAX_ENTRIES:
                for k in sorted(entries, key=lambda k: entries[k].get("ts", 0))[: len(entries) - CACHE_MAX_ENTRIES]:
                    del entries[k]
            CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    except Exception as e:
        log("cache write failed:", e)

def cache_key(img_bytes, tool, *parts):
    h = hashlib.sha256(img_bytes).hexdigest()
    return "|".join([h, tool, *[str(p) for p in parts], MODEL])

# ----------------------------- 视觉原语 -----------------------------

def normalize_primitives(raw_primitives, w, h, fmt="generic"):
    """把视觉模型输出的 primitives 规范化为统一结构（像素 + 归一化 + 钳制标记）。"""
    out = []
    for i, p in enumerate(raw_primitives or []):
        if not isinstance(p, dict):
            continue
        box = p.get("box") or p.get("bbox") or p.get("bbox_2d") or p.get("box_2d")
        point = p.get("point") or p.get("point_2d") or p.get("center")
        label = p.get("label") or p.get("ref") or p.get("text") or p.get("name") or f"item{i+1}"
        conf = p.get("confidence")
        entry = {
            "id": str(p.get("id") or f"v{i+1}"),
            "label": str(label).strip()[:96] or f"item{i+1}",
            "confidence": conf,
            "rotation": p.get("rotation") if p.get("rotation") is not None else p.get("angle"),
        }
        if fmt == "gemini" and isinstance(box, (list, tuple)) and len(box) == 4:
            ymin, xmin, ymax, xmax = box
            box = [xmin, ymin, xmax, ymax]
        if isinstance(box, (list, tuple)) and len(box) == 4:
            pts, clamped = to_pixel(box, w, h, "pixel")
            entry["type"] = "box"
            entry["box_pixel"] = pts
            entry["box_norm"] = to_norm(pts, w, h)
            entry["clamped"] = clamped
        elif isinstance(point, (list, tuple)) and len(point) == 2:
            pts, clamped = to_pixel(point, w, h, "pixel")
            entry["type"] = "point"
            entry["point_pixel"] = pts
            entry["point_norm"] = to_norm(pts, w, h)
            entry["clamped"] = clamped
        else:
            continue
        out.append(entry)
    return out

def median_primitives(batches):
    """Multi-sample aggregation: cluster by spatial center, median per cluster."""
    all_items = [p for batch in batches for p in batch]
    groups = []
    for p in all_items:
        c = _box_center(p)
        if c is None:
            groups.append([p])
            continue
        placed = False
        for g in groups:
            cg = _box_center(g[0])
            if cg is None:
                continue
            dx = abs(c[0] - cg[0])
            dy = abs(c[1] - cg[1])
            w = max(c[2], cg[2], 1.0)
            h = max(c[3], cg[3], 1.0)
            if dx <= max(40.0, w * 0.5) and dy <= max(40.0, h * 0.5):
                g.append(p)
                placed = True
                break
        if not placed:
            groups.append([p])
    from collections import Counter
    out = []
    for g in groups:
        base = dict(g[0])
        for field in ("box_pixel", "box_norm", "point_pixel", "point_norm", "confidence"):
            vals = [i.get(field) for i in g if i.get(field) is not None]
            if not vals:
                continue
            if isinstance(vals[0], list):
                base[field] = [round(statistics.median([v[j] for v in vals])) for j in range(len(vals[0]))]
            elif isinstance(vals[0], (int, float)):
                base[field] = round(statistics.median(vals), 3)
        labels = [i.get("label") for i in g if i.get("label")]
        if labels:
            base["label"] = Counter(labels).most_common(1)[0][0]
        out.append(base)
    return out

def _box_center(p):
    if p.get("box_pixel"):
        x1, y1, x2, y2 = p["box_pixel"]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, float(x2 - x1), float(y2 - y1))
    if p.get("point_pixel"):
        x, y = p["point_pixel"]
        return (float(x), float(y), 0.0, 0.0)
    return None
PRIMITIVE_PROMPT = (
    "请定位并输出 JSON（不要输出任何其他文字；可用 ```json 代码块包裹）:\n"
    '{"visual_primitives":[{"id":"v1","type":"box","label":"简短标签","box":[x1,y1,x2,y2],"confidence":0.9,"rotation":0}]}\n'
    "- box 为像素坐标 [x1,y1,x2,y2]（左上角、右下角），必须包含目标元件本体和全部引脚焊盘的完整边界框，图像实际宽 W px、高 H px，坐标必须在 0..W / 0..H 范围内。\n"
    "- 列出所有可疑目标（不只一个），每个目标一条。\n"
    "- rotation：若目标相对水平/垂直轴有旋转，估计旋转角度（度）；无旋转填 0。\n"
    "- 找不到目标时返回 {\"visual_primitives\":[]}。\n"
)
# ----------------------------- 工具实现 -----------------------------

def tool_describe_image(args):
    img, raw, label = load_image(args["image"])
    question = str(args.get("question") or "").strip()
    detail = str(args.get("detail") or "balanced").lower()
    if detail not in ("brief", "balanced", "detailed"):
        detail = "balanced"
    key = cache_key(raw, "describe", question, detail)
    hit = cache_get(key)
    if hit is not None:
        log("cache hit: describe")
        return hit
    prompt = f"请描述这张图像（宽 {img.width}px，高 {img.height}px）。"
    prompt += {"brief": "用 2-3 句话简要概括。", "balanced": "描述主要内容：布局、对象、文字、颜色。", "detailed": "尽可能详细地描述所有元素、文字内容、位置关系与颜色。"}[detail]
    if question:
        prompt += f"\n用户问题：{question}"
    text = call_chat([
        {"role": "system", "content": AUX_VISION_SYSTEM},
        image_message(prompt, img),
    ]).strip()
    cache_set(key, text)
    return text

def tool_analyze_image(args):
    img, raw, label = load_image(args["image"])
    question = str(args.get("question") or "").strip()
    fmt = str(args.get("format") or "generic").lower()
    if fmt not in ("generic", "gemini", "qwen"):
        raise VisionError(f"format 必须是 generic/gemini/qwen，收到: {fmt}")
    key = cache_key(raw, "analyze", question, fmt)
    hit = cache_get(key)
    if hit is not None:
        log("cache hit: analyze")
        return hit

    field = {"generic": "box", "gemini": "box_2d", "qwen": "bbox_2d"}[fmt]
    shape = {"generic": "box 为 [x1,y1,x2,y2] 像素坐标", "gemini": "box_2d 为 [ymin,xmin,ymax,xmax]，坐标 0-1000 归一化", "qwen": "bbox_2d 为 [x1,y1,x2,y2] 像素坐标"}[fmt]
    prompt = (
        f"分析这张图像（宽 {img.width}px，高 {img.height}px），输出 JSON（不要输出其他文字）:\n"
        '{"description":"对图像的整体描述","visual_primitives":[{"id":"v1","type":"box","label":"简短标签","%s":[0,0,0,0],"confidence":0.0}]}\n' % field
        + f"- {shape}。\n"
        + "- 列出重要对象/文字区域/按钮等，最多 12 个；没有框的就不输出。\n"
        + (f"- 用户关注点：{question}\n" if question else "")
        + "- confidence 为 0-1 的置信度；若目标有旋转，请添加 rotation 字段（估计角度，度）。"
    )
    batches = []
    for _ in range(SAMPLES):
        text = call_chat([
            {"role": "system", "content": AUX_VISION_SYSTEM},
            image_message(prompt, img),
        ])
        obj = extract_json(text)
        batches.append(normalize_primitives(obj.get("visual_primitives"), img.width, img.height, fmt))
    prims = median_primitives(batches) if SAMPLES > 1 else batches[0]
    result = {
        "description": str(obj.get("description") or "") if SAMPLES == 1 else "",
        "visual_primitives": prims,
        "image_size": [img.width, img.height],
    }
    if SAMPLES > 1:
        result["samples"] = SAMPLES
    cache_set(key, result)
    return result

def tool_locate_object(args):
    img, raw, label = load_image(args["image"])
    target = str(args.get("target") or "").strip()
    if not target:
        raise VisionError("缺少参数: target")
    coords = str(args.get("coords") or "pixel").lower()
    if coords not in ("pixel", "norm"):
        raise VisionError(f"coords 必须是 'pixel' 或 'norm'，收到: {coords}")
    refine = args.get("refine") in (True, "true", "1", 1)
    key = cache_key(raw, "locate", target, coords, "refine" if refine else "")
    hit = cache_get(key)
    if hit is not None:
        log("cache hit: locate")
        return hit
    prompt = f"在图像（宽 {img.width}px，高 {img.height}px）中定位目标：{target}\n" + PRIMITIVE_PROMPT
    batches = []
    for _ in range(SAMPLES):
        text = call_chat([
            {"role": "system", "content": AUX_VISION_SYSTEM},
            image_message(prompt, img),
        ])
        obj = extract_json(text)
        batches.append(normalize_primitives(obj.get("visual_primitives"), img.width, img.height, "generic"))
    prims = median_primitives(batches) if SAMPLES > 1 else batches[0]
    if refine and prims:
        # 两阶段精修：粗框 -> 裁切放大 -> 二次定位 -> 换算回原图
        tmp_dir = CACHE_DIR / "scan_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        refined = []
        for p in prims:
            if not p.get("box_pixel"):
                refined.append(p)
                continue
            b = p["box_pixel"]
            pad = max(20, int(min(b[2] - b[0], b[3] - b[1]) * 0.35))
            cb = (max(0, b[0] - pad), max(0, b[1] - pad), min(img.width, b[2] + pad), min(img.height, b[3] + pad))
            if cb[2] - cb[0] < 40 or cb[3] - cb[1] < 40:
                refined.append(p)
                continue
            crop = img.crop(cb)
            crop2 = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
            tp = tmp_dir / f"refine_{os.getpid()}_{len(refined)}.png"
            crop2.save(tp, "PNG")
            try:
                loc2 = tool_locate_object({"image": str(tp), "target": target, "coords": "pixel"})
            finally:
                try:
                    tp.unlink()
                except OSError:
                    pass
            if loc2.get("primitives") and loc2["primitives"][0].get("box"):
                bx = loc2["primitives"][0]["box"]
                # 换算回原图坐标（裁切图被 2x 放大）
                rb = [cb[0] + bx[0] // 2, cb[1] + bx[1] // 2, cb[0] + bx[2] // 2, cb[1] + bx[3] // 2]
                rb, _ = _clamp_list(rb, img.width, img.height)
                if rb[2] - rb[0] >= 2 and rb[3] - rb[1] >= 2:
                    p = dict(p)
                    p["box_pixel"] = rb
                    p["box_norm"] = to_norm(rb, img.width, img.height)
                    p["refined"] = True
            refined.append(p)
        prims = refined
    for p in prims:
        if coords == "norm":
            p["box"] = p.get("box_norm")
            p["point"] = p.get("point_norm")
        else:
            p["box"] = p.get("box_pixel")
            p["point"] = p.get("point_pixel")
    result = {
        "target": target,
        "count": len(prims),
        "primitives": prims,
        "image_size": [img.width, img.height],
        "coords": coords,
    }
    if not prims:
        result["note"] = "视觉模型未找到该目标，请核对描述或换一种说法重试"
    cache_set(key, result)
    return result

def tool_ocr_image(args):
    img, raw, label = load_image(args["image"])
    language = str(args.get("language") or "auto")
    key = cache_key(raw, "ocr", language)
    hit = cache_get(key)
    if hit is not None:
        log("cache hit: ocr")
        return hit
    prompt = (
        f"请对这张图像（宽 {img.width}px，高 {img.height}px）做 OCR：提取图像中所有文字块，输出 JSON 数组（不要输出其他文字）:\n"
        '[{"text":"文字内容","box":[x1,y1,x2,y2]}]'
        f"\n- box 为像素坐标 [x1,y1,x2,y2]，图像宽 {img.width}px 高 {img.height}px。\n"
        "- 每行/每个独立文本块一条；没有文字返回 []。"
        + (f"\n- 语言提示：{language}" if language != "auto" else "")
    )
    text = call_chat([
        {"role": "system", "content": AUX_VISION_SYSTEM},
        image_message(prompt, img),
    ])
    rows = extract_json(text)
    if not isinstance(rows, list):
        raise VisionError(f"OCR 结果格式异常: {str(rows)[:300]}")
    items = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        box = r.get("box") or r.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        pts, clamped = to_pixel(box, img.width, img.height, "pixel")
        items.append({
            "text": str(r.get("text") or "").strip(),
            "box_pixel": pts,
            "box_norm": to_norm(pts, img.width, img.height),
            "confidence": r.get("confidence"),
            "clamped": clamped,
        })
    result = {"count": len(items), "items": items, "image_size": [img.width, img.height]}
    cache_set(key, result)
    return result

def _resolve_out_path(out_path):
    if not out_path:
        return None
    p = Path(out_path).expanduser()
    try:
        p = p.resolve()
    except Exception:
        p = Path(str(p))
    out_root = OUTPUT_DIR.resolve()
    if not (p == out_root or out_root in p.parents):
        raise VisionError(f"out_path 必须位于输出目录内: {OUTPUT_DIR}")
    return p

def _unique_path(prefix):
    ts = time.strftime("%Y%m%d-%H%M%S")
    return OUTPUT_DIR / f"{prefix}_{ts}_{os.getpid()}.png"

def tool_annotate_image(args):
    img, raw, label = load_image(args["image"])
    items = args.get("items")
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list) or not items:
        raise VisionError("items 必须是标注项数组（或单个对象）")
    coords = str(args.get("coords") or "pixel").lower()
    if coords not in ("pixel", "norm"):
        raise VisionError(f"coords 必须是 'pixel' 或 'norm'，收到: {coords}")
    style = args.get("style") or {}
    if not isinstance(style, dict):
        raise VisionError("style 必须是对象")
    out_path = _resolve_out_path(args.get("out_path")) or _unique_path("annotate")
    draw = ImageDraw.Draw(img)
    lw = max(1, int(style.get("line_width") or 3))
    fs = int(style.get("font_size") or max(14, img.width // 50))
    default_color = str(style.get("color") or "#ff3b30")
    fnt = _font(fs)
    clamped_any = False
    count = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        color = str(item.get("color") or default_color)
        box = item.get("box")
        point = item.get("point")
        label_text = str(item.get("label") or "").strip()
        if box is not None:
            pts, clamped = to_pixel(box, img.width, img.height, coords)
            clamped_any = clamped_any or clamped
            x1, y1, x2, y2 = pts
            draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
            if label_text:
                tb = draw.textbbox((0, 0), label_text, font=fnt)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                ly = max(0, y1 - th - 6)
                draw.rectangle([x1, ly, x1 + tw + 8, ly + th + 6], fill=color)
                draw.text((x1 + 4, ly + 2), label_text, fill="white", font=fnt)
            count += 1
        elif point is not None:
            pts, clamped = to_pixel(point, img.width, img.height, coords)
            clamped_any = clamped_any or clamped
            px, py = pts
            r = max(6, lw + 3)
            draw.ellipse([px - r, py - r, px + r, py + r], fill=color)
            if label_text:
                tb = draw.textbbox((0, 0), label_text, font=fnt)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                draw.rectangle([px + r + 2, py - th // 2 - 3, px + r + 2 + tw + 8, py + th // 2 + 3], fill=color)
                draw.text((px + r + 6, py - th // 2 - 1), label_text, fill="white", font=fnt)
            count += 1
    img.save(out_path, "PNG")
    return {
        "path": str(out_path),
        "image_size": [img.width, img.height],
        "annotations": count,
        "clamped": clamped_any,
        "note": "clamped=true 表示部分坐标超出图像边界已被自动钳制" if clamped_any else "",
    }

def tool_crop_image(args):
    img, raw, label = load_image(args["image"])
    box = args.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise VisionError("box 必须是 [x1,y1,x2,y2]（长度 4 的数组）")
    coords = str(args.get("coords") or "pixel").lower()
    if coords not in ("pixel", "norm"):
        raise VisionError(f"coords 必须是 'pixel' 或 'norm'，收到: {coords}")
    expand = int(args.get("expand_px") or 0)
    if expand < 0:
        raise VisionError("expand_px 不能为负数")
    out_path = _resolve_out_path(args.get("out_path")) or _unique_path("crop")
    pts, clamped = to_pixel(box, img.width, img.height, coords)
    if expand:
        x1, y1, x2, y2 = pts
        pts = [x1 - expand, y1 - expand, x2 + expand, y2 + expand]
        pts, clamped2 = _clamp_list(pts, img.width, img.height)
        clamped = clamped or clamped2
    x1, y1, x2, y2 = pts
    if x2 - x1 < 1 or y2 - y1 < 1:
        raise VisionError(f"裁切区域为空或过小: {pts}")
    region = img.crop((x1, y1, x2, y2))
    region.save(out_path, "PNG")
    return {
        "path": str(out_path),
        "box_used": pts,
        "size": [region.width, region.height],
        "expand_px": expand,
        "clamped": clamped,
    }

def tool_zoom_region(args):
    img, raw, label = load_image(args["image"])
    coords = str(args.get("coords") or "pixel").lower()
    if coords not in ("pixel", "norm"):
        raise VisionError(f"coords 必须是 'pixel' 或 'norm'，收到: {coords}")
    scale = int(args.get("scale") or 2)
    if not 1 <= scale <= 8:
        raise VisionError("scale 必须是 1-8 的整数")
    out_path = _resolve_out_path(args.get("out_path")) or _unique_path("zoom")
    box = args.get("box")
    if box is None:
        region = img
        pts = [0, 0, img.width, img.height]
        clamped = False
    else:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise VisionError("box 必须是 [x1,y1,x2,y2]（长度 4 的数组）")
        pts, clamped = to_pixel(box, img.width, img.height, coords)
        x1, y1, x2, y2 = pts
        if x2 - x1 < 1 or y2 - y1 < 1:
            raise VisionError(f"放大区域为空或过小: {pts}")
        region = img.crop((x1, y1, x2, y2))
    region = region.resize((region.width * scale, region.height * scale), Image.LANCZOS)
    region.save(out_path, "PNG")
    return {
        "path": str(out_path),
        "box_used": pts,
        "scale": scale,
        "size": [region.width, region.height],
        "clamped": clamped,
    }

def tool_vision_health(args=None):
    problems = []
    if not API_KEY:
        problems.append("VISION_API_KEY 未配置")
    if not API_BASE:
        problems.append("VISION_API_BASE 未配置")
    if not HAS_PIL:
        problems.append("Pillow 未安装（无法执行 annotate/crop/zoom）")
    backend = None
    try:
        req = urllib.request.Request(f"{API_BASE}/models", headers={"Authorization": f"Bearer {API_KEY}"})
        with urllib.request.urlopen(req, timeout=20) as r:
            backend = r.status
            data = json.loads(r.read().decode("utf-8"))
        ids = [m.get("id") for m in data.get("data", [])] if isinstance(data, dict) else []
        if MODEL not in ids:
            problems.append(f"模型 {MODEL} 不在后端模型列表: {', '.join(ids)}")
    except Exception as e:
        problems.append(f"后端连通性探测失败: {e}")
        backend = None
    return {
        "ok": not problems,
        "api_base": API_BASE,
        "model": MODEL,
        "backend_status": backend,
        "problems": problems,
        "pillow": HAS_PIL,
        "cache_enabled": CACHE_ENABLED,
        "samples": SAMPLES,
        "output_dir": str(OUTPUT_DIR),
        "max_image_mb": MAX_IMAGE_BYTES // (1024 * 1024),
    }






# ----------------------------- 虚拟标注推理（annotate_infer） -----------------------------

ANNOT_TYPES = ("box", "point", "line", "arrow", "circle", "polygon", "bubble")

def _parse_annot_item(item, w, h):
    """解析标注项 -> (id, type, label, color, geometry_pixel)。"""
    if not isinstance(item, dict):
        raise VisionError("标注项必须是对象")
    typ = str(item.get("type") or "box").lower()
    if typ not in ANNOT_TYPES:
        raise VisionError(f"不支持的标注类型: {typ}（支持: {', '.join(ANNOT_TYPES)}）")
    label = str(item.get("label") or "").strip()
    color = str(item.get("color") or "#ff3b30")
    aid = str(item.get("id") or "").strip()
    coords = str(item.get("coords") or "pixel").lower()
    if coords not in ("pixel", "norm"):
        raise VisionError(f"coords 必须是 'pixel' 或 'norm'，收到: {coords!r}")

    def px(v, n):
        if not isinstance(v, (list, tuple)) or len(v) != n:
            raise VisionError(f"标注几何长度应为 {n}: {v!r}")
        if coords == "norm":
            return [round(float(v[i]) * (w / 1000.0 if i % 2 == 0 else h / 1000.0)) for i in range(n)]
        return [round(float(x)) for x in v]

    if typ == "box":
        box, _ = _clamp_list(px(item.get("box"), 4), w, h)
        return aid, typ, label, color, {"box": box}
    if typ == "point":
        pt, _ = _clamp_list(px(item.get("point"), 2), w, h)
        return aid, typ, label, color, {"point": pt}
    if typ in ("line", "arrow"):
        frm, _ = _clamp_list(px(item.get("from"), 2), w, h)
        to, _ = _clamp_list(px(item.get("to"), 2), w, h)
        return aid, typ, label, color, {"from": frm, "to": to}
    if typ == "circle":
        c, _ = _clamp_list(px(item.get("center"), 2), w, h)
        try:
            r = int(item.get("radius") or 20)
        except (TypeError, ValueError):
            r = 20
        r = max(1, min(r, w, h))
        return aid, typ, label, color, {"center": c, "radius": r}
    if typ == "polygon":
        pts = item.get("points")
        if not isinstance(pts, list) or len(pts) < 3:
            raise VisionError("polygon 需要 points（至少 3 个 [x,y] 点）")
        poly = []
        for p in pts:
            pp, _ = _clamp_list(px(p, 2), w, h)
            poly.append(pp)
        return aid, typ, label, color, {"points": poly}
    if typ == "bubble":
        pt, _ = _clamp_list(px(item.get("point"), 2), w, h)
        text = str(item.get("text") or label or "").strip()
        if not text:
            raise VisionError("bubble 需要 text（气泡文字）或 label")
        direction = str(item.get("direction") or "auto").lower()
        if direction not in ("auto", "up", "down", "left", "right"):
            raise VisionError(f"bubble direction 必须是 auto/up/down/left/right，收到: {direction}")
        return aid, typ, label, color, {"point": pt, "text": text, "direction": direction}
    raise VisionError(f"未实现的标注类型: {typ}")

def _annot_to_text(aid, typ, label, color, geo):
    name = label or aid or {"box": "框", "point": "点", "line": "连线", "arrow": "箭头连线", "circle": "圆", "polygon": "多边形", "bubble": "气泡标注"}[typ]
    if typ == "box":
        b = geo["box"]
        return f"{name}：框 [({b[0]},{b[1]}) -> ({b[2]},{b[3]})]（左上到右下，像素坐标）"
    if typ == "point":
        p = geo["point"]
        return f"{name}：点 ({p[0]},{p[1]})"
    if typ in ("line", "arrow"):
        f, t = geo["from"], geo["to"]
        return f"{name}：{'箭头连线' if typ == 'arrow' else '连线'} 从 ({f[0]},{f[1]}) 到 ({t[0]},{t[1]})"
    if typ == "circle":
        c, r = geo["center"], geo["radius"]
        return f"{name}：圆 圆心 ({c[0]},{c[1]}) 半径 {r}px"
    if typ == "polygon":
        pts = ", ".join(f"({x},{y})" for x, y in geo["points"])
        return f"{name}：多边形 顶点 {pts}"
    if typ == "bubble":
        p = geo["point"]
        return f"{name}：气泡标注 文字「{geo['text']}」 指向点 ({p[0]},{p[1]})"
    return ""

def _draw_annot_overlay(draw, aid, typ, label, color, geo, lw, fnt):
    img_w, img_h = draw._image.size
    if typ == "box":
        b = geo["box"]
        draw.rectangle(b, outline=color, width=lw, fill=color + "33")
        txt = label or aid
        if txt:
            tb = draw.textbbox((0, 0), txt, font=fnt)
            th = tb[3] - tb[1]
            draw.rectangle([b[0], max(0, b[1] - th - 6), b[0] + (tb[2] - tb[0]) + 8, b[1]], fill=color)
            draw.text((b[0] + 4, max(0, b[1] - th - 4)), txt, fill="white", font=fnt)
    elif typ == "point":
        p = geo["point"]
        r = max(6, lw + 3)
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color)
    elif typ in ("line", "arrow"):
        f, t = geo["from"], geo["to"]
        draw.line([f, t], fill=color, width=lw)
        if typ == "arrow":
            import math as _m
            ang = _m.atan2(t[1] - f[1], t[0] - f[0])
            hlen = 12
            for da in (0.5, -0.5):
                draw.line([t, (round(t[0] - hlen * _m.cos(ang + da)), round(t[1] - hlen * _m.sin(ang + da)))], fill=color, width=lw)
    elif typ == "circle":
        c, r = geo["center"], geo["radius"]
        draw.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=color, width=lw, fill=color + "22")
    elif typ == "polygon":
        pts = geo["points"]
        draw.polygon(pts, outline=color, fill=color + "33")
        draw.line(pts + [pts[0]], fill=color, width=lw)
        txt = label or aid
        if txt:
            cx = sum(p[0] for p in pts) // len(pts)
            cy = sum(p[1] for p in pts) // len(pts)
            tb = draw.textbbox((0, 0), txt, font=fnt)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            draw.rectangle([cx, max(0, cy - th - 4), cx + tw + 8, cy + 2], fill=color)
            draw.text((cx + 4, max(0, cy - th - 2)), txt, fill="white", font=fnt)
    elif typ == "bubble":
        p = geo["point"]
        txt = geo["text"]
        direction = geo["direction"]
        tb = draw.textbbox((0, 0), txt, font=fnt)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = 6
        if direction == "auto":
            direction = "right" if p[0] < img_w * 0.5 else "left"
        if direction == "up":
            rect = [p[0] - tw // 2 - pad, max(0, p[1] - th - 2 * pad - 10), p[0] + tw // 2 + pad, p[1] - 10]
            tail = [(p[0], p[1] - 10), (p[0] - 6, p[1]), (p[0] + 6, p[1])]
        elif direction == "down":
            rect = [p[0] - tw // 2 - pad, p[1] + 10, min(img_w, p[0] + tw // 2 + pad), p[1] + th + 2 * pad + 10]
            tail = [(p[0], p[1] + 10), (p[0] - 6, p[1]), (p[0] + 6, p[1])]
        elif direction == "left":
            rect = [max(0, p[0] - tw - 2 * pad - 10), p[1] - th // 2 - pad, p[0] - 10, p[1] + th // 2 + pad]
            tail = [(p[0] - 10, p[1]), (p[0], p[1] - 6), (p[0], p[1] + 6)]
        else:
            rect = [p[0] + 10, p[1] - th // 2 - pad, min(img_w, p[0] + tw + 2 * pad + 10), p[1] + th // 2 + pad]
            tail = [(p[0] + 10, p[1]), (p[0], p[1] - 6), (p[0], p[1] + 6)]
        draw.rounded_rectangle(rect, radius=8, fill="white", outline=color, width=2)
        draw.polygon(tail, fill="white", outline=color)
        draw.line([p, tail[0]], fill=color, width=2)
        tx = rect[0] + pad
        ty = rect[1] + max(0, (rect[3] - rect[1] - th) // 2)
        draw.text((tx, ty), txt, fill="#111111", font=fnt)

MAX_MODEL_SIDE = _env_int("VISION_MAX_MODEL_SIDE", 2600)
NORM_SIZE = _env_int("VISION_NORM_SIZE", 0)

def _fit_model(img):
    """把图归一化到视觉模型友好的尺寸。返回 (模型图, scale_x, scale_y)。
    NORM_SIZE>0 时精确归一化到目标长边（保持纵横比）；否则仅限制 MAX_MODEL_SIDE 上限。"""
    w, h = img.size
    m = max(w, h)
    if NORM_SIZE > 0:
        s = NORM_SIZE / m
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        return img.resize((nw, nh), Image.LANCZOS), w / nw, h / nh
    if m <= MAX_MODEL_SIDE:
        return img, 1.0, 1.0
    s = MAX_MODEL_SIDE / m
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    return img.resize((nw, nh), Image.LANCZOS), w / nw, h / nh

def _scale_geo(geo, sx, sy):
    """把原图坐标几何换算到模型图坐标。"""
    ngeo = {}
    for k, v in geo.items():
        if k == "points":
            ngeo[k] = [[round(x / sx), round(y / sy)] for x, y in v]
        elif k in ("box", "from", "to", "center", "point") and isinstance(v, list) and len(v) in (2, 4) and all(isinstance(x, (int, float)) for x in v):
            ngeo[k] = [round(v[0] / sx), round(v[1] / sy)] if len(v) == 2 else [round(v[0] / sx), round(v[1] / sy), round(v[2] / sx), round(v[3] / sy)]
        elif k == "radius":
            ngeo[k] = max(1, round(v / min(sx, sy)))
        else:
            ngeo[k] = v
    return ngeo

def tool_annotate_infer(args):
    src = args["image"]
    items = args.get("items")
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        raise VisionError("items 必须是标注数组（或单个对象）")
    auto_boxes_arg = args.get("auto_boxes")
    if not items and not auto_boxes_arg:
        raise VisionError("items 或 auto_boxes 至少提供一个")
    question = str(args.get("question") or "").strip()
    if not question:
        raise VisionError("缺少参数: question（推理问题）")
    mode = str(args.get("mode") or "virtual").lower()
    if mode not in ("virtual", "overlay"):
        raise VisionError("mode 必须是 'virtual' 或 'overlay'")
    try:
        alpha = float(args.get("alpha") or 0.35)
    except (TypeError, ValueError):
        alpha = 0.35
    if not 0 < alpha <= 1:
        raise VisionError("alpha 必须在 (0, 1] 之间")

    img, raw, label = load_image(src)
    w, h = img.size

    # ---- 基准标注（含 id 分配） ----
    parsed = []
    auto_id = 0
    for it in items:
        aid, typ, lbl, color, geo = _parse_annot_item(it, w, h)
        if not aid:
            auto_id += 1
            aid = f"a{auto_id}"
        parsed.append([aid, typ, lbl, color, geo])

    # ---- 多轮修正 corrections ----
    corrections = args.get("corrections")
    if corrections is not None:
        if not isinstance(corrections, list):
            raise VisionError("corrections 必须是操作数组")
        for op in corrections:
            if not isinstance(op, dict):
                raise VisionError("corrections 每项必须是对象")
            opn = str(op.get("op") or "").lower()
            oid = str(op.get("id") or "")
            if opn == "add":
                it = op.get("item")
                if not isinstance(it, dict):
                    raise VisionError("corrections add 需要 item")
                aid2, typ2, lbl2, color2, geo2 = _parse_annot_item(it, w, h)
                if not aid2:
                    auto_id += 1
                    aid2 = f"a{auto_id}"
                parsed.append([aid2, typ2, lbl2, color2, geo2])
            elif opn == "remove":
                parsed = [p for p in parsed if p[0] != oid]
            elif opn in ("move", "resize", "set"):
                hit = None
                for i, p in enumerate(parsed):
                    if p[0] == oid:
                        hit = i
                        break
                if hit is None:
                    raise VisionError(f"corrections 引用的标注不存在: {oid}")
                aid2, typ2, lbl2, color2, geo2 = parsed[hit]
                if opn == "move":
                    delta = op.get("delta")
                    to = op.get("to")
                    ngeo = dict(geo2)
                    if isinstance(delta, (list, tuple)) and len(delta) == 2:
                        dx, dy = int(delta[0]), int(delta[1])
                        for k, v in ngeo.items():
                            if k == "points" and isinstance(v, list):
                                ngeo[k] = [[p[0] + dx, p[1] + dy] for p in v]
                            elif isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                                ngeo[k] = [v[0] + dx, v[1] + dy]
                            elif isinstance(v, list) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v):
                                ngeo[k] = [v[0] + dx, v[1] + dy, v[2] + dx, v[3] + dy]
                    elif isinstance(to, (list, tuple)) and len(to) == 2:
                        t = [int(to[0]), int(to[1])]
                        if "point" in ngeo:
                            ngeo["point"] = t
                        elif "box" in ngeo:
                            dw = (ngeo["box"][2] - ngeo["box"][0]) // 2
                            dh = (ngeo["box"][3] - ngeo["box"][1]) // 2
                            ngeo["box"] = [t[0] - dw, t[1] - dh, t[0] + dw, t[1] + dh]
                        elif "from" in ngeo and "to" in ngeo:
                            dw = (ngeo["to"][0] - ngeo["from"][0]) // 2
                            dh = (ngeo["to"][1] - ngeo["from"][1]) // 2
                            ngeo["from"] = [t[0] - dw, t[1] - dh]
                            ngeo["to"] = [t[0] + dw, t[1] + dh]
                        elif "center" in ngeo:
                            ngeo["center"] = t
                        elif "points" in ngeo:
                            cx = sum(p[0] for p in ngeo["points"]) // len(ngeo["points"])
                            cy = sum(p[1] for p in ngeo["points"]) // len(ngeo["points"])
                            dw, dh = t[0] - cx, t[1] - cy
                            ngeo["points"] = [[p[0] + dw, p[1] + dh] for p in ngeo["points"]]
                    else:
                        raise VisionError("corrections move 需要 delta [dx,dy] 或 to [x,y]")
                    re_item = {"type": typ2, "label": lbl2, "color": color2}
                    re_item.update(ngeo)
                    aid3, typ3, lbl3, color3, geo3 = _parse_annot_item(re_item, w, h)
                    parsed[hit] = [aid2, typ3, lbl3, color3, geo3]
                else:
                    patch = {k: v for k, v in op.items() if k in ("box", "point", "from", "to", "center", "points", "radius", "text", "direction")}
                    if not patch:
                        raise VisionError("corrections resize/set 需要提供几何字段（box/point/from/to/center/points/radius/text/direction）")
                    re_item = {"type": typ2, "label": lbl2, "color": color2}
                    re_item.update(patch)
                    aid3, typ3, lbl3, color3, geo3 = _parse_annot_item(re_item, w, h)
                    parsed[hit] = [aid2, typ3, lbl3, color3, geo3]
            else:
                raise VisionError(f"不支持的修正操作: {opn}（支持 add/remove/move/resize/set）")

    # ---- 自动框选 auto_boxes ----
    auto_boxes = args.get("auto_boxes")
    auto_applied = []
    if auto_boxes:
        if isinstance(auto_boxes, str):
            auto_boxes = [auto_boxes]
        if not isinstance(auto_boxes, list):
            raise VisionError("auto_boxes 必须是字符串或数组")
        for tgt in auto_boxes:
            tgt = str(tgt).strip()
            if not tgt:
                continue
            loc = tool_locate_object({"image": src, "target": tgt})
            for pr in loc.get("primitives", []):
                if pr.get("box"):
                    auto_id += 1
                    aid2 = f"auto{auto_id}"
                    parsed.append([aid2, "box", tgt, "#9b59b6", {"box": pr["box"]}])
                    auto_applied.append({"id": aid2, "target": tgt, "box": pr["box"]})

    # ---- 模型图（大图降采样）与标注描述 ----
    mimg, sx, sy = _fit_model(img)
    parsed_model = []
    for aid2, typ2, lbl2, color2, geo2 in parsed:
        parsed_model.append([aid2, typ2, lbl2, color2, _scale_geo(geo2, sx, sy)])
    descs = [_annot_to_text(*p) for p in parsed_model]
    annot_text = "\n".join(descs)
    key = cache_key(raw, "annotate_infer", mode, question, annot_text, str(alpha))
    hit = cache_get(key)
    if hit is not None:
        log("cache hit: annotate_infer")
        return hit

    base_prompt = (
        "请分析这张图像，并结合以下**虚拟标注**（这些标注不是图像上的实际内容，"
        "仅用于指示位置、区域和关系；请以标注为参考进行空间推理，并区分图像实际内容与标注）：\n"
        + annot_text + f"\n推理问题：{question}\n请结合图像内容与标注关系给出分析。"
    )

    applied = []
    for aid2, typ2, lbl2, color2, geo2 in parsed:
        entry = {"id": aid2, "type": typ2, "color": color2}
        if lbl2:
            entry["label"] = lbl2
        entry.update(geo2)
        applied.append(entry)

    if mode == "virtual":
        text = call_chat([
            {"role": "system", "content": AUX_VISION_SYSTEM},
            image_message(base_prompt, mimg),
        ]).strip()
        out = {
            "mode": "virtual",
            "answer": text,
            "annotations": len(parsed),
            "applied": applied,
            "auto_boxes_applied": auto_applied if auto_applied else None,
            "corrections_applied": bool(corrections),
            "model_image_size": [mimg.width, mimg.height],
        }
    else:
        mw, mh = mimg.size
        overlay = Image.new("RGBA", (mw, mh), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        lw = max(2, min(6, mw // 300))
        fnt = _font(max(12, mw // 60))
        for aid2, typ2, lbl2, color2, geo2 in parsed_model:
            _draw_annot_overlay(od, aid2, typ2, lbl2, color2, geo2, lw, fnt)
        if alpha < 1.0:
            a_layer = overlay.getchannel("A").point(lambda a: round(a * alpha))
            overlay = overlay.copy()
            overlay.putalpha(a_layer)
        composite = Image.alpha_composite(mimg.convert("RGBA"), overlay).convert("RGB")
        out_path = _unique_path("annotate_infer")
        composite.save(out_path, "PNG")
        prompt = "这张图像上已叠加半透明标注层（框/点/线/圆/多边形/气泡，叠加不会遮挡原图内容）。" + base_prompt
        text = call_chat([
            {"role": "system", "content": AUX_VISION_SYSTEM},
            image_message(prompt, composite),
        ]).strip()
        out = {
            "mode": "overlay",
            "answer": text,
            "annotations": len(parsed),
            "overlay_path": str(out_path),
            "alpha": alpha,
            "applied": applied,
            "auto_boxes_applied": auto_applied if auto_applied else None,
            "corrections_applied": bool(corrections),
            "model_image_size": [mw, mh],
        }
    cache_set(key, out)
    return out






# ----------------------------- 电脑控制（Computer Use） -----------------------------

SCREEN_CONTROL_ALLOWED = _env("VISION_ALLOW_SCREEN_CONTROL", "0") == "1"

_VK = {
    "enter": 0x0D, "return": 0x0D, "tab": 0x09, "backspace": 0x08, "space": 0x20,
    "escape": 0x1B, "esc": 0x1B, "delete": 0x2E, "insert": 0x2D,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "ctrl": 0x11, "shift": 0x10, "alt": 0x12, "win": 0x5B, "capslock": 0x14,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74, "f6": 0x75,
    "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
}

def _require_screen_control():
    if not SCREEN_CONTROL_ALLOWED:
        raise VisionError("电脑控制未启用：请设置环境变量 VISION_ALLOW_SCREEN_CONTROL=1（注意：此开关会允许模型操控你的鼠标键盘）")

def _user32():
    import ctypes
    return ctypes.windll.user32

def _vk_for_char(ch):
    o = ord(ch)
    if 0x30 <= o <= 0x39 or 0x41 <= o <= 0x5A:
        return o  # 数字/字母
    return _VK.get(ch.lower())

def _press_vk(vk, hold=0.03):
    u = _user32()
    u.keybd_event(vk, 0, 0, 0)
    time.sleep(hold)
    u.keybd_event(vk, 0, 2, 0)
    time.sleep(0.02)

def _key_combo(combo):
    parts = [p.strip().lower() for p in combo.split("+")]
    mods = [p for p in parts if p in ("ctrl", "shift", "alt", "win")]
    keys = [p for p in parts if p not in ("ctrl", "shift", "alt", "win")]
    u = _user32()
    for m in mods:
        u.keybd_event(_VK[m], 0, 0, 0)
        time.sleep(0.03)
    for k in keys:
        if len(k) == 1 and _vk_for_char(k):
            _press_vk(_vk_for_char(k))
        elif k in _VK:
            _press_vk(_VK[k])
        else:
            raise VisionError(f"不支持的按键: {k}")
    for m in reversed(mods):
        u.keybd_event(_VK[m], 0, 2, 0)
        time.sleep(0.03)

def tool_screen_capture(args):
    region = args.get("region")
    out_path = _resolve_out_path(args.get("out_path")) or _unique_path("screen")
    try:
        from PIL import ImageGrab
    except ImportError:
        raise VisionError("PIL ImageGrab 不可用（需要 Windows/带 GUI 环境）")
    if region:
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            raise VisionError("region 必须是 [x1,y1,x2,y2]")
        region = [int(v) for v in region]
        if region[2] <= region[0] or region[3] <= region[1]:
            raise VisionError("region 无效")
        img = ImageGrab.grab(bbox=tuple(region))
    else:
        img = ImageGrab.grab()
    img.save(out_path, "PNG")
    return {"path": str(out_path), "size": [img.width, img.height], "region": region}

def tool_screen_info(args=None):
    u = _user32()
    sw, sh = u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    import ctypes
    dpi = 96
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        dpi = ctypes.windll.shcore.GetDpiForWindow(u.GetDesktopWindow())
    except Exception:
        pass
    return {
        "screen_size": [sw, sh],
        "dpi": dpi,
        "control_allowed": SCREEN_CONTROL_ALLOWED,
        "note": "control_allowed=false 时点击/输入/滚动/拖拽/按键不可用（设 VISION_ALLOW_SCREEN_CONTROL=1 开启）",
    }

def tool_screen_click(args):
    _require_screen_control()
    x, y = int(args.get("x")), int(args.get("y"))
    button = str(args.get("button") or "left").lower()
    if button not in ("left", "right", "middle"):
        raise VisionError(f"button 必须是 left/right/middle，收到: {button}")
    double = args.get("double") in (True, "true", "1", 1)
    u = _user32()
    u.SetCursorPos(x, y)
    time.sleep(0.05)
    down = {"left": 0x0002, "right": 0x0008, "middle": 0x0020}[button]
    up = {"left": 0x0004, "right": 0x0010, "middle": 0x0040}[button]
    u.mouse_event(down, 0, 0, 0, 0)
    time.sleep(0.05)
    u.mouse_event(up, 0, 0, 0, 0)
    if double:
        time.sleep(0.05)
        u.mouse_event(down, 0, 0, 0, 0)
        time.sleep(0.05)
        u.mouse_event(up, 0, 0, 0, 0)
    return {"clicked": [x, y], "button": button, "double": double}

def tool_screen_move(args):
    _require_screen_control()
    x, y = int(args.get("x")), int(args.get("y"))
    _user32().SetCursorPos(x, y)
    return {"moved_to": [x, y]}

def tool_screen_drag(args):
    _require_screen_control()
    x1, y1, x2, y2 = (int(args.get(k)) for k in ("x1", "y1", "x2", "y2"))
    duration = max(0.0, float(args.get("duration") or 0.2))
    button = str(args.get("button") or "left").lower()
    if button not in ("left", "right", "middle"):
        raise VisionError(f"button 必须是 left/right/middle，收到: {button}")
    u = _user32()
    down = {"left": 0x0002, "right": 0x0008, "middle": 0x0020}[button]
    up = {"left": 0x0004, "right": 0x0010, "middle": 0x0040}[button]
    u.SetCursorPos(x1, y1)
    time.sleep(0.05)
    u.mouse_event(down, 0, 0, 0, 0)
    steps = max(1, int(duration * 50))
    for i in range(1, steps + 1):
        cx = x1 + (x2 - x1) * i // steps
        cy = y1 + (y2 - y1) * i // steps
        u.SetCursorPos(cx, cy)
        time.sleep(duration / steps)
    u.mouse_event(up, 0, 0, 0, 0)
    return {"dragged": [[x1, y1], [x2, y2]], "button": button}

def tool_screen_scroll(args):
    _require_screen_control()
    delta = int(args.get("delta") or 0)
    if delta == 0:
        raise VisionError("delta 不能为 0（正=向上滚，负=向下滚）")
    _user32().mouse_event(0x0800, 0, 0, delta * 120, 0)
    return {"scrolled": delta}

def tool_screen_type(args):
    _require_screen_control()
    text = str(args.get("text") or "")
    if not text:
        raise VisionError("text 不能为空")
    # 纯 ASCII 直接按键；含非 ASCII（中文等）走剪贴板粘贴
    if all(ord(c) < 128 for c in text):
        for ch in text:
            vk = _vk_for_char(ch)
            if vk is None:
                raise VisionError(f"不支持的字符: {ch!r}")
            _press_vk(vk)
        return {"typed": text, "method": "keyboard"}
    import subprocess
    enc = text.encode("utf-8")
    subprocess.run(["powershell", "-NoProfile", "-Command", "[Console]::InputEncoding=[Text.Encoding]::UTF8; Set-Clipboard -Value ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('" + base64.b64encode(enc).decode() + "')))"], creationflags=0x08000000, timeout=30)
    _key_combo("ctrl+v")
    return {"typed": text, "method": "clipboard"}

def tool_screen_key(args):
    _require_screen_control()
    key = str(args.get("key") or "").strip().lower()
    if not key:
        raise VisionError("key 不能为空（如 enter / tab / ctrl+c / alt+tab）")
    if "+" in key:
        _key_combo(key)
        return {"pressed": key, "combo": True}
    if len(key) == 1 and _vk_for_char(key):
        _press_vk(_vk_for_char(key))
        return {"pressed": key}
    if key in _VK:
        _press_vk(_VK[key])
        return {"pressed": key}
    raise VisionError(f"不支持的按键: {key}")

# ----------------------------- 多图联合推理（compare_infer） -----------------------------

def tool_compare_infer(args):
    images = args.get("images")
    if not isinstance(images, list) or not (2 <= len(images) <= 4):
        raise VisionError("images 必须是 2-4 张图片（本地路径或 http(s) URL）的数组")
    question = str(args.get("question") or "").strip()
    if not question:
        raise VisionError("缺少参数: question（联合推理问题）")
    mode = str(args.get("mode") or "virtual").lower()
    if mode not in ("virtual", "overlay"):
        raise VisionError("mode 必须是 'virtual' 或 'overlay'")
    try:
        alpha = float(args.get("alpha") or 0.35)
    except (TypeError, ValueError):
        alpha = 0.35
    if not 0 < alpha <= 1:
        raise VisionError("alpha 必须在 (0, 1] 之间")

    items_per_image = args.get("items_per_image") or {}
    if not isinstance(items_per_image, dict):
        raise VisionError("items_per_image 必须是 {图索引: 标注数组} 对象")

    mimages = []
    per_desc = []
    per_applied = []
    overlay_paths = []
    for i, src in enumerate(images):
        img, raw, label = load_image(src)
        mimg, sx, sy = _fit_model(img)
        mimages.append(mimg)
        items = items_per_image.get(i) or []
        if isinstance(items, dict):
            items = [items]
        parsed = []
        auto_id = 0
        for it in items:
            aid, typ, lbl, color, geo = _parse_annot_item(it, img.width, img.height)
            if not aid:
                auto_id += 1
                aid = f"i{i+1}a{auto_id}"
            parsed.append([aid, typ, lbl, color, _scale_geo(geo, sx, sy)])
        applied = []
        for aid2, typ2, lbl2, color2, geo2 in parsed:
            entry = {"id": aid2, "type": typ2, "color": color2}
            if lbl2:
                entry["label"] = lbl2
            entry.update(geo2)
            applied.append(entry)
        per_applied.append(applied)
        if parsed:
            desc = "\n".join(_annot_to_text(*p) for p in parsed)
            per_desc.append(f"【图{i+1}】标注：\n{desc}")
        else:
            per_desc.append(f"【图{i+1}】无标注")
        if mode == "overlay" and parsed:
            mw, mh = mimg.size
            overlay = Image.new("RGBA", (mw, mh), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            lw = max(2, min(6, mw // 300))
            fnt = _font(max(12, mw // 60))
            for aid2, typ2, lbl2, color2, geo2 in parsed:
                _draw_annot_overlay(od, aid2, typ2, lbl2, color2, geo2, lw, fnt)
            if alpha < 1.0:
                a_layer = overlay.getchannel("A").point(lambda a: round(a * alpha))
                overlay = overlay.copy()
                overlay.putalpha(a_layer)
            composite = Image.alpha_composite(mimg.convert("RGBA"), overlay).convert("RGB")
            op = _unique_path("compare_infer")
            composite.save(op, "PNG")
            overlay_paths.append({"index": i, "path": str(op)})
            mimages[i] = composite

    key = cache_key(b"|".join([str(len(images)).encode(), question.encode("utf-8"), json.dumps(items_per_image, ensure_ascii=False).encode("utf-8")]), "compare_infer", mode, str(alpha))
    hit = cache_get(key)
    if hit is not None:
        log("cache hit: compare_infer")
        return hit

    names = "、".join(f"图{i+1}" for i in range(len(images)))
    prompt = (
        f"请联合分析以下 {len(images)} 张图像（编号：{names}）。"
        "先分别理解每张图的内容，再对比/联合推理它们之间的关系（相同点、差异、因果、时序、整体结论）。\n"
        + "\n".join(per_desc)
        + f"\n联合推理问题：{question}"
    )
    content = [{"type": "text", "text": prompt}]
    for mimg in mimages:
        data_url, _ = encode_png(mimg)
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    text = call_chat([
        {"role": "system", "content": AUX_VISION_SYSTEM},
        {"role": "user", "content": content},
    ]).strip()
    out = {
        "mode": mode,
        "answer": text,
        "images": len(images),
        "applied_per_image": per_applied,
        "overlay_paths": overlay_paths if overlay_paths else None,
    }
    cache_set(key, out)
    return out


# ----------------------------- 交互式图形推理协议（reason_graph） -----------------------------

def _resolve_measure_coords(refs, prims):
    coords = []
    for r in refs or []:
        if isinstance(r, (list, tuple)) and len(r) in (2, 4) and all(isinstance(x, (int, float)) for x in r):
            coords.append([float(x) for x in r])
        else:
            p = next((x for x in prims if x.get("id") == str(r)), None)
            if p and p.get("geometry"):
                g = p["geometry"]
                coords.append([(g[0] + g[2]) / 2.0, (g[1] + g[3]) / 2.0] if len(g) == 4 else [float(g[0]), float(g[1])])
    return coords

def tool_reason_graph(args):
    image = args["image"]
    question = str(args.get("question") or "").strip()
    session = args.get("session") or {}
    if not isinstance(session, dict):
        raise VisionError("session 必须是对象")
    step = args.get("step") or {}
    if not isinstance(step, dict):
        raise VisionError("step 必须是对象")
    stype = str(step.get("type") or "next").lower()
    valid = ("locate", "measure", "annotate", "semantic", "hypothesis", "verify", "next")
    if stype not in valid:
        raise VisionError(f"不支持的 step 类型: {stype}（支持: {', '.join(valid)}）")

    img, raw, label = load_image(image)
    prims = session.get("primitives") or []
    anns = session.get("annotations") or []
    sem = session.get("semantics") or []
    hyps = session.get("hypotheses") or []
    results = {}

    if stype == "locate":
        target = str(step.get("target") or "").strip() or question
        if not target:
            raise VisionError("locate step 需要 target（定位目标）")
        refine = step.get("refine") in (True, "true", "1", 1)
        loc = tool_locate_object({"image": image, "target": target, "refine": refine})
        got = []
        for i, pr in enumerate(loc.get("primitives", [])):
            pid = str(step.get("id") or f"p{len(prims) + i + 1}")
            entry = {
                "id": pid,
                "type": "box",
                "label": target,
                "geometry": pr.get("box_pixel") or pr.get("box"),
                "confidence": pr.get("confidence"),
                "source": "locate",
            }
            prims.append(entry)
            got.append(entry)
        results["located"] = got
    elif stype == "measure":
        mtype = str(step.get("measure") or step.get("metric") or "distance").lower()
        refs = step.get("refs") or []
        coords = _resolve_measure_coords(refs, prims)
        if mtype == "distance":
            if len(coords) < 2:
                raise VisionError("distance 需要至少 2 个参考点（坐标或 primitives id）")
            import math as _m
            results["distance_px"] = round(_m.dist(coords[0], coords[1]), 1)
        elif mtype == "angle":
            if len(coords) < 3:
                raise VisionError("angle 需要 3 个参考点（以第 2 个为顶点）")
            import math as _m
            a, b, c = coords[0], coords[1], coords[2]
            v1 = (a[0] - b[0], a[1] - b[1])
            v2 = (c[0] - b[0], c[1] - b[1])
            den = _m.hypot(*v1) * _m.hypot(*v2)
            cosv = (v1[0] * v2[0] + v1[1] * v2[1]) / (den or 1.0)
            results["angle_deg"] = round(_m.degrees(_m.acos(max(-1.0, min(1.0, cosv)))), 1)
        elif mtype == "area":
            if not coords:
                raise VisionError("area 需要参考点或框")
            p0 = coords[0]
            if len(p0) == 4:
                results["area_px2"] = round((p0[2] - p0[0]) * (p0[3] - p0[1]), 1)
            elif len(coords) >= 3:
                import math as _m
                pts = coords
                area = abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))) / 2.0
                results["area_px2"] = round(area, 1)
            else:
                raise VisionError("area 需要至少 3 个点或 1 个框")
        else:
            raise VisionError(f"不支持的测量类型: {mtype}（支持 distance/angle/area）")
        results["measure"] = mtype
    elif stype == "annotate":
        items = step.get("items")
        if items is None:
            items = []
            for p in prims:
                g = p.get("geometry")
                if g and len(g) == 4:
                    items.append({"id": p["id"], "type": "box", "label": p.get("label") or p["id"], "box": g})
                elif g and len(g) == 2:
                    items.append({"id": p["id"], "type": "point", "label": p.get("label") or p["id"], "point": g})
        ann = tool_annotate_infer({"image": image, "items": items, "question": question or "确认标注位置", "mode": "overlay", "alpha": 0.4})
        anns = ann.get("applied") or []
        results["overlay_path"] = ann.get("overlay_path")
        results["annotations"] = anns
    elif stype == "semantic":
        text = str(step.get("text") or "").strip()
        if not text:
            raise VisionError("semantic step 需要 text（语义描述）")
        sem.append(text)
        results["semantics"] = list(sem)
    elif stype == "hypothesis":
        text = str(step.get("text") or "").strip()
        if not text:
            raise VisionError("hypothesis step 需要 text（假设内容）")
        hid = str(step.get("id") or f"h{len(hyps) + 1}")
        hyps.append({"id": hid, "text": text, "status": str(step.get("status") or "proposed")})
        results["hypotheses"] = list(hyps)
    elif stype == "verify":
        hid = str(step.get("id") or "")
        hyp = next((x for x in hyps if x["id"] == hid), None)
        items = step.get("items")
        if items is None:
            items = []
            for p in prims:
                g = p.get("geometry")
                if g and len(g) == 4:
                    items.append({"id": p["id"], "type": "box", "label": p.get("label") or p["id"], "box": g})
        q = f"验证推理假设：{hyp['text'] if hyp else '（未指定）'}。{question}"
        r = tool_annotate_infer({"image": image, "items": items, "question": q, "mode": "virtual"})
        results["verdict"] = r.get("answer")
        if hyp:
            hyp["status"] = "verified"
            results["hypothesis_status"] = "verified"
    elif stype == "next":
        items = []
        for p in prims:
            g = p.get("geometry")
            if g and len(g) == 4:
                items.append({"id": p["id"], "type": "box", "label": p.get("label") or p["id"], "box": g})
        q = (f"当前图形推理状态：原语 {len(prims)} 个、语义记录 {len(sem)} 条、假设 {len(hyps)} 条。"
             f"请基于现状建议下一步最有价值的推理动作（继续定位/测量/验证哪个假设/新假设），并说明理由。{question}")
        r = tool_annotate_infer({"image": image, "items": items, "question": q, "mode": "virtual"})
        results["next_step"] = r.get("answer")

    session_out = {"primitives": prims, "annotations": anns, "semantics": sem, "hypotheses": hyps}
    return {"session": session_out, "step": stype, "results": results}

# ----------------------------- 多图对比（compare_images） -----------------------------

def tool_compare_images(args):
    images = args.get("images")
    if not isinstance(images, list) or not (2 <= len(images) <= 4):
        raise VisionError("images 必须是 2-4 张图片（本地路径或 http(s) URL）的数组")
    question = str(args.get("question") or "").strip()
    detail = str(args.get("detail") or "balanced").lower()
    if detail not in ("brief", "balanced", "detailed"):
        detail = "balanced"
    imgs, raws = [], []
    for src in images:
        img, raw, label = load_image(src)
        imgs.append(img)
        raws.append(raw)
    key = cache_key(b"|".join(raws), "compare", question, detail)
    hit = cache_get(key)
    if hit is not None:
        log("cache hit: compare")
        return hit
    names = "、".join(f"图{i+1}" for i in range(len(imgs)))
    prompt = (
        f"请对比分析以下 {len(imgs)} 张图像（编号：{names}）。逐项对比："
        "1) 整体内容与布局；2) 相同点；3) 差异点（文字、元素、颜色、位置、状态等，尽量具体）；4) 结论/判断。\n"
    )
    prompt += {"brief": "简要回答，每项 1-2 句。", "balanced": "每项给出要点即可。", "detailed": "尽可能详细，逐条列出差异。"}[detail]
    if question:
        prompt += f"\n用户关注点（重点回答）：{question}"
    content = [{"type": "text", "text": prompt}]
    for img in imgs:
        data_url, _ = encode_png(img)
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    text = call_chat([
        {"role": "system", "content": AUX_VISION_SYSTEM},
        {"role": "user", "content": content},
    ]).strip()
    cache_set(key, text)
    return text

# ----------------------------- 异常元件扫描（scan_anomalies） -----------------------------

def _build_tiles(region, tile_size, overlap, max_tiles):
    """把 region 切成带重叠的块。返回 [(x0,y0,x1,y1), ...]。"""
    x0, y0, x1, y1 = region
    rw, rh = x1 - x0, y1 - y0
    if tile_size and tile_size > 0:
        cols = max(1, min(max_tiles, math.ceil(rw / tile_size)))
        rows = max(1, min(max_tiles // cols if cols else 1, math.ceil(rh / tile_size)))
    else:
        ratio = rw / max(rh, 1)
        if ratio >= 2.0:
            cols, rows = max(1, min(max_tiles, max(1, round(rw / max(rh, 1))))), 1
        elif ratio <= 0.5:
            cols, rows = 1, max(1, min(max_tiles, max(1, round(rh / max(rw, 1)))))
        else:
            cols = rows = max(1, int(math.sqrt(max_tiles)))
            while cols * rows > max_tiles:
                cols -= 1
            cols = max(1, cols)
            rows = max(1, min(rows, max_tiles // cols))
    step_x = rw / cols if cols > 1 else rw
    step_y = rh / rows if rows > 1 else rh
    tiles = []
    for r in range(rows):
        for c in range(cols):
            bx0 = x0 + round(c * step_x)
            by0 = y0 + round(r * step_y)
            bx1 = x0 + round((c + 1) * step_x) if c < cols - 1 else x1
            by1 = y0 + round((r + 1) * step_y) if r < rows - 1 else y1
            if cols > 1 and c > 0:
                bx0 = max(x0, bx0 - overlap)
            if rows > 1 and r > 0:
                by0 = max(y0, by0 - overlap)
            if cols > 1 and c < cols - 1:
                bx1 = min(x1, bx1 + overlap)
            if rows > 1 and r < rows - 1:
                by1 = min(y1, by1 + overlap)
            if bx1 - bx0 >= 200 and by1 - by0 >= 200:
                tiles.append((bx0, by0, bx1, by1))
    return tiles

def _box_center_xy(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

def _boxes_close(a, b, min_dist=150):
    ca, cb = _box_center_xy(a), _box_center_xy(b)
    if abs(ca[0] - cb[0]) <= min_dist and abs(ca[1] - cb[1]) <= min_dist:
        return True
    # IoU 判定
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return False
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(union, 1) >= 0.2

def _merge_candidates(cands):
    merged = []
    for c in cands:
        box = c["box"]
        placed = False
        for m in merged:
            if _boxes_close(box, m["box"]):
                m["box"] = [
                    min(box[0], m["box"][0]), min(box[1], m["box"][1]),
                    max(box[2], m["box"][2]), max(box[3], m["box"][3]),
                ]
                m["confidence"] = max(m.get("confidence") or 0, c.get("confidence") or 0)
                if c.get("label"):
                    m["labels"].append(c["label"])
                if c.get("tile"):
                    m["tiles"].append(c["tile"])
                placed = True
                break
        if not placed:
            merged.append({
                "box": box,
                "confidence": c.get("confidence"),
                "labels": [c.get("label")] if c.get("label") else [],
                "tiles": [c.get("tile")] if c.get("tile") else [],
            })
    return merged

VERIFY_PROMPT = (
    "请客观检查这张 PCB 局部放大图，只描述你实际看到的，不要猜测：\n"
    "1) 图中是否存在明显歪斜/旋转摆放的元件（相对图像水平或垂直轴明显偏转，且与周边元件方向不一致）？回答 是/否。\n"
    "2) 若有，旋转角度约多少度？\n"
    "3) 该元件的丝印字符是什么（如无则写：无）？\n"
    "4) 元件类型（三极管/MOS管/LDO稳压器/电阻/电容/其他）？\n"
    "如果没有任何明显歪斜的元件，直接回答：没有明显歪斜元件。"
)

def _parse_verdict(text):
    t = text or ""
    verdict = "unclear"
    neg = re.search(r"没有明显歪斜|未发现歪斜|不存在歪斜|无歪斜|没有歪斜|没有明显|没有可识别|无法判断", t)
    pos = re.search(r"(歪斜|倾斜|旋转|偏转)", t) or re.search(r"^\s*1\)\s*是", t, re.M)
    if neg:
        verdict = "not_skewed"
    elif pos or re.search(r"\d+\s*(?:°|度)", t):
        verdict = "skewed"
    rotation = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:°|度)", t)
    if m:
        try:
            rotation = float(m.group(1))
        except ValueError:
            rotation = None
    silkscreen = None
    m = re.search(r"(?:丝印|字符|marking)[：:为是]?\s*([A-Za-z0-9_\-]{1,16})", t, re.I)
    if not m:
        m = re.search(r"^\s*\d+\)\s*([A-Za-z0-9_\-]{1,16})\s*$", t, re.M)
    if m:
        silkscreen = m.group(1)
    etype = None
    for kw in ("LDO", "三极管", "MOS", "稳压", "电阻", "电容", "电感"):
        if kw in t:
            etype = kw
            break
    return {"verdict": verdict, "rotation": rotation, "silkscreen": silkscreen, "component_type": etype}

def tool_scan_anomalies(args):
    src = args["image"]
    target = str(args.get("target") or "摆放歪斜、方向与周边不一致的元件").strip()
    verify = args.get("verify", True)
    if not isinstance(verify, bool):
        verify = True
    max_tiles = max(1, min(12, int(args.get("max_tiles") or 6)))
    overlap = max(0, int(args.get("overlap") or 250))
    tile_size = max(0, int(args.get("tile_size") or 0))
    img, raw, label = load_image(src)
    W, H = img.size
    region = args.get("region")
    if region:
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            raise VisionError("region 必须是 [x1,y1,x2,y2]")
        region, _ = _clamp_list([int(v) for v in region], W, H)
        if region[2] - region[0] < 200 or region[3] - region[1] < 200:
            raise VisionError("region 过小（至少 200x200 像素）")
    else:
        region = [0, 0, W, H]

    tiles = _build_tiles(region, tile_size, overlap, max_tiles)
    tmp_dir = CACHE_DIR / "scan_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    raw_cands = []
    try:
        for i, tbox in enumerate(tiles):
            crop = img.crop(tbox)
            tp = tmp_dir / f"tile_{os.getpid()}_{i}.png"
            crop.save(tp, "PNG")
            log("scan tile", i, tbox)
            try:
                loc = tool_locate_object({"image": str(tp), "target": target})
            finally:
                try:
                    tp.unlink()
                except OSError:
                    pass
            for pr in loc.get("primitives", []):
                if pr.get("box"):
                    raw_cands.append({
                        "tile": str(i),
                        "label": pr.get("label"),
                        "confidence": pr.get("confidence"),
                        "rotation": pr.get("rotation"),
                        "box": [tbox[0] + pr["box"][0], tbox[1] + pr["box"][1], tbox[0] + pr["box"][2], tbox[1] + pr["box"][3]],
                    })
        merged = _merge_candidates(raw_cands)
    finally:
        try:
            for f in tmp_dir.glob(f"tile_{os.getpid()}_*.png"):
                f.unlink()
        except OSError:
            pass

    out = []
    for m in merged:
        entry = {
            "box": m["box"],
            "box_norm": [to_norm([m["box"][0], m["box"][1]], W, H)[0], to_norm([m["box"][0], m["box"][1]], W, H)[1],
                          to_norm([m["box"][2], m["box"][3]], W, H)[0], to_norm([m["box"][2], m["box"][3]], W, H)[1]],
            "confidence": m.get("confidence"),
            "labels": m.get("labels") or [],
            "tiles": m.get("tiles") or [],
            "verified": None,
        }
        if verify:
            pad = max(150, int((m["box"][2] - m["box"][0]) * 0.3))
            b = (max(0, m["box"][0] - pad), max(0, m["box"][1] - pad),
                 min(W, m["box"][2] + pad), min(H, m["box"][3] + pad))
            crop = img.crop(b)
            if max(crop.size) < 600:
                crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
            vp = tmp_dir / f"verify_{os.getpid()}_{len(out)}.png"
            crop.save(vp, "PNG")
            try:
                text = tool_describe_image({"image": str(vp), "question": VERIFY_PROMPT, "detail": "balanced"})
            finally:
                try:
                    vp.unlink()
                except OSError:
                    pass
            entry["verified"] = _parse_verdict(text)
            entry["verified"]["raw"] = text[:400]
        out.append(entry)

    # 排序：歪斜的排前面
    def rank(e):
        v = e.get("verified") or {}
        return 0 if v.get("verdict") == "skewed" else (1 if v.get("verdict") == "unclear" else 2)
    out.sort(key=rank)

    result = {
        "target": target,
        "region": region,
        "image_size": [W, H],
        "tiles": len(tiles),
        "candidates_found": len(raw_cands),
        "candidates_merged": len(out),
        "candidates": out,
    }
    if not out:
        result["note"] = "未找到候选目标，可尝试：换一种 target 描述、缩小 region、增大 max_tiles"
    return result

HANDLERS = {
    "describe_image": tool_describe_image,
    "analyze_image": tool_analyze_image,
    "locate_object": tool_locate_object,
    "ocr_image": tool_ocr_image,
    "annotate_image": tool_annotate_image,
    "crop_image": tool_crop_image,
    "zoom_region": tool_zoom_region,
    "vision_health": tool_vision_health,
    "scan_anomalies": tool_scan_anomalies,
    "compare_images": tool_compare_images,
    "annotate_infer": tool_annotate_infer,
    "compare_infer": tool_compare_infer,
    "reason_graph": tool_reason_graph,
    "screen_capture": tool_screen_capture,
    "screen_info": tool_screen_info,
    "screen_click": tool_screen_click,
    "screen_move": tool_screen_move,
    "screen_drag": tool_screen_drag,
    "screen_scroll": tool_screen_scroll,
    "screen_type": tool_screen_type,
    "screen_key": tool_screen_key,
}

TOOLS = [
    {
        "name": "describe_image",
        "description": "用视觉模型描述图片内容，返回文字描述。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string", "description": "本地图片绝对路径或 http(s) 图片 URL"},
                "question": {"type": "string", "description": "可选的针对性问题"},
                "detail": {"type": "string", "enum": ["brief", "balanced", "detailed"], "description": "细节程度，默认 balanced"},
            },
            "required": ["image"],
        },
    },
    {
        "name": "analyze_image",
        "description": "结构化分析：返回 description + visual_primitives（box/point 坐标与标签）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "question": {"type": "string", "description": "可选的关注点"},
                "format": {"type": "string", "enum": ["generic", "gemini", "qwen"], "description": "primitives 字段风格，默认 generic"},
            },
            "required": ["image"],
        },
    },
    {
        "name": "locate_object",
        "description": "在图片中定位目标对象，返回坐标 primitives（让 LLM 输出坐标）。找不到会返回 count=0。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "target": {"type": "string", "description": "要定位的目标，如：蓝色提交按钮 / 红色圆形 / 报错文字"},
                "coords": {"type": "string", "enum": ["pixel", "norm"], "description": "返回坐标单位：pixel（默认，像素）或 norm（0-1000 归一化）"},
                "refine": {"type": "boolean", "description": "两阶段精修：粗定位后裁切放大二次定位（更准，代价是每个候选多一次视觉调用），默认 false"},
            },
            "required": ["image", "target"],
        },
    },
    {
        "name": "ocr_image",
        "description": "OCR 提取图片中所有文字块，返回 text + bbox（像素与归一化坐标）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "language": {"type": "string", "description": "语言提示，默认 auto"},
            },
            "required": ["image"],
        },
    },
    {
        "name": "annotate_image",
        "description": "在图片上画矩形框/圆点/标签（圈画标记），保存标注图并返回路径。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "items": {"description": "标注项数组或单个对象：[{label, box 或 point, color}]，box=[x1,y1,x2,y2]，point=[x,y]"},
                "coords": {"type": "string", "enum": ["pixel", "norm"], "description": "坐标单位，默认 pixel"},
                "out_path": {"type": "string", "description": "输出路径（必须位于输出目录内），默认自动命名"},
                "style": {"type": "object", "description": "{line_width, font_size, color}"},
            },
            "required": ["image", "items"],
        },
    },
    {
        "name": "crop_image",
        "description": "按坐标裁切图片（可边缘外扩），保存并返回路径与新尺寸。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": "[x1,y1,x2,y2]"},
                "coords": {"type": "string", "enum": ["pixel", "norm"], "description": "坐标单位，默认 pixel"},
                "expand_px": {"type": "integer", "description": "四边外扩像素数，默认 0"},
                "out_path": {"type": "string"},
            },
            "required": ["image", "box"],
        },
    },
    {
        "name": "zoom_region",
        "description": "放大图片指定区域（默认整图 2 倍），保存并返回路径。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": "[x1,y1,x2,y2]，省略则放大整图"},
                "coords": {"type": "string", "enum": ["pixel", "norm"], "description": "坐标单位，默认 pixel"},
                "scale": {"type": "integer", "description": "放大倍数 1-8，默认 2"},
                "out_path": {"type": "string"},
            },
            "required": ["image"],
        },
    },
    {
        "name": "vision_health",
        "description": "检查视觉后端配置与连通性。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "annotate_infer",
        "description": "虚拟标注 + 增强图形推理：把框/点/连线/箭头/圆等标注（不修改原图）注入视觉模型，引导空间关系推理。mode=virtual 用坐标文本注入；mode=overlay 生成半透明叠加图。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "items": {"description": "标注数组或单个对象：[{id?, type: box|point|line|arrow|circle|polygon|bubble, label, color, coords, box/point/from/to/center/radius/points/text/direction}]"},
                "corrections": {"description": "多轮修正操作数组：[{op: add|remove|move|resize|set, id, delta/to/box/point/...}]，基于 items 修正后推理"},
                "auto_boxes": {"description": "自动框选：字符串或目标数组（如 '所有按钮'），内部 locate 后生成紫色框参与推理"},
                "question": {"type": "string", "description": "推理问题，如：框A中的元件是什么？A到B的连线代表什么连接关系？"},
                "mode": {"type": "string", "enum": ["virtual", "overlay"], "description": "virtual=坐标文本注入（默认，原图零修改）；overlay=半透明叠加图"},
                "alpha": {"type": "number", "description": "overlay 模式叠加透明度 (0,1]，默认 0.35"},
                "detail": {"type": "string", "enum": ["brief", "balanced", "detailed"], "description": "细节程度"},
            },
            "required": ["image", "items", "question"],
        },
    },
    {
        "name": "screen_capture",
        "description": "截屏（全屏或指定区域），保存 PNG 并返回路径。配合 locate_object/describe 实现「看屏幕」。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "region": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": "可选 [x1,y1,x2,y2] 屏幕坐标区域，默认全屏"},
                "out_path": {"type": "string"},
            },
        },
    },
    {
        "name": "screen_info",
        "description": "屏幕信息：分辨率、DPI、电脑控制开关状态。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "screen_click",
        "description": "鼠标点击（需 VISION_ALLOW_SCREEN_CONTROL=1）。坐标通常来自 locate_object 对截图的定位结果。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "默认 left"},
                "double": {"type": "boolean", "description": "是否双击，默认 false"},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "screen_move",
        "description": "仅移动鼠标光标（需 VISION_ALLOW_SCREEN_CONTROL=1）。",
        "inputSchema": {
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["x", "y"],
        },
    },
    {
        "name": "screen_drag",
        "description": "鼠标拖拽（需 VISION_ALLOW_SCREEN_CONTROL=1）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "x1": {"type": "number"}, "y1": {"type": "number"}, "x2": {"type": "number"}, "y2": {"type": "number"},
                "duration": {"type": "number", "description": "拖拽时长秒，默认 0.2"},
                "button": {"type": "string", "enum": ["left", "right", "middle"]},
            },
            "required": ["x1", "y1", "x2", "y2"],
        },
    },
    {
        "name": "screen_scroll",
        "description": "滚轮滚动（需 VISION_ALLOW_SCREEN_CONTROL=1）。正数向上，负数向下。",
        "inputSchema": {
            "type": "object",
            "properties": {"delta": {"type": "number", "description": "滚动格数（正=上，负=下）"}},
            "required": ["delta"],
        },
    },
    {
        "name": "screen_type",
        "description": "键盘输入文本（需 VISION_ALLOW_SCREEN_CONTROL=1）。ASCII 直接按键；中文等经剪贴板粘贴。",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "screen_key",
        "description": "按键或组合键（需 VISION_ALLOW_SCREEN_CONTROL=1）。如 enter / tab / ctrl+c / alt+tab。",
        "inputSchema": {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "单个按键或 ctrl+shift+key 组合"}},
            "required": ["key"],
        },
    },
    {
        "name": "compare_infer",
        "description": "多图联合推理（2-4 张）：每张图可带独立标注（items_per_image），联合对比/推理关系（差异、因果、时序、整体结论）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "images": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4},
                "items_per_image": {"type": "object", "description": "{图索引(0开始): 标注数组}，每图可选"},
                "question": {"type": "string", "description": "联合推理问题"},
                "mode": {"type": "string", "enum": ["virtual", "overlay"], "description": "默认 virtual"},
                "alpha": {"type": "number", "description": "overlay 透明度，默认 0.35"},
            },
            "required": ["images", "question"],
        },
    },
    {
        "name": "reason_graph",
        "description": "交互式图形推理协议：原语(locate/measure) → 语义(semantic/hypothesis) → 标注(annotate/verify) 多轮循环。session 跨轮传递状态。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "question": {"type": "string", "description": "总体推理目标（各轮可带）"},
                "session": {"type": "object", "description": "上一轮返回的 session（primitives/annotations/semantics/hypotheses），第一轮可省略"},
                "step": {"type": "object", "description": "本轮动作：{type: locate|measure|annotate|semantic|hypothesis|verify|next, ...}（locate: target/refine；measure: measure=distance|angle|area + refs；verify: id；hypothesis/semantic: text）"},
            },
            "required": ["image", "step"],
        },
    },
    {
        "name": "compare_images",
        "description": "多图对比分析（2-4 张）：A/B 截图对比、设计稿一致性、多帧分析，返回逐项对比结果。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "images": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4, "description": "2-4 张本地图片路径或 http(s) URL"},
                "question": {"type": "string", "description": "对比重点，如：UI 有什么变化"},
                "detail": {"type": "string", "enum": ["brief", "balanced", "detailed"], "description": "细节程度，默认 balanced"},
            },
            "required": ["images"],
        },
    },
    {
        "name": "scan_anomalies",
        "description": "自动扫描图片中的异常/歪斜元件：把区域切成带重叠的块逐块定位候选，再从原图高清裁切逐个验证，输出带置信度与角度/丝印的报告。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "target": {"type": "string", "description": "要找的异常特征描述，默认：摆放歪斜、方向与周边不一致的元件"},
                "region": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": "可选：限定扫描区域 [x1,y1,x2,y2] 像素，默认全图"},
                "verify": {"type": "boolean", "description": "是否自动高清验证候选，默认 true"},
                "max_tiles": {"type": "integer", "description": "切块数上限（1-12），默认 6"},
                "overlap": {"type": "integer", "description": "切块重叠像素，默认 250"},
                "tile_size": {"type": "integer", "description": "切块边长（像素），默认自动"},
            },
            "required": ["image"],
        },
    },
]

# ----------------------------- MCP stdio 协议 -----------------------------

# 双协议支持：标准 Content-Length 帧（Codex/Claude 等）与 JSONL 行传输（Hana 等）。
# 检测到 JSONL 输入后自动切换，之后所有响应都以 JSONL 输出。
JSONL_MODE = {"on": False}


def write_frame(stream, obj):
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    if JSONL_MODE["on"]:
        stream.write(payload + b"\n")
    else:
        stream.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8"))
        stream.write(payload)
    stream.flush()


def read_frame(stream):
    first = stream.readline()
    if not first:
        return None
    stripped = first.strip()
    if stripped.startswith(b"{"):
        # JSONL 模式：一行一个 JSON 消息，自动切换输出协议
        JSONL_MODE["on"] = True
        try:
            return json.loads(stripped.decode("utf-8"))
        except json.JSONDecodeError:
            return None
    # 标准 Content-Length 帧：首行是头，继续读完剩余头
    headers = {}
    if b":" in first:
        k, v = first.split(b":", 1)
        headers[k.strip().lower()] = v.strip()
    while True:
        line = stream.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        if b":" in line:
            k, v = line.split(b":", 1)
            headers[k.strip().lower()] = v.strip()
    try:
        length = int(headers.get(b"content-length", b"0"))
    except ValueError:
        length = 0
    if length <= 0:
        return None
    payload = stream.read(length)
    try:
        return json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError:
        return None

def _text_result(rid, text, is_error=False):
    return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": text}], "isError": is_error}}

def _error(rid, code, message):
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}

def handle_message(msg):
    if not isinstance(msg, dict):
        return None
    method = msg.get("method")
    rid = msg.get("id")
    if method == "initialize":
        req_pv = (msg.get("params") or {}).get("protocolVersion", "2025-06-18")
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "protocolVersion": req_pv,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "vision-primitives-mcp", "version": "1.9.1"},
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        tool = next((t for t in TOOLS if t["name"] == name), None)
        if tool is None:
            return _error(rid, -32601, f"未知工具: {name}")
        required = tool["inputSchema"].get("required", [])
        missing = [r for r in required if r not in args or args[r] in (None, "")]
        if missing:
            return _error(rid, -32602, f"缺少必需参数: {', '.join(missing)}")
        handler = HANDLERS.get(name)
        if handler is None:
            return _error(rid, -32601, f"工具未实现: {name}")
        try:
            result = handler(args)
            if isinstance(result, str):
                return _text_result(rid, result)
            return _text_result(rid, json.dumps(result, ensure_ascii=False, indent=2))
        except McpParamError as e:
            return _error(rid, -32602, f"Invalid params: {e}")
        except VisionError as e:
            return _text_result(rid, f"错误: {e}", is_error=True)
        except Exception as e:
            log("tool crash:", name, e)
            return _text_result(rid, f"内部错误: {e}", is_error=True)
    if rid is not None:
        return _error(rid, -32601, f"未知方法: {method}")
    return None

def main():
    if "--health" in sys.argv:
        print(json.dumps(tool_vision_health(), ensure_ascii=False, indent=2))
        return
    while True:
        msg = read_frame(sys.stdin.buffer)
        if msg is None:
            break
        resp = handle_message(msg)
        if resp is not None:
            write_frame(sys.stdout.buffer, resp)

if __name__ == "__main__":
    main()
