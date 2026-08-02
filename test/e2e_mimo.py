#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实端到端测试：小米 MiMo V2.5 全闭环（describe -> locate -> crop -> annotate -> zoom -> OCR）。
运行：set VISION_API_KEY=sk-... && python test/e2e_mimo.py
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("VISION_OUTPUT_DIR", str(ROOT / "generated"))
os.environ.setdefault("VISION_API_BASE", "https://api.xiaomimimo.com/v1")
os.environ.setdefault("VISION_MODEL", "mimo-v2.5")
os.environ.setdefault("VISION_CACHE", "0")  # e2e 不缓存

import vision_primitives_mcp as vb  # noqa: E402

SAMPLE = ROOT / "sample.png"


def show(title, obj):
    print(f"\n===== {title} =====")
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main():
    key = os.environ.get("VISION_API_KEY", "")
    if not key:
        print("需要 VISION_API_KEY 环境变量（小米 MiMo API key）")
        sys.exit(2)
    health = vb.tool_vision_health()
    show("health", health)
    if not health.get("ok"):
        print("health 检查失败，中止"); sys.exit(1)

    img = str(SAMPLE)

    desc = vb.tool_describe_image({"image": img, "question": "图里有什么？"})
    print("\n===== describe =====")
    print(desc)

    loc = vb.tool_locate_object({"image": img, "target": "蓝色的提交按钮"})
    show("locate 蓝色提交按钮", loc)

    loc2 = vb.tool_locate_object({"image": img, "target": "红色的圆形"})
    show("locate 红色圆形", loc2)

    ocr = vb.tool_ocr_image({"image": img})
    show("ocr", ocr)

    # crop：用 locate 的按钮框（pixel）裁切
    if loc.get("count"):
        box = loc["primitives"][0].get("box")
        if box:
            cr = vb.tool_crop_image({"image": img, "box": box, "out_path": str(ROOT / "generated" / "e2e_crop_button.png")})
            show("crop 按钮", cr)

    # annotate：把两个目标圈出来
    items = []
    for i, p in enumerate(loc.get("primitives", [])):
        if p.get("box"):
            items.append({"label": f"button{i+1}", "box": p["box"], "color": "#ff3b30"})
    for i, p in enumerate(loc2.get("primitives", [])):
        if p.get("box"):
            items.append({"label": f"circle{i+1}", "box": p["box"], "color": "#00b0f0"})
    if items:
        an = vb.tool_annotate_image({"image": img, "items": items, "out_path": str(ROOT / "generated" / "e2e_annotated.png")})
        show("annotate", an)

    zm = vb.tool_zoom_region({"image": img, "box": [120, 180, 300, 240], "scale": 2, "out_path": str(ROOT / "generated" / "e2e_zoom.png")})
    show("zoom 按钮区域 2x", zm)

    print("\n===== e2e 完成 =====")


if __name__ == "__main__":
    main()