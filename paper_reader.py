#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""长文档视觉阅读工作流：任意来源（arXiv / URL / PDF / 本地图片）-> 多屏截图 -> 分屏 scratch_think 理解

用法:
  python paper_reader.py 2509.21552                    # arXiv ID
  python paper_reader.py https://arxiv.org/abs/2509.21552
  python paper_reader.py https://example.com/paper.html  # 任意 URL
  python paper_reader.py paper.pdf                     # 本地 PDF（浏览器渲染截图，零额外依赖）
  python paper_reader.py image.png                    # 本地图片（跳过截图）

参数:
  --screens N     截图屏数（默认 3：标题摘要/方法/实验分布），1 表示只截首屏
  --rounds N      scratch_think 每屏最大轮数（默认 3）
  --question      自定义问题（--screens 1 时使用）
  --model / --api-base / --api-key   视觉后端
  --outdir        输出目录（默认 generated/）
"""
import argparse
import io
import os
import re
import sys
import time
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
# 注意：vision_primitives_mcp 在 import 时读取环境变量（API 后端），
# 因此必须在设置 os.environ 之后再 import（见 main() 内）。


def normalize_arxiv(id_or_url):
    """解析 arXiv ID/URL，返回 (base_id, full_id)。full_id 带最新版本号。"""
    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", id_or_url)
    if not m:
        return None, None
    base, ver = m.group(1), m.group(2)
    if ver:
        return base, base + ver
    try:
        r = urllib.request.urlopen(
            f"http://export.arxiv.org/api/query?id_list={base}", timeout=20
        )
        text = r.read().decode("utf-8")
        m2 = re.search(r"<id>http://arxiv.org/abs/([\d.]+v\d+)</id>", text)
        if m2:
            return base, m2.group(1)
    except Exception as e:
        print(f"[warn] arXiv 版本查询失败: {e}")
    return base, base


def resolve_target(target, outdir):
    """把输入归一化为可截图 URL 或本地图片路径。返回 (kind, path_or_url, label)。"""
    target = target.strip()
    if os.path.isfile(target):
        ext = os.path.splitext(target)[1].lower()
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
            return "image", os.path.abspath(target), target
        if ext == ".pdf":
            return "pdf", "file:///" + os.path.abspath(target).replace("\\", "/"), target
        raise SystemExit(f"不支持的本地文件类型: {ext}")
    base, full = normalize_arxiv(target)
    if full:
        return "url", f"https://arxiv.org/abs/{full}", f"arXiv:{full}"
    if target.startswith(("http://", "https://")):
        return "url", target, target
    raise SystemExit(f"无法识别输入: {target}（支持 arXiv ID/URL、任意 URL、本地 PDF/图片）")


def screenshots_web(page, url, out_path, n_screens, viewport=(1280, 900)):
    """打开 URL，滚动截 n 屏（均匀分布），返回截图路径列表。"""
    try:
        page.goto(url, wait_until="networkidle", timeout=60000)
    except Exception:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(1500)
    w, h = viewport
    paths = []
    if n_screens <= 1:
        p = f"{out_path}_s1.png"
        page.screenshot(path=p, clip={"x": 0, "y": 0, "width": w, "height": h})
        paths.append(p)
        return paths
    # 先测页面总高，决定滚动分布
    total = page.evaluate("() => document.body.scrollHeight") or h * 4
    for i in range(n_screens):
        p = f"{out_path}_s{i + 1}.png"
        if i == 0:
            y = 0
        else:
            y = int(total * i / n_screens)
            page.evaluate(f"window.scrollTo(0, {y})")
            page.wait_for_timeout(800)
        page.screenshot(path=p, clip={"x": 0, "y": 0, "width": w, "height": h})
        paths.append(p)
    return paths


def capture(target, outdir, n_screens):
    """按输入类型截图。返回 (label, [png 路径])。"""
    kind, loc, label = resolve_target(target, outdir)
    if kind == "image":
        return label, [loc]
    os.makedirs(outdir, exist_ok=True)
    base = re.sub(r"[^0-9a-zA-Z]", "_", label)[:40]
    out_path = os.path.join(outdir, base)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        try:
            paths = screenshots_web(page, loc, out_path, n_screens)
        finally:
            browser.close()
    return label, paths


def understand(screen, question, rounds, outdir):
    """单屏 scratch_think 理解。"""
    r = vb.tool_scratch_think(
        {
            "image": screen,
            "question": question,
            "max_rounds": rounds,
            "out_path": os.path.join(outdir, "scratch_" + os.path.basename(screen)),
        }
    )
    return r


def main():
    ap = argparse.ArgumentParser(description="长文档视觉阅读：多屏截图 + 分屏 scratch_think 理解")
    ap.add_argument("target", help="arXiv ID/URL、任意 URL、本地 PDF 或图片")
    ap.add_argument("--screens", type=int, default=3, help="截图屏数（默认 3；1 表示只截首屏）")
    ap.add_argument("--rounds", type=int, default=3, help="每屏 scratch_think 最大轮数")
    ap.add_argument("--question", default=None, help="自定义问题（--screens 1 时使用；多屏时按阶段提问）")
    ap.add_argument("--model", default=os.environ.get("VISION_MODEL", "qwen/qwen2.5-vl-7b"))
    ap.add_argument("--api-base", default=os.environ.get("VISION_API_BASE", "http://198.18.0.1:1234/v1"))
    ap.add_argument("--api-key", default=os.environ.get("VISION_API_KEY", "lm-studio"))
    ap.add_argument("--outdir", default="generated")
    args = ap.parse_args()

    os.environ["VISION_MODEL"] = args.model
    os.environ["VISION_API_BASE"] = args.api_base
    os.environ["VISION_API_KEY"] = args.api_key
    os.environ["VISION_CACHE"] = "0"

    # 环境变量设置后再 import（模块 import 时锁定后端）
    vpm_dir = os.environ.get("VPM_DIR") or os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, vpm_dir)
    global vb
    import vision_primitives_mcp as vb

    print(f"[1/3] 截图: {args.target}")
    label, screens = capture(args.target, args.outdir, args.screens)
    print(f"      {label} -> {len(screens)} 屏")

    print("[2/3] 分屏理解...")
    stage_questions = [
        "这一屏是文档的标题/摘要部分。论文的标题、核心方法、主要贡献是什么？",
        "这一屏是文档的中段（方法/正文）。这里展示了什么技术细节、方法流程或图表？",
        "这一屏是文档的后段（实验/结论）。主要实验结果、指标数字和结论是什么？",
        "这一屏的内容要点是什么？与论文核心问题相关的信息有哪些？",
    ]
    answers = []
    for i, s in enumerate(screens):
        if args.question:
            q = args.question
        else:
            q = stage_questions[min(i, len(stage_questions) - 1)]
        print(f"  屏 {i + 1}/{len(screens)}: {os.path.basename(s)}")
        t0 = time.time()
        r = understand(s, q, args.rounds, args.outdir)
        ans = r.get("answer") or "(无回答)"
        answers.append(ans)
        print(f"    {time.time() - t0:.1f}s | {r.get('rounds')} 轮 | {ans[:80]}")

    print("[3/3] 结果:")
    print("=" * 60)
    for i, a in enumerate(answers):
        print(f"--- 屏 {i + 1} ---")
        print(a)
    print("=" * 60)


if __name__ == "__main__":
    main()
