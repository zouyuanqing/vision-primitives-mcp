# Vision Primitives MCP — 给纯文本 LLM 装上眼睛的视觉原语 MCP 服务器

[简体中文](./README.md) | [English](./README.en.md)

让纯文本模型（DeepSeek / Codex / 任意 MCP 客户端）通过 **30 个 MCP 工具**获得完整视觉能力：**描述 → 定位（坐标）→ OCR → 标注 → 裁切/放大 → 异常扫描 → UI 结构化 → 多轮推理 → 电脑操控**。视觉后端可切换（MiMo V2.5 云端 / LM Studio 本地 Qwen2.5-VL 等），单文件 Python，核心仅依赖 Pillow。

> **项目定位**：通用视觉推理桥接层——文本模型 + **任意 VLM** 的通用视觉工作流，本地隐私 + 单文件轻量。**高泛化优先**：核心能力（定位/切块/放大/读取）全部是纯 VLM + PIL，专用检测器（YOLO/CRAFT）只是可插拔加速器，有则加速、无则照跑。**不**与 UI-TARS / CogAgent 等端到端 GUI 模型竞争（它们有专门训练）；核心差异化是**无 grounding 模型的兜底定位**与**任何模型可用的多轮视觉推理**。

```
纯文本模型（推理与决策）
    │  MCP 协议（stdio, JSON-RPC 2.0）
    ▼
vision_primitives_mcp.py（单文件，30 工具，186 项测试）
    │  OpenAI 兼容 API
    ▼
MiMo V2.5（云端）/ Qwen2.5-VL-7B（本地 LM Studio）/ 任意视觉 VLM
```

## 实测基准（2026-08-02，程序化 ground truth）

测试图（900×600，元素位置为已知真值）：红圆中心 (150,140)、绿三角中心 (740,417)：

![测试基准图](docs/bench-test-image.png)

### 定位对比（三模型 × 多模式）

![定位实测对比](docs/bench-locate-compare.png)

![Qwen2.5-VL 对比](docs/bench-locate-compare-v25.png)

**定位矩阵**（同一基准图，像素级验证）：

| 模型 | locate 红圆 | locate 绿三角 | som 红圆 | som 绿三角 | 单次调用 | ui_parse | ui_refine |
|---|---|---|---|---|---|---|---|
| MiMo V2.5（云端） | 10-64px（波动） | 79-97px（波动） | 33-82px（波动） | 13-123px（波动） | 15-25s | 21.5s | — |
| MiMo + som-cv（兜底管线） | **0px** | **4px** | — | — | 10-12s | — | — |
| Qwen3-VL-8B（本地） | 90px | 164px | 77px | 123px | 10-30s | 29.4s | >357s |
| GLM-4v-flash（智谱云） | 89px | 84px | 191px | 145px | 1.5-3.3s | — | — |
| **Qwen2.5-VL-7B（本地）** | **28px** | **27px** | **15px** | 123px | **1.3-1.7s** | **12.4s** | **12.6s** |

**能力矩阵**（describe / OCR / 兜底定位 / 文本锚定 / 多轮推理）：

| 模型 | describe | OCR（prompt 修复后） | som-cv（颜色目标） | ui_locate 文本锚定 | scratch 论文推理 |
|---|---|---|---|---|---|
| MiMo V2.5（云端） | 4.7s ✓ | 8.9s，4/4 块 | **0px**（3.2s） | 8.6s ✓ | 45.7s 3轮（漏答图趋势） |
| **Qwen2.5-VL-7B（本地）** | 1.1s ✓ | 3.5-4.6s，4/4 块 | **0px** | 12.8s ✓ | **9.6s 1轮全对** |
| GLM-4v-flash（智谱云） | 1.5s ✓ | 2.0s，3/4 块（max_tokens≤1024 限制） | 53px（格子选偏） | — | — |
| Qwen3-VL-8B（本地） | ~10s | 23s，4/4 块 | 0px | — | — |

**消融：定位误差根因是"分辨率稀释"**——输入 0.5x→208px / 2x→70px / 目标单独裁切→**0-3px**。整图定位差不是模型能力问题，是每个对象分到的视觉 token 太少；**"粗定位 → 裁切 → 精定位"的管线是正解**（`som_locate` 递归、`final="cv"`、`text_zoom` 网格均基于此）。

## 工具一览（30 个）

| 分类 | 工具 | 作用 |
|---|---|---|
| 基础视觉 | `describe_image` / `analyze_image` | 描述 / 结构化分析（含坐标原语） |
| 定位 | `locate_object` | 坐标输出定位，`refine` 两阶段精修 |
| | `som_locate` | SoM 编号网格递归定位（`final`: box/number/cv 三模式） |
| | `cursor_locate` | 移动光标 + 视觉反馈循环定位（GUI-Cursor 范式，建议云端强模型） |
| | `cv_locate` | CV 兜底：颜色分割/模板匹配，像素级（numpy 加速 285x） |
| 文本 | `ocr_image` | 逐文本块 OCR，带 bbox |
| | `text_detect` | CRAFT 文本区域检测（本地 onnx，可选加速器） |
| | `text_zoom` | **程序化切块放大读取（高泛化默认路径）**：网格逐块放大 VLM 精读 |
| UI 结构化 | `ui_parse` / `ui_locate` / `ui_refine` | YOLO 检测 + 文本锚定 + VLM 语义编辑检测框 |
| 图像处理 | `annotate_image` / `crop_image` / `zoom_region` | 标注 / 裁切 / 放大 |
| 高级推理 | `scratch_think` | **视觉草稿纸**：跨轮层栈 + 自适应裁切放大（图文论文理解） |
| | `compare_images` / `compare_infer` / `reason_graph` / `annotate_infer` | 多图对比 / 联合推理 / 图形推理协议 / 虚拟标注 |
| 扫描 | `scan_anomalies` | 自动异常扫描：切块候选 → 高清逐点验证（PCB 实战） |
| 电脑控制 | `screen_capture` / `screen_info` / `screen_click` / `screen_move` / `screen_drag` / `screen_scroll` / `screen_type` / `screen_key` | 截屏 + 鼠标键盘（Windows-only，安全开关默认关） |
| 诊断 | `vision_health` | 后端配置与连通性 |

配套脚本 [paper_reader.py](paper_reader.py)：arXiv/URL/PDF/图片 → 多屏截图 → 分屏 scratch_think → 结构化摘要。

## 核心方法论

1. **对抗分辨率稀释**：一切定位精度的基础。`som_locate` 编号递归（模型只选格子）、`final="cv"`（VLM 收敛 + CV 像素级）、`text_zoom` 网格切块（插图小字）——三个工具共享"局部放大"原理，只是块的产生方式不同（模型决策 / 检测器 / 网格）
2. **文本锚定**：UI 操作指令几乎总带文字——OCR 定位文字 → 控件框。比让模型猜坐标可靠一个量级
3. **检测器可插拔**：YOLO（icon_detect）/ CRAFT（text）都是"有则加速、无则照跑"——高泛化路径（纯 VLM）始终可用
4. **视觉草稿纸**（scratch_think）：无 grounding 模型缺"视觉工作记忆"——层栈把中间状态渲染回图，模型每轮可见可编辑；坐标链保证裁切放大后一切可映射回原图
5. **几何判定程序化**：重叠合并、坐标换算、收敛检测这类确定性问题交给代码，VLM 只做语义判断（实测 VLM 合并会把相邻独立元素误并）

## 快速上手

```toml
# ~/.codex/config.toml
[mcp_servers.vision-primitives]
command = "python"
args = ['/path/to/vision-primitives-mcp/vision_primitives_mcp.py']
startup_timeout_sec = 60

[mcp_servers.vision-primitives.env]
VISION_API_BASE = "https://api.xiaomimimo.com/v1"      # 或本地 http://127.0.0.1:1234/v1
VISION_API_KEY = "你的MiMo密钥"                          # 本地 LM Studio 用任意占位
VISION_MODEL = "mimo-v2.5"                              # 或 qwen/qwen2.5-vl-7b
VISION_OUTPUT_DIR = '/path/to/vision-primitives-mcp/generated'
```

**模型选型**：定位/审查/多轮推理首选 **Qwen2.5-VL-7B**（grounding 专才 + 非思考型，1.3-1.7s/次）；MiMo 通用但定位波动大，需 `VISION_SAMPLES=3`；Qwen3-VL-8B 仅描述/OCR 场景考虑；**GLM-4v-flash**（智谱）速度优秀（1.5-3.3s）适合 describe/OCR/粗定位，定位精度弱（84-191px），且 `max_tokens` 上限 1024（需 `VISION_MAX_TOKENS=1024`）

**可选增强**（全部"有则加速、无则照跑"）：

| 增强 | 放置 | 收益 |
|---|---|---|
| numpy | `pip install numpy` | 模板匹配 285x 加速 |
| YOLO UI 检测 | `models/icon_detect.pt`（[下载](https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/model.pt)，39.7MB） | ui_parse 像素级 UI 元素检测 |
| CRAFT 文本检测 | `models/craft_text.onnx`（[下载](https://huggingface.co/KvaytG/craft-mlt-25k-onnx/resolve/main/craft.onnx)，79MB） | text_detect/text_zoom 检测框加速 |
| playwright | `pip install playwright` | paper_reader 截图 |

## 已知边界（诚实声明）

- **整图定位受分辨率稀释**（20-164px），用 som 递归 / final="cv" / text_zoom 对抗
- **极小小字（<10px）**：text_zoom 网格放大可读（实测 False Stop/Class Settings），但个别字符仍可能误读——模型识别极限
- **`cursor_locate` 本地模型不可用**（483/100px），仅云端强模型；**`ui_refine` VLM 审查建议云端**（本地 8B 数分钟）
- **OCR 对 prompt 长度敏感**（已修复，保持简短指令）；表格数据首选 DOM 提取，视觉 OCR 兜底（90-120s）
- **平台**：屏幕控制 8 工具 Windows-only；**安全**：URL 图片有 SSRF 防护（私网拦截）与解压炸弹限制（URL 源 50MP）
- **延迟**：MiMo 15-25s/次、本地 Qwen2.5-VL 1.3-1.7s/次、多轮推理按轮倍增

## 版本历史（精简）

- **v1.14（2026-08-02）**：text_detect（CRAFT 可选）+ text_zoom（程序化切块，高泛化默认路径，实测读出 <10px 图内标注）；30 工具 / 186 测试
- **v1.13（2026-08-02）**：scratch_think 视觉草稿纸（层栈 + 自适应 zoom + 坐标链）；paper_reader.py 论文阅读工作流；OCR prompt 修复（召回 1→4 块）；extract_json 截断容错
- **v1.12（2026-08-02）**：ui_refine 检测框语义编辑（VLM 删除误检/标注 + 程序化几何合并）
- **v1.11（2026-08-02）**：ui_parse / ui_locate（UI 结构化 + 文本锚定）+ YOLO 可选检测 + SSRF/解压炸弹防护 + numpy 模板匹配
- **v1.10（2026-08-02）**：som_locate（SoM 编号递归，final=box/number/cv）+ cursor_locate + cv_locate
- **v1.9（2026-08-02）**：Computer Use（8 个屏幕控制工具，安全开关）
- **v1.7-1.8**：refine 两阶段精修（70→20px）、reason_graph、annotate_infer、坐标格式实验（像素最稳）
- **v1.1-1.6**：scan_anomalies（PCB 实战）、compare_images、中文 OCR 修复
- **v1.9.1 / v1.13.3 修复记录**：MCP 响应帧 jsonrpc/id（严格客户端兼容）、长 JSON 截断容错

## 测试与安全

```bash
python test\run_tests.py   # 186 项 mock 测试，不依赖真实 key
python test\e2e_mimo.py    # 真实端到端（需要 VISION_API_KEY）
```

- 输入图片只读，上传仅发往配置的后端；`out_path` 强制限定输出目录
- URL 图片：SSRF 防护（私网/链路本地/元数据拦截，回环放行）+ 解压后 50MP 限制
- API key 仅存本地配置；屏幕控制类工具默认拒绝（`VISION_ALLOW_SCREEN_CONTROL=1` 启用）

**仓库**：[github.com/zouyuanqing/vision-primitives-mcp](https://github.com/zouyuanqing/vision-primitives-mcp)
