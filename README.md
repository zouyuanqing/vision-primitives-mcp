# Vision Primitives MCP — 给纯文本 LLM 装上眼睛的视觉原语 MCP 服务器

[简体中文](./README.md) | [English](./README.en.md)

让纯文本模型（DeepSeek / Codex / 任意 MCP 客户端）通过 27 个 MCP 工具获得完整视觉能力：**描述 → 定位（坐标）→ OCR → 标注 → 裁切/放大 → 异常扫描 → 电脑操控**。视觉后端可切换（小米 MiMo V2.5 云端 / LM Studio 本地 Qwen3-VL 等），单文件 Python，核心仅依赖 Pillow；numpy 可选（模板匹配 285x 加速）；YOLO 检测器可选（`models/icon_detect.pt` 放置后自动启用）。

```
纯文本模型（推理与决策）
    │  MCP 协议（stdio, JSON-RPC 2.0）
    ▼
vision_primitives_mcp.py（单文件，24 工具，零第三方运行时依赖）
    │  OpenAI 兼容 API
    ▼
MiMo V2.5（云端）/ Qwen3-VL-8B（本地 LM Studio）/ 任意视觉 VLM
```

## 实测基准（2026-08-02，程序化 ground truth 验证）

测试图（900×600，元素位置为已知真值）：红圆中心 (150,140)、绿三角中心 (740,417)：

![测试基准图](docs/bench-test-image.png)

### 整图定位对比（MiMo V2.5 vs 本地 Qwen3-VL-8B，各模式偏差）

![定位实测对比](docs/bench-locate-compare.png)

| 方法 | 红圆 | 绿三角 | 备注 |
|---|---|---|---|
| `locate_object`（坐标输出） | MiMo 64px / Qwen 89px | MiMo 97px / Qwen 164px | 通用 VLM 直接输出坐标，受视觉 token 粒度限制 |
| `som_locate`（编号引用，3×3×2 轮） | Qwen **34px** | Qwen 123px | 把坐标问题变成编号问题，对无 grounding 模型更友好 |
| `som_locate` 4×4 网格 | Qwen **16px** | 206px（首轮选偏锁死） | 更细网格提升，但依赖首轮正确性 |
| `cursor_locate`（交互搜索） | Qwen 100px（5 步 80s） | — | 本地小模型对相对偏移估计有限，留给云端强模型 |
| `som_locate final="cv"` | Qwen **0px**（<1s） | Qwen **4px**（<1s） | VLM 收敛 + CV 精定位，像素级且纯本地 |

### 消融测试：定位偏差的根因是"分辨率稀释"

![消融测试](docs/bench-ablation.png)

- 输入分辨率 0.5x / 1x / 2x → 偏差 208 / 89 / 70px：**精度随分辨率单调提升**
- 目标单独裁切后定位：红圆 **0px**、绿三角 **3px**：模型 grounding 能力完全在线
- 结论：整图定位差 = 全局分辨率稀释（每个对象分到的视觉 token 太少），**对抗稀释的正确方法是"粗定位 → 裁切 → 精定位"**，这正是 `som_locate` 递归与 `final="cv"` 的原理

### SoM 编号标记示例

![SoM 编号标记](docs/bench-som-marks.png)

每轮在图上叠加编号标记，模型只回答编号，随后裁切 2x 放大进入下一轮。

### CV 备选方案：简单目标的像素级定位

![CV 定位结果](docs/bench-cv-result.png)

`cv_locate`（颜色分割 + 连通域质心）：纯本地、零 API 调用、实测 0-4px。**仅适用简单目标**（纯色 UI 点击元素、几何图形、固定模板），泛化有限，通用目标请用 VLM 定位。

## 工具一览（27 个）

| 分类 | 工具 | 作用 |
|---|---|---|
| 基础视觉 | `describe_image` / `analyze_image` | 描述 / 结构化分析（含坐标原语） |
| 定位 | `locate_object` | 坐标输出定位，`refine` 两阶段精修 |
| | `som_locate` | **SoM 编号网格递归定位**（`final`: box/number/cv 三模式） |
| | `cursor_locate` | 移动光标 + 视觉反馈循环定位（GUI-Cursor 范式） |
| | `cv_locate` | **CV 备选**：颜色分割/模板匹配，像素级，仅简单目标 |
| `ui_parse` / `ui_locate` / `ui_refine` | **UI 结构化解析 / 文本锚定定位 / 检测框语义编辑**：YOLO 像素级检测 + 文本锚定 + VLM 审查修正（删除误检/语义标注/程序化合并） |
| 文字 | `ocr_image` | 逐文本块 OCR，带 bbox |
| 图像处理 | `annotate_image` / `crop_image` / `zoom_region` | 标注 / 裁切 / 放大 |
| 高级推理 | `compare_images` / `compare_infer` / `reason_graph` / `annotate_infer` | 多图对比 / 联合推理 / 交互式图形推理 / 虚拟标注推理 |
| 扫描 | `scan_anomalies` | 自动异常扫描：切块候选 → 高清逐点验证（PCB 元件实战） |
| 电脑控制 | `screen_capture` / `screen_info` / `screen_click` / `screen_move` / `screen_drag` / `screen_scroll` / `screen_type` / `screen_key` | 截屏 + 鼠标键盘（安全开关默认关） |
| 诊断 | `vision_health` | 后端配置与连通性检查 |

坐标系统：`pixel`（默认，实测最准）或 `norm`（0-1000 归一化），越界自动钳制。

## 定位方法学：四种模式与 VLM 选型

| 模式 | 原理 | 适用 |
|---|---|---|
| `locate_object` | 模型直接输出坐标 | 通用兜底；快速粗定位 |
| `som_locate` | 编号引用 + 递归裁切，对抗分辨率稀释 | **通用推荐**（`final="box"` 末轮局部图直接输出框） |
| `cursor_locate` | 相对偏移 + 视觉反馈逼近 | 需要交互式收敛时（建议云端强模型） |
| `cv_locate` | 颜色分割 / 模板匹配，纯本地 | 简单目标备选：纯色 UI、几何、固定模板 |
| `ui_parse` / `ui_locate` | **结构化解析 + 文本锚定**（OCR 文本匹配控件框）；可选 YOLO 检测器（OmniParser icon_detect） | **UI 点击类首选**：检测框像素级（实测红圆 4px 内），VLM 只做编号语义选择 |

**VLM 选型建议**：定位场景优先 grounding 训练的模型（业界证据：GUI-Actor / SE-GUI / GUI-Cursor 均指出文本坐标生成的"空间-语义对齐弱"问题）：

- **首选 Qwen-2.5-VL-7B**（本地 LM Studio）：专门 grounding 训练（RefCOCO 93.7%），实测整图 locate 27-28px / som 15px，单次调用 1.3-1.7s（非思考型，快 10-20 倍）
- **备选 Qwen-3-VL-8B**：实测整图 locate 90-164px、慢 10-20 倍（思考型且 grounding 未继承），仅描述/OCR 场景可考虑
- 不推荐：Qwen-3.5-9b（无 grounding 训练，实测 210px）、Gemma4-E4B（无 grounding 记录）
- 云端 MiMo V2.5：通用描述/OCR 优秀，定位需配合 `som_locate`；ui_refine 审查建议云端强模型
- **分工建议（实测）**：定位/审查用 Qwen2.5-VL-7B；**OCR 召回率模型差异大**（Qwen2.5-VL 1.2s 但只回 1 块，Qwen3-VL/MiMo 全量返回）——文本锚定场景建议 OCR 用 Qwen3-VL/MiMo；`cursor_locate` 两种本地模型均不可用（483/100px），仅云端强模型
- **实测基准（2026-08-02，Qwen2.5-VL-7B vs Qwen3-VL-8B）**：locate 红圆 28 vs 90px、绿三角 27 vs 164px；som 红圆 15 vs 77px；单次调用 1.5 vs 20s；ui_parse 12.4 vs 29.4s；单问题审查 0.6 vs 4.2s

## 快速上手

**UI 检测器（可选）**：下载 [OmniParser icon_detect](https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/model.pt)（39.7MB，MIT）放入 `models/icon_detect.pt`，`ui_parse` 自动启用 YOLO 像素级元素检测（需 `pip install ultralytics`）。

```toml
# ~/.codex/config.toml
[mcp_servers.vision-primitives]
command = "python"
args = ['/path/to/vision-primitives-mcp/vision_primitives_mcp.py']
startup_timeout_sec = 60

[mcp_servers.vision-primitives.env]
VISION_API_BASE = "https://api.xiaomimimo.com/v1"      # 或本地 http://127.0.0.1:1234/v1
VISION_API_KEY = "你的MiMo密钥"                          # 本地 LM Studio 用任意占位
VISION_MODEL = "mimo-v2.5"                              # 或 qwen/qwen3-vl-8b
VISION_OUTPUT_DIR = '/path/to/vision-primitives-mcp/generated'
```

关键环境变量：`VISION_TIMEOUT_S`(120) / `VISION_MAX_IMAGE_MB`(20) / `VISION_SAMPLES`(1，定位稳定性) / `VISION_DISABLE_THINKING`(本地思考型模型设 1) / `VISION_NO_SYSTEM`(本地小模型设 1)。完整列表见 [README.en.md](README.en.md)。

## 已知边界（诚实声明）

- **整图定位受分辨率稀释**：目标越复杂/越小偏差越大（20-164px），用 `som_locate` 递归或 `final="cv"` 对抗
- **延迟**：MiMo 推理型模型单次 15-25s；本地 Qwen3-VL 单次 3-10s；`cv_locate` 纯本地毫秒级
- **`scan_anomalies` 角度估计不稳定**（10°-35° 波动），价值在多候选 + 逐点验证，最终需实物核对
- **`cursor_locate` 在本地小模型上表现一般**（相对偏移估计有限），建议云端强模型
- **`ui_refine` 的 VLM 审查建议云端强模型**（本地 8B 思考型输出长 JSON 可达数分钟）；程序化合并/去重部分无此限制
- **`cv_locate` 泛化有限**：仅颜色/模板特征明显的简单目标；模板匹配对纯色（零纹理）目标退化（ZNCC 数学特性）
- **平台**：屏幕控制 8 工具为 Windows-only（ctypes + ImageGrab），macOS/Linux 下仅截屏/信息可用
- **SSRF 防护**：URL 图片默认拦截私网/链路本地/元数据地址（回环放行），`VISION_ALLOW_PRIVATE_NET=1` 放行；URL 来源图片解压后限 50MP（本地文件支持 200MP PCB）

## 版本历史（精简）

- **v1.10（2026-08-02）**：SoM 编号定位（`som_locate`，final=box/number/cv）、Cursor 交互搜索、CV 备选方案（`cv_locate`）；工具 24 个；测试 142 项；定位方法学 + grounding VLM 选型章节；MCP 协议修复（响应帧补 jsonrpc/id，严格客户端兼容）
- **v1.9（2026-08-02）**：Computer Use（8 个屏幕控制工具，安全开关默认关）
- **v1.8.x**：compare_infer / reason_graph / annotate_infer（虚拟标注 + 图形推理）/ 坐标格式实验（像素最稳）/ extract_json 增强
- **v1.7**：locate_object `refine` 两阶段精修（误差 70→20px）
- **v1.1-1.6**：scan_anomalies 自动异常扫描（PCB 实战）、compare_images、中文 OCR 修复（GDI+ 渲染）

## 测试与安全

```bash
python test\run_tests.py   # 142 项 mock 测试，不依赖真实 key
python test\e2e_mimo.py    # 真实端到端（需要 VISION_API_KEY）
```

- 输入图片只读，上传仅发往配置的后端；`out_path` 强制限定输出目录；URL 图片有 SSRF 与解压炸弹防护
- API key 仅存本地配置；屏幕控制类工具默认拒绝（`VISION_ALLOW_SCREEN_CONTROL=1` 才启用）

**仓库**：[github.com/zouyuanqing/vision-primitives-mcp](https://github.com/zouyuanqing/vision-primitives-mcp)
