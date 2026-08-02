# Vision Primitives MCP — 给纯文本 LLM 装上眼睛的视觉原语 MCP 服务器

[简体中文](./README.md) | [English](./README.en.md)

让纯文本模型（如 `deepseek-v4-flash`）通过 MCP 工具获得完整"看图 + 操作图片"能力：
**看图 → 定位（输出坐标）→ 圈画标注 → 裁切/放大 → OCR 提取** 的多步交互闭环，处理后的图片可直接交还展示。

视觉后端：**小米 MiMo V2.5**（OpenAI 兼容 API，全模态，支持图片输入与坐标输出）。

- 实现：单文件 Python（`vision_bridge_mcp.py`），依赖仅 Pillow（本机已装 12.2.0），零第三方运行时依赖
- 协议：手写 MCP stdio（JSON-RPC 2.0 + Content-Length 帧），兼容 Codex 桌面端/CLI
- 灵感来源：HanaAgent 的 Vision Bridge（辅助视觉模型 + 结构化"视觉原语"）+ DeepSeek《Thinking with Visual Primitives》（坐标框/点 + 标签）

## 工具一览（21 个）

| 工具 | 作用 | 关键参数 |
|---|---|---|
| `describe_image` | 文字描述图片 | `image` 必填；`question`、`detail`(brief/balanced/detailed) |
| `analyze_image` | 结构化分析：描述 + visual_primitives（box/point+标签+置信度） | `image` 必填；`format`(generic/gemini/qwen) |
| `locate_object` | 定位目标对象，返回坐标（让 LLM 输出坐标）；`refine=true` 两阶段精修 | `image`、`target` 必填；`coords`(pixel/norm)、`refine` |
| `ocr_image` | 逐文本块 OCR，带 bbox（像素+归一化） | `image` 必填；`language` |
| `annotate_image` | 在图上画框/圆点/标签，保存标注图 | `image`、`items` 必填；`coords`、`out_path`、`style` |
| `crop_image` | 按坐标裁切（可边缘外扩 expand_px） | `image`、`box` 必填；`coords`、`expand_px` |
| `zoom_region` | 区域放大（scale 1-8） | `image` 必填；`box`、`scale` |
| `vision_health` | 检查后端配置与连通性 | 无 |
| `compare_images` | **多图对比**（2-4 张）：A/B 截图对比、设计稿一致性、多帧分析 | `images` 必填（2-4 张）；`question`、`detail` |
| `compare_infer` | **多图联合推理**：每图可带独立标注，联合推理差异/因果/时序/整体结论 | `images`、`question` 必填；`items_per_image`、`mode` |
| `reason_graph` | **交互式图形推理协议**：原语(locate/measure) → 语义(semantic/hypothesis) → 标注(annotate/verify) 多轮循环，session 跨轮传递 | `image`、`step` 必填；`session`、`question` |
| `screen_capture` | 截屏（全屏/区域），配合视觉工具实现「看屏幕」 | `region`、`out_path` 可选 |
| `screen_info` | 屏幕分辨率 / DPI / 控制开关状态 | 无 |
| `screen_click` / `screen_move` / `screen_drag` / `screen_scroll` / `screen_type` / `screen_key` | 鼠标键盘控制（需 `VISION_ALLOW_SCREEN_CONTROL=1`） | 坐标/按键参数 |
| `annotate_infer` | **虚拟标注 + 图形推理**：框/点/连线/箭头/圆/多边形/气泡注入视觉模型（原图零修改或半透明叠加），支持多轮修正与自动框选 | `image`、`question` 必填；`items`/`auto_boxes` 至少其一；`mode`、`alpha`、`corrections` |
| `scan_anomalies` | **自动异常扫描**：切块定位候选 → 高清逐点验证 → 输出带角度/丝印/置信度的报告 | `image` 必填；`target`、`region`、`verify`、`max_tiles` |

坐标系统：所有工具接受 `coords="pixel"`（默认，MiMo 实测更准）或 `coords="norm"`（0–1000 归一化）；越界坐标自动钳制并返回 `clamped: true`。


## 截图演示（Demo）

使用仓库内 `sample.png`（测试样例图）走完整视觉原语工作流的效果：

**定位 + 圈画标注**（`locate_object` 找到"蓝色提交按钮" → `annotate_image` 画框标注）：

![工作流对比](docs/demo-workflow.png)

**标注输出**（`annotate_image` 返回的标注图）：

![标注演示](docs/demo-annotate.png)

**按坐标裁切**（`crop_image`，裁出按钮区域）：

![裁切演示](docs/demo-crop.png)

**区域放大**（`zoom_region`，2 倍放大便于细节识别）：

![放大演示](docs/demo-zoom.png)

**多图联合推理**（`compare_infer`：UI 测试图 + 电源框图联合分析，带标注叠加）：

![多图联合推理演示](docs/demo-compare-infer.png)
**模型对比矩阵**（MiMo V2.5 vs LM Studio 本地模型，同一 benchmark 实测）：

![模型对比矩阵](docs/demo-model-matrix.png)

**定位效果对比**（同一测试图上各模型 locate 输出框叠加，黑虚线为真实位置）：

![定位效果对比矩阵](docs/demo-locate-matrix.png)

**Computer Use 实战**（纯文本模型真实操控电脑：打开 B 站 → 刷新 → 定位并点击第一个视频 → 验证进入播放页）：

![电脑控制点击前](docs/demo-cu-before.png)

![电脑控制点击后](docs/demo-cu-after.png)

完整报告见 [docs/computer-use-test-report.md](docs/computer-use-test-report.md)。

## 安装与配置

### 1. 在 `~/.codex/config.toml` 追加

```toml
# --- MCP: Vision Bridge ---
[mcp_servers.vision-bridge]
command = "python"
args = ['/path/to/vision-bridge-mcp/vision_bridge_mcp.py']
startup_timeout_sec = 60

[mcp_servers.vision-bridge.env]
VISION_API_BASE = "https://api.xiaomimimo.com/v1"
VISION_API_KEY = "sk-你的小米MiMo密钥"
VISION_MODEL = "mimo-v2.5"
VISION_OUTPUT_DIR = '/path/to/vision-bridge-mcp/generated'
```

重启 Codex 后生效，用 `vision_health` 验证。

### 2. 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `VISION_API_BASE` | `https://api.xiaomimimo.com/v1` | OpenAI 兼容端点 |
| `VISION_API_KEY` | （必填） | 小米 MiMo API key |
| `VISION_MODEL` | `mimo-v2.5` | 注意：实测 `mimo-v2.5-pro` 在该 API 上不支持图片输入（404），如需更强视觉能力请在 MiMo 平台确认图像模型名 |
| `VISION_MAX_TOKENS` | `4096` | MiMo 为推理模型，思维链耗 token 多 |
| `VISION_TIMEOUT_S` | `120` | 单次调用超时 |
| `VISION_MAX_IMAGE_MB` | `20` | 图片大小上限 |
| `VISION_CACHE` | `1` | 结果缓存（sha256 图片+问题+模型），设 `0` 关闭 |
| `VISION_SAMPLES` | `1` | >1 时 locate/analyze 多次取样，按空间位置聚类取中位数（更稳，但更慢更贵） |
| `VISION_OUTPUT_DIR` | `generated/` | 生成图片输出目录（out_path 必须在该目录内） |
| `VISION_DEBUG` | `0` | 设 `1` 输出日志到 stderr |
| `VISION_DISABLE_THINKING` | `0` | LM Studio 思考型视觉模型（minicpm-v-4_5 等）设 `1` 避免 content 为空 |
| `VISION_NO_SYSTEM` | `0` | 本地小模型 system 消息可能触发空响应，设 `1` 跳过 system 消息 |
| `VISION_MAX_MODEL_SIDE` | `2600` | 送模型的图像长边上限（px），调大可提高输入分辨率 |
| `VISION_NORM_SIZE` | `0` | >0 时精确归一化图像长边到该尺寸（保持纵横比，坐标自动换算回原图），用于对齐模型原生分辨率 |

## 交互工作流示例

```text
用户：帮我看一下这张截图里有哪些报错，并把报错位置圈出来
模型（Codex）：
1. ocr_image("screenshot.png")            # 提取文字 + 坐标
2. locate_object("screenshot.png", "报错文字区域")
3. crop_image("screenshot.png", box)      # 裁切细节细看
4. zoom_region("screenshot.png", box, scale=3)
5. annotate_image("screenshot.png", [{label:"报错", box, color:"#ff3b30"}])
   -> 返回标注图路径，Codex 用 Markdown 展示给用户
```

## 实测结论（MiMo V2.5，2026-08-01）

| 能力 | 实测表现 |
|---|---|
| 描述（describe） | 优秀：布局、对象、文字、颜色全部准确 |
| 定位（locate） | 简单几何图形（圆形 [120,320,220,420]）**完全精确**；复杂元素（圆角按钮）偏移约 20-40px —— 建议关键任务开 `VISION_SAMPLES=3` 或切 `mimo-v2.5-pro`，圈画后人工核对 |
| OCR | 中英文均准确（"你好，视觉桥接插件测试图"、"提交"及 LED 屏车次/站名全部正确读取），返回带 bbox 的逐块文字 |
| 标注/裁切/放大 | 程序化像素验证通过（框线命中、裁切尺寸正确） |
| 延迟 | 单次视觉调用约 15-25 秒（MiMo 推理型），完整闭环约 2 分钟 |

## 测试

```bash
# mock 测试（不依赖真实 key，45 项全绿）
python test\run_tests.py

# 真实端到端（需要 VISION_API_KEY）
python test\e2e_mimo.py
```

## 安全说明

- 输入图片只读，不写入/修改原文件；上传仅发往所配置的 MiMo API
- `out_path` 强制限定在 `VISION_OUTPUT_DIR` 内，防止越界写入
- API key 仅存于本地 `~/.codex/config.toml`（与现有 GitHub MCP 的 token 存放方式一致）；若 key 泄露，请到 [MiMo 控制台](https://platform.xiaomimimo.com) 轮换
- 图片大小 ≤20MB，扩展名白名单：png/jpg/jpeg/webp/gif/bmp

## 文件结构

```
vision-bridge-mcp/
├── vision_bridge_mcp.py   # MCP server（单文件实现）
├── sample.png             # 测试样例图
├── README.md / README.en.md   # 中英双语说明
├── .env.example
├── generated/             # 工具生成的图片（e2e 产物）
├── .cache/                # 结果缓存（自动创建）
└── test/
    ├── run_tests.py       # mock 测试套件
    └── e2e_mimo.py        # 真实 MiMo e2e 脚本
```
## scan_anomalies — 自动异常元件扫描（v1.1 新增）

一步完成"找异常元件"的完整流程，无需手动多轮操作：

```text
scan_anomalies(image, target="摆放歪斜、方向与周边不一致的元件",
               region=[x1,y1,x2,y2]?, verify=true, max_tiles=6, overlap=250)
```

工作方式：
1. 把 `region`（默认全图）切成带重叠的块（最多 `max_tiles` 块），逐块 `locate_object` 收集候选（**完整边界框 + rotation 角度 + 全部目标**）
2. 按空间位置合并去重（中心距离 / IoU）
3. `verify=true` 时：每个候选从**原图**高清裁切放大，客观化提问验证（是否歪斜/角度/丝印/类型），解析为结构化判定
4. 输出按"歪斜 → 不确定 → 正常"排序的候选报告

实测（2026-08-01，200MP PCB 板，region 限定右下区域）：4 块扫描 → 3 候选 → 自动验证排除 2 个 → 锁定歪斜元件（A1142C LDO 所在区域，验证判定歪斜 ~30°，丝印 5C）。总耗时约 2 分钟（8 次视觉调用）。

> 已知边界：视觉模型对"歪斜"的检测召回率低、角度估计不稳定（10°~35° 波动），`scan_anomalies` 的价值在于**自动多候选 + 逐点验证排除**，最终仍建议实物核对。大图（如 200MP）请把 `VISION_MAX_IMAGE_MB` 调到文件实际大小（默认 20MB）。

## 实测更新（v1.1，2026-08-01）

- 定位可靠性：单次 `locate_object` 对"歪斜"类目标幻觉率较高（两次测试均需纠错）；**`scan_anomalies` 的多候选+验证流程可自动排除误报**
- 完整边界框：`locate_object`/`scan_anomalies` 输出改为"包含本体+焊盘"的完整框（旧版经常只框到元件局部，如 404×131 实际元件 700×540）
- 角度原语：`rotation` 字段随 primitives 输出（模型估计，仅供参考）
- 网络健壮性：`call_chat` 对连接超时/网络错误也重试 1 次
## v1.2（2026-08-02）

- 新增 `compare_images`：2-4 张图多图对比（A/B 截图、设计稿一致性、多帧分析），图片以多图消息原生送入 MiMo，不拼接不降质；带缓存与参数校验
- 测试增至 **62 项**（mock，不依赖真实 key）
## v1.4（2026-08-02）

- **修复 `sample.png` 中文渲染**：原测试图用 PIL 生成时 CJK 被渲染为 notdef（`?`），导致早期得出"中文 OCR 弱"的错误结论；改用 GDI+（System.Drawing）生成，中文显示与识别均正常
- **修正实测结论**：MiMo V2.5 中文 OCR 正常（高铁站 LED 屏车次/站名、中文按钮文字均正确读取）
- 演示图基于修复后的 `sample.png` 重新生成（标注框来自真实 `locate_object` 定位结果）


## v1.5（2026-08-02）

- 新增 `annotate_infer`：虚拟标注 + 增强图形推理
  - `mode=virtual`（默认）：框/点/连线/箭头/圆等标注几何以坐标文本注入 prompt，**原图零修改**
  - `mode=overlay`：生成半透明标注叠加层（alpha 可调，不遮挡原图），叠加后送模型推理
  - 支持类型：`box` / `point` / `line` / `arrow` / `circle`，坐标支持 pixel/norm
  - 实测：框 A=提交、框 B=Cancel 识别正确；箭头连线被推理为"操作闭环"布局语义
- 测试增至 **71 项**（mock，不依赖真实 key）


## v1.6（2026-08-02）

- `annotate_infer` 大幅升级：
  - 新标注类型：**polygon（多边形）**、**bubble（文本气泡标注，带引出线）**
  - **多轮标注修正 `corrections`**：`add / remove / move(delta|to) / resize / set`，基于已有标注增量修正后推理（`applied` 返回全部标注 id 供下一轮复用）
  - **自动框选 `auto_boxes`**：内部 `locate_object` 自动框选目标（紫色框），可与手动标注混合
  - **大图支持**：>2600px 图自动降采样后送模型（标注坐标同步换算，返回坐标仍为原图系）
  - 200MP PCB 实测：虚拟标注 A1142C（歪斜 LDO）+ SL2.1S HUB + 供电箭头 → 模型正确推理供电链路（TD1583→LDO→HUB VCC）与歪斜焊点/散热/引脚三大可靠性风险
  - mermaid 复杂连线图实测：自动框选 4 节点 + 两轮交互式推理（SW 节点、Vout→LDO、反馈分压）
- 修复：`load_image` 内置支持超大图（200MP），`items` 与 `auto_boxes` 可二选一
- 测试增至 **85 项**（mock，不依赖真实 key）


## v1.7（2026-08-02）

- `locate_object` 新增 **`refine` 两阶段精修**：粗定位 → 裁切放大 → 二次定位 → 换算回原图坐标
  - 实测（sample.png 蓝色按钮）：误差从 **70px 降到 20px**（y 方向完全精确）
  - 原理：第一轮模型在整图上"估算"（受视觉 token 粒度限制），第二轮在放大的局部图上"细看"，消除整图量化误差
- 测试增至 **88 项**（mock，不依赖真实 key）

### 关于"框为什么歪"的已知边界（实测总结）
- 根因：MiMo 视觉 token 粒度粗（200×100 图仅 18 个 image token）+ 无 grounding 专用训练 + 推理采样波动
- 对比度强、形状规则的几何体（圆形/色块）定位 **0 误差**；圆角按钮/文字/PCB 元件边缘模糊时偏差 20-70px
- 缓解手段（已内置）：`refine` 两阶段精修、`VISION_SAMPLES` 多次取样中位数、`scan_anomalies` 逐点验证、坐标钳制


## v1.8（2026-08-02）

- 新增 `compare_infer`：**多图联合推理**（2-4 张，每图可带独立标注 `items_per_image`，支持 virtual/overlay 模式）
  - 实测：电源框图 + UI 测试图联合推理 → 正确得出"硬件供电 + 软件交互构成完整系统"的跨域结论
- 新增 `reason_graph`：**原语-语义-标注交互式推理协议**
  - `step` 七种动作：`locate`（定位，支持 refine）/ `measure`（程序化测量：distance/angle/area，零 API 成本）/ `annotate`（固化为标注+叠加图）/ `semantic`（语义记录）/ `hypothesis`（假设）/ `verify`（虚拟标注验证）/ `next`（下一步建议）
  - `session` 跨轮传递（primitives/annotations/semantics/hypotheses），主模型可多轮循环直到收敛
  - 实测（mermaid 电源框图）：locate 2 节点 → 程序化测距 833px → 提出"LDO 输入来自 TD1583"假设 → 模型验证成立 → next 给出后续推理建议
- 测试增至 **102 项**（mock，不依赖真实 key）


## v1.8.1（2026-08-02）

- `PRIMITIVE_PROMPT` 兼容小模型：允许 ```json 代码块包裹输出（修复 qwen3.5-9b 等本地小模型"禁止代码块"指令下输出空响应的问题）
- `MAX_MODEL_SIDE` 环境变量化（`VISION_MAX_MODEL_SIDE`，默认 2600）——调大可提高输入分辨率（实测对 LM Studio 端到端定位精度无显著帮助，服务端有固定缩放）
- 本地多后端实测（LM Studio qwen3.5-9b）：describe/OCR 中文优秀（<15s）；**定位精度弱**（按钮误差 210px，粗框偏移导致 refine 失效）——精细定位建议用 grounding 较强的云端模型（MiMo V2.5 refine 后误差 20px）


## v1.8.2（2026-08-02）

- 新增本地后端兼容：`VISION_DISABLE_THINKING=1`（思考型模型禁用思考）、`VISION_NO_SYSTEM=1`（跳过 system 消息）
- LM Studio `minicpm-v-4_5` 推荐配置：`VISION_DISABLE_THINKING=1` + `VISION_NO_SYSTEM=1` + `VISION_MAX_TOKENS=16000` + `VISION_MAX_MODEL_SIDE=4096`
- 本地模型实测对比（同一 benchmark）：
  - `qwen3.5-9b`：describe/OCR 优秀（<15s）；**定位不可用**（按钮误差 210px、颜色-位置映射错误）
  - `minicpm-v-4_5`：describe/OCR 优秀（bbox 准确）；**定位可用**（x 方向精确命中，y 方向系统性偏上 ~90px）；偶发 400（LM Studio peg 格式校验）


## v1.8.3（2026-08-02）

- 新增 `VISION_NORM_SIZE`：客户端精确归一化图像到目标长边（默认 0=仅上限），坐标自动换算回原图
- 归一化校准实验（minicpm-v-4_5，红圆定位，448/640/896/1152/1280/1600px）：
  - 误差 60~337px **随机波动**，无稳定"原生分辨率最优"规律 → 该模型定位误差主因是空间理解波动，非服务端缩放
  - 归一化价值在于**确定性**与适配有原生分辨率偏好的模型（如 Qwen-VL 系）；对抗随机性建议用 `VISION_SAMPLES` 多次取样中位数
- 测试增至 **104 项**（mock，不依赖真实 key）


## v1.8.4（2026-08-02）

- **坐标系统校准实验**（minicpm-v-4_5，红圆定位，同图 3 种输出格式 × 2 次）：
  - 像素坐标：误差 **40 / 60px**（最佳，x 方向精确命中）
  - 0-1000 归一化：模型输出格式混乱（LaTeX 尾巴 + 不合理坐标）
  - 0-1 比例：误差 130px（最差）
  - **结论：像素坐标是本地小模型最稳定的输出格式**；比例坐标假设被实证否定，工具维持像素为主 + norm 换算输出
- 修复 `extract_json`：逐位置 raw_decode 取代贪婪正则——现在能处理"JSON+LaTeX 尾巴 / 文本前缀 + JSON / 代码块"等混合输出（实验中发现 0-1000 模式因尾巴解析失败的 bug）
- 测试增至 **108 项**（mock，不依赖真实 key）


## v1.9（2026-08-02）— Computer Use（无视觉模型做电脑控制）

- 新增 **8 个电脑控制工具**：`screen_capture`（截屏）/ `screen_info` / `screen_click` / `screen_move` / `screen_drag` / `screen_scroll` / `screen_type`（中文走剪贴板粘贴）/ `screen_key`（含 ctrl+shift 组合键）
- 零依赖：ctypes + PIL ImageGrab（Windows）
- **安全开关 `VISION_ALLOW_SCREEN_CONTROL`（默认关）**：关闭时所有控制类工具拒绝执行，只有截屏/信息可用——防止纯文本模型未经允许操控鼠标键盘
- 电脑控制闭环（纯文本模型可用）：
  ```
  1. screen_capture → 截图
  2. locate_object(截图, "确定按钮") → 坐标
  3. screen_click(x, y) → 点击
  4. screen_capture → 验证结果（视觉反馈循环）
  ```
- 实测：2560×1440 截屏成功；安全开关拒绝路径全部生效
- 测试增至 **116 项**（mock，不依赖真实 key）

## v1.9.1（2026-08-02）— MCP 响应协议修复（严格客户端兼容）

- **修复**：`_text_result` 构造的工具响应帧缺少 `jsonrpc` 与 `id` 字段，严格校验的 MCP 客户端（如 HanaAgent）无法将响应与请求匹配，所有工具调用表现为"超时"（实际上工具已处理完并返回，但响应被客户端丢弃）。现在响应帧补全 `jsonrpc` + `id`，与 initialize / ping / 错误响应保持一致
- 影响面：Codex 桌面端对 id 缺失宽容，故此前实测未暴露；修复向后兼容，无接口变更
- **HanaAgent 全链路验证**（修复后）：`vision_health` 0.18s、`describe_image` 命中缓存秒回、`ocr_image` 23s、`locate_object` 4.7~6.3s，全部正常返回
- **交叉验证基准**（程序化 ground truth，PIL 像素级逐项核对）：
  - `locate_object` 红圆：框内红色像素占比 43.3%，中心偏差 64px —— 框得住目标，粗定位可用，符合既有"20~100px 级偏差"结论
  - `locate_object` 绿三角：框内绿色像素占比 14.2%，中心偏差 97px（框整体偏右下）
  - `ocr_image` 位置：x 方向近乎精确命中，y 偏差 ≤15px；文字内容 4/4 全部正确
  - `describe_image` 语义准确；唯一瑕疵：210×190 的矩形被描述为正方形（接近方形）
- mock 测试维持 **116 项**（本修复为协议层改动，MCP probe 验证见会话记录）
