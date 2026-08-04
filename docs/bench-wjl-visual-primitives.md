# 外部项目基准测试：Wjl1224734792/visual-primitives-mcp vs vision-primitives-mcp

> 日期：2026-08-03 · 测试者：zouyuanqing · 复跑脚本见同目录 `bench_against_mimo.py`

## 背景

对 GitHub 仓库 [Wjl1224734792/visual-primitives-mcp](https://github.com/Wjl1224734792/visual-primitives-mcp) 进行真实 API 基准测试。该仓库与本项目同源于 DeepSeek《Thinking with Visual Primitives》论文，声称"精确空间推理"，但无任何精度基准数据。

## 测试环境

| 项 | 值 |
|---|---|
| 视觉后端 | MiMo V2.5（OpenAI 兼容端点 `api.xiaomimimo.com/v1`） |
| 测试图 | `sample.png`（800×500，含蓝色提交按钮等 UI 元素） |
| Ground truth | `cv_locate`（color-cc 颜色分割+连通域，像素级，置信度 0.97） |
| 目标 | 蓝色提交按钮，真实 box `[121,181,299,239]`，中心 `(210,210)` |
| 流程 | 按对方 README 推荐：`visual_describe` → `visual_locate` |

## 实测结果

### 1. visual_describe 定位偏差

对方输出"提交按钮"：`bbox=[150,250,350,320]`（0-1000 归一化）→ 像素 `[120,125,280,160]`，中心 `(200,142.5)`。

- 中心偏差：**68.2px**（x 差 10px，y 差 67.5px）
- **IoU = 0**：对方框 y 范围 [125,160] 与真实 [181,239] 完全不重叠

结论：无 grounding 模型的单次绝对坐标输出，误差落在此前实测的 20-70px 区间上沿，且对方无任何修正机制。

### 2. visual_locate 是零验证缓存回放

```
raw_visual_analysis: null
from_cache: true
objects_count: 7
耗时: 0.0s
```

缓存命中时不调用视觉模型，把 describe 第一轮的 7 个物体原样全部返回，未执行"只定位用户要求的目标"规则。定位精度永远停留在第一次猜测水平，无二次确认、无裁切细看、无采样验证。

### 3. 协议层缺陷

pino 日志写入 stdout，污染 MCP stdio 的 JSON-RPC 响应流（日志必须走 stderr）。严格客户端按协议解析 stdout 会直接失败。

### 4. 测试套件盲区

13 个测试文件 176 用例全部为 mock 单元测试（断路器/并发/配置/解析器等），无任何真实图片 + ground truth 的端到端精度基准。

## 根因分析

- `validator.ts` 只校验格式合法性（范围、ID 唯一、x1<x2、centroid 在框内），不校验空间正确性，偏移 200px 的框也能通过
- describe 误差在 locate 阶段被当作"精确视觉分析"注入 prompt（`prompt-builder.ts`），错误被继承强化
- 推荐模型 qwen3.7-flash 无公开 grounding 基准，属闭源 Flash 档

## 对比：本项目的解决路径

| 问题 | 对方 | 本项目 |
|---|---|---|
| 绝对坐标偏差 20-70px | 无修正机制 | `refine` 两阶段精修（70px→20px） |
| 输出随机性 | 单次猜测 | `VISION_SAMPLES` 多次取样取中位数 |
| 无 grounding 模型定位 | 直接信任 | `som_locate` 编号网格递归、`cursor_locate` 相对偏移迭代 |
| 像素级兜底 | 无 | `cv_locate` 颜色分割+连通域（0-4px） |
| 文本类 UI 元素 | 无 | `ui_locate` 文本锚定 |
| 坐标验证 | 仅格式校验 | 坐标钳制 + 程序化验证 |

## 对外动作

2026-08-03 已向对方仓库提交 issue #1（实测数据 + 根因分析 + 解决路径 + 本项目链接），并追加评论建议换用带 grounding 训练的模型（Qwen2.5-VL 系列）。

- Issue: https://github.com/Wjl1224734792/visual-primitives-mcp/issues/1
- 复跑脚本: `bench_against_mimo.py`（需 `VISION_API_KEY` 环境变量，指向 MiMo key）
