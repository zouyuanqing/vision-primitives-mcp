# ZeroBench 评测工作流

ZeroBench（"Impossible" 视觉推理基准）持续追踪：主问题 + 子问题集。

## 结果（2026-08-02，MiMo V2.5 直答 + pass@5 采样）

| 数据集 | 题数 | 命中 | numeric 命中率 |
|---|---|---|---|
| 主问题 | 100（numeric 84） | 2 | **2.4%** |
| 子问题集 | 334（numeric 257） | 183 | **71.2%** |

对照：ZeroBench 发布时 SOTA 0%（GPT-4o 等前沿模型），一年后 pass@5 约 19%。

## 用法

```bash
# 下载数据（mm-eval/ZeroBench 镜像，非 gated）
python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/datasets/mm-eval/ZeroBench/resolve/main/data/zerobench-00000-of-00001.parquet', 'zerobench.parquet')"
python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/datasets/mm-eval/ZeroBench/resolve/main/data/zerobench_subquestions-00000-of-00001.parquet', 'zerobench_sub.parquet')"

# 主问题评测（断点续跑，可分批）
python eval_main.py 1 100 --samples 5

# 子问题集评测
python eval_sub.py 1 334 --samples 5

# 统计（自动去重）
python stats.py
```

## Pipeline

- **视觉**：MiMo V2.5 云端直答（FINAL ANSWER 格式强制 + 采样温度 0/0.9）
- **评测**：numeric 数字匹配（容差 0.01）；mcq/yes_no/open 在 FINAL ANSWER 后找答案字符串
- 依赖：pyarrow（读 parquet）、Pillow（图片缩放）；MiMo API key 通过环境变量 `MIMO_API_KEY` 提供（脚本不硬编码）

## 结论（第一轮基线）

1. 主问题考"多步组合视觉推理"，当前模型（MiMo/Qwen2.5-VL-7B）2.4% 是合理水平——瓶颈是模型能力，不是工具链
2. **子问题集是工具链的主场**：拆解后的单步问题，MiMo 直答 + 采样达 71.2% numeric
3. 提升路径：更强视觉模型（Qwen3-VL-32B / GLM-5V 接入后重跑）、采样加倍、针对性补采
