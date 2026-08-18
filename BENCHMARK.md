# Reward Agent Benchmark

## Prepare local data first

The evaluator never downloads datasets implicitly. Before training or evaluation,
generate the normalized local JSONL files explicitly from the repository root:

```bash
python -m reward_harness.prepare_data --benchmarks rewardbench rewardbench2 rmbench
```

If the original Hugging Face/raw files are already cached, prohibit network access:

```bash
python -m reward_harness.prepare_data \
  --benchmarks rewardbench rewardbench2 rmbench \
  --offline
```

Files are written under `data/{benchmark}/` with a manifest containing
the case count, source fingerprint and SHA-256 checksum. The normalized files are
tracked by Git. Use `--force` only when intentionally regenerating them.

## 四卡 vLLM（推荐）

在 Linux/WSL 服务器上，先修改根目录 `start_vllm_4gpu.sh` 中的 `MODEL_PATH`，然后启动：

```bash
bash start_vllm_4gpu.sh
```

默认使用 DP=4、TP=1，让每张 GPU 各运行一个 Qwen3-8B 副本，以提高全量评测吞吐。
如果单卡显存无法容纳 BF16 权重，按脚本末尾说明改成 DP=1、TP=4。

服务健康后运行 benchmark，并自动评测 `agents/` 下的所有 agent。批量脚本目前默认
运行 RewardBench 2 与 RM-Bench；原版 RewardBench 可通过 CLI 显式加入：

```bash
bash run_reward_benchmarks_vllm.sh

python -m reward_harness.benchmark \
  --benchmarks rewardbench \
  --smoke-per-group 0
```

若 benchmark 和 vLLM 不在同一台服务器，把脚本中的 `VLLM_BASE_URL` 改成 vLLM
服务器地址。

从 `Reward-Harness` 根目录运行：

```bash
python -m reward_harness.benchmark \
  --benchmarks rewardbench2 rmbench \
  --smoke-per-group 0
```

根入口默认连接本机 vLLM 的 `Qwen/Qwen3-8B`。CLI 默认是每组 2 条的冒烟评测；
完整评测需要显式传入 `--smoke-per-group 0`。无需再传 `--resume`，续跑和完整结果跳过
均已自动启用。

默认会自动扫描 `reward_harness/agents/*.py`。每个文件需要定义一个
`RewardSystem` 子类，文件名就是 agent 名称；`__init__.py` 和以下划线开头的文件会忽略。
使用 `--agents no_skill` 可以只评测指定 agent；旧的 `--harnesses` 仍作为兼容别名保留。

无 Rubric 的直接标量 baseline 位于 `reward_harness/agents/no_rubric.py`。它的 `build_rubrics()`
返回空集合且不调用模型，随后对每个候选调用一次 Judge，直接解析 0～1 reward：

```bash
python -m reward_harness.benchmark \
  --benchmarks rewardbench rewardbench2 rmbench \
  --agents no_rubric \
  --smoke-per-group 0
```

Baseline 提示词吸收了 Eval-Skill 的核心意图分析与去位置偏差、AdaRubrics 的
task-specific/observable/orthogonal/calibrated rubric 原则，以及 OpenJudge 的严格结构化
pointwise 输出。没有移植会同时展示多个候选的 pairwise/listwise 提示词，也没有在测试题上
执行依赖 gold label 的 rubric 修订流程，以保持候选隔离和 benchmark 可信性。

常用参数：

```text
--benchmarks rewardbench rewardbench2 rmbench
--agents no_skill init_skill
--workers 4
--request-workers 16
--smoke-per-group 2
--logs-dir logs/reward_agent
--results-dir results/reward_agent
```

每次运行会根据数据 manifest、样本 ID、agent 源码、evaluator 代码、模型和抽样配置生成
`run_signature`。相同签名已经存在完整 `summary.json` 时会自动跳过；部分运行则自动读取
已有轨迹继续执行。修改 agent 代码会得到新的签名和独立目录。仅在明确需要重跑时使用
`--force`。

目录结构参考原 Meta-Harness，把轨迹日志和最终结果分开：

```text
logs/reward_agent/{benchmark}/{agent}/{model}_{signature}/
├── config.json
└── trajectories.jsonl

results/reward_agent/{benchmark}/{agent}/{model}_{signature}/
├── config.json
└── summary.json
```

- `trajectories.jsonl`：唯一的完整轨迹文件。每行独立包含 Query、Responses、
  evaluator-only gold、Harness metadata、Rubrics、Judgments、Reward、完整模型请求响应、
  token、延迟、错误及 benchmark 单题结果，可直接作为 Harness Optimization 输入。
- `summary.json`：数据集指标、错误数量、token/延迟汇总、运行签名和轨迹文件路径。
- `config.json`：脱敏模型配置、agent 文件及 SHA-256、数据目录和结果路径。

Runner 会先为一题生成一次共享 RubricSet，再并行执行该题的候选 `score()`；
所有 rubric/judge 请求共同受 `--request-workers` 限流。成功响应按 Prompt 哈希
缓存在 `{logs-dir}/.llm_cache/`，断点后可以复用；JSON 解析或接口校验失败时，
只清除当前线程对应的坏缓存并重新请求。

增加数据集时，实现 `BenchmarkAdapter` 的三个方法，并在
`benchmarks/__init__.py` 的 `ADAPTERS` registry 注册即可，runner 无需修改。
