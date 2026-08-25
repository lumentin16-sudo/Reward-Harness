# Reward Agent Benchmark

## Prepare local data first

The evaluator never downloads datasets implicitly. Before training or evaluation,
generate the normalized local JSONL files explicitly from the repository root:

```bash
python -m reward_harness.prepare_data --benchmarks helpsteer3 rewardbench rmbench
```

If the original Hugging Face/raw files are already cached, prohibit network access:

```bash
python -m reward_harness.prepare_data \
  --benchmarks helpsteer3 rewardbench rmbench \
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

服务健康后运行 benchmark，并自动评测 `agents/` 下的所有 agent。仓库脚本默认
运行 RewardBench；RM-Bench 可通过 CLI 显式加入（rewardbench2 暂不评测）：

```bash
bash run_benchmark.sh

python -m reward_harness.benchmark \
  --benchmarks rewardbench rmbench \
  --smoke-per-group 0
```

若 benchmark 和 vLLM 不在同一台服务器，把脚本中的 `VLLM_BASE_URL` 改成 vLLM
服务器地址。

从 `Reward-Harness` 根目录运行：

```bash
python -m reward_harness.benchmark \
  --benchmarks rmbench \
  --smoke-per-group 0
```

根入口默认连接本机 vLLM 的 `Qwen/Qwen3-8B`。CLI 默认是每组 2 条的冒烟评测；
完整评测需要显式传入 `--smoke-per-group 0`。无需再传 `--resume`，续跑和完整结果跳过
均已自动启用。

test 分支使用 winner-only 协议，可选择四个具体 Harness：
`no_rubric`、`no_skill`、`init_skill_no_rubric` 和 `init_skill`。前三者分别对应
vanilla、online-rubric 和无 Rubric 的 J-stage Skill；`init_skill` 使用 G-stage
Rubric Skill 生成 response-aware Rubrics，再直接输出唯一 winner。

复现实验调用参数可使用：

```bash
python -m reward_harness.benchmark \
  --benchmarks rewardbench \
  --agents no_rubric no_skill init_skill_no_rubric init_skill \
  --model Qwen/Qwen3-8B \
  --temperature 0.7 \
  --trial-num 3 \
  --max-tokens 10000 \
  --workers 32 \
  --request-workers 32 \
  --smoke-per-group 0
```

Runner 当前统一关闭 Qwen3 thinking；请求重试、并发和结果落盘仍由外部 evaluator 管理。

常用参数：

```text
--benchmarks rewardbench
--agents no_rubric no_skill init_skill_no_rubric init_skill
--workers 4
--request-workers 16
--trial-num 1
--smoke-per-group 2
--output-dir results
--run-tag 20260819_153045
```

默认使用启动时的本地时间 `YYYYMMDD_HHMMSS` 作为顶层运行目录。复用同一个
`--run-tag` 时，完整结果会跳过，部分轨迹会按 `(trial_index, case_id)` 续跑。修改 agent、模型、数据或
抽样配置时应使用新的 tag；`--force` 会清空当前 tag 下的已有轨迹并重新评测。

每个模型配置的轨迹和汇总结果保存在同一目录：

```text
results/{run_tag}/{benchmark}/{harness}/{model}/
├── config.json
├── trajectories.jsonl
└── summary.json
```

- `trajectories.jsonl`：唯一的完整轨迹文件。每行独立包含 Query、Responses、
  evaluator-only gold、Harness metadata、Rubrics、WinnerResult、完整模型请求响应、
  token、延迟、错误及 benchmark 单题结果；多次 trial 时额外包含 `trial_index`。
- `summary.json`：`primary_metrics` 保存 Average@N 关键指标，`metrics` 保存其余指标，`trials` 保留每轮结果，`voting` 保存 Voting@N，并通过 `counts`、`usage`、`artifacts` 保存计数、开销和轨迹路径。
- `config.json`：脱敏模型配置、agent 文件及 SHA-256、数据目录和结果路径。

每个 summary 最前面提供统一关键指标，并保留原有明细：

- RewardBench：`beyond_rubric` 与 `auto_rubric` 均为 Chat、Chat Hard、Safety、Reasoning 四个 section 的宏平均；
- RewardBench 2：数据可以离线准备，但当前 runner 暂不注册该 adapter，也不计算其指标；
- RM-Bench：先在每个 domain 内平均 3×3 比较矩阵；`beyond_rubric` 为四个 domain 的宏平均，`auto_rubric` 为四个 domain 按样本数加权的平均。

Runner 会先移除原始 Response ID/metadata，并按内容稳定重排全部回答，再将同一组匿名
Responses 交给 `build_rubrics()` 和 `judge()`。Judge 一次看到完整 A/B 并返回唯一
`WinnerResult`；evaluator 随后映射回原始 ID，gold 从不进入模型 Prompt。

RM-Bench 每个原始6回答 prompt 会展开成9个 chosen/rejected pair；每个 pair 调用一次 Judge，汇总时按 original case ID 重建3×3矩阵。因此全量1,327个 prompt 对应11,943个评测 case。
对 RM-Bench 使用 `--sample-size N` 时，N 表示原始 prompt 数量；runner 会为每个被抽中的 prompt 保留完整9个 pair。
所有 rubric/judge 请求共同受 `--request-workers` 限流。成功响应按 Prompt 哈希
缓存在 `results/{run_tag}/.llm_cache/`，断点后可以复用；`trial-num` 大于 1 时关闭 Rubric/Judge 缓存，保证每轮真正重新请求。JSON 解析或接口校验失败时，
只清除当前线程对应的坏缓存并重新请求。

增加数据集时，实现 `BenchmarkAdapter` 的三个方法，并在
`benchmarks/__init__.py` 的 `ADAPTERS` registry 注册即可，runner 无需修改。
