# Reward-Harness

Reward-Harness 是一个面向生成式 Reward Model / LLM-as-a-Judge 的评测框架。它使用统一的 `RewardSystem` 接口组织任务、Rubric 生成、候选回答评分和结果聚合，并提供 RewardBench、RewardBench 2、RM-Bench 的本地 vLLM 评测流程。

当前默认模型为 `Qwen/Qwen3-8B`。正式评测只连接本地或自建的 OpenAI-compatible vLLM 服务，不包含 SiliconFlow 等云 API 调用逻辑。

## 主要功能

- 统一的 Reward System 数据结构和调用接口。
- 每道题只生成一次共享 `RubricSet`，随后分别评价该题的所有候选回答。
- 自动发现 `reward_harness/agents/` 中的 Reward Harness，无需修改 runner。
- 支持有 Rubric、动态 Skill 和无 Rubric 直接评分三种 baseline。
- 支持 RewardBench、RewardBench 2 和 RM-Bench 官方风格指标。
- 数据准备与模型评测解耦；评测阶段不会隐式下载数据。
- 双层并发、请求限流、LLM 响应缓存、错误重试和中断续跑。
- 保存逐题轨迹、原始模型响应、Rubric、Judgment、token、延迟和汇总指标。
- 使用运行签名识别相同实验；已完成的相同配置会自动跳过。

## 目录结构

```text
Reward-Harness/
├── reward_harness/                 # 可直接 import 的 Python 包
│   ├── agents/                     # Reward Harness 实现
│   │   ├── no_rubric.py            # 不生成 Rubric，直接输出标量分数
│   │   ├── no_skill.py             # 直接生成 Rubric，再进行评分
│   │   └── init_skill.py           # 动态选择 Skill、生成 Rubric 并评分
│   ├── benchmarks/                 # 数据集适配器和官方风格指标
│   │   ├── base.py                 # BenchmarkCase/BenchmarkAdapter 统一协议
│   │   ├── rewardbench.py
│   │   ├── rewardbench2.py
│   │   └── rmbench.py
│   ├── reward_system.py            # RewardSystem 核心接口与数据类型
│   ├── agent_loader.py             # 自动发现 agents/*.py
│   ├── model_client.py             # vLLM OpenAI-compatible 客户端
│   ├── prepare_data.py             # 数据集下载和标准化
│   └── benchmark.py                # benchmark runner
├── start_vllm_4gpu.sh
└── run_*_vllm.sh                   # 常用运行脚本
```

标准化 benchmark 数据位于 `data/` 并随仓库提交；运行日志和结果分别写入 `logs/`、`results/`，不会提交到 Git。

## 环境准备

推荐使用 Linux 服务器、Python 3.10+ 和支持 OpenAI-compatible API 的 vLLM。

安装 Python 依赖：

```bash
cd Reward-Harness
pip install -r requirements.txt
```

项目自身只显式依赖：

- `openai`：访问 vLLM 的 OpenAI-compatible 接口。
- `datasets`：在数据准备阶段下载或读取 Hugging Face 数据集。

vLLM 请根据服务器 CUDA、PyTorch 和驱动环境单独安装。

## 1. 准备本地数据

正式评测不会联网下载数据。首次运行前，先将原始数据集转换成统一的本地 JSONL：

```bash
python -m reward_harness.prepare_data \
  --benchmarks rewardbench rewardbench2 rmbench
```

默认输出到：

```text
data/{benchmark}/{split}.jsonl
data/{benchmark}/{split}.manifest.json
```

manifest 会记录数据条数、来源 fingerprint 和内容 SHA-256，用于校验数据完整性与生成可复现的运行签名。

如果 Hugging Face 数据已经存在于本机缓存，并且不希望程序访问网络：

```bash
python -m reward_harness.prepare_data \
  --benchmarks rewardbench rewardbench2 rmbench \
  --offline
```

只有在明确需要覆盖现有标准化数据时才使用 `--force`。

## 2. 启动 Qwen3-8B vLLM

先修改 [start_vllm_4gpu.sh](start_vllm_4gpu.sh) 中的 `MODEL_PATH`：

```bash
MODEL_PATH="/path/to/Qwen3-8B"
```

然后启动服务：

```bash
bash start_vllm_4gpu.sh
```

脚本默认采用：

- 4 张 GPU；
- Data Parallel = 4；
- Tensor Parallel = 1；
- 每张 GPU 加载一个 Qwen3-8B 副本；
- `max-num-seqs=64`。

这种配置适用于单卡能够容纳模型的情况，目标是提高大量独立 Judge 请求的总吞吐。如果单卡显存不足，可以按照脚本中的示例改为 TP=4、DP=1。

默认服务地址和模型名为：

```text
http://127.0.0.1:8000/v1
Qwen/Qwen3-8B
```

## 3. 运行评测

### 冒烟评测

不传 `--smoke-per-group` 时，每个数据分组默认抽取 2 条，用于验证完整链路：

```bash
python -m reward_harness.benchmark \
  --benchmarks rewardbench2 rmbench \
  --agents no_skill
```

### 全量评测

`--smoke-per-group 0` 表示加载全部样本：

```bash
python -m reward_harness.benchmark \
  --benchmarks rewardbench rewardbench2 rmbench \
  --agents no_rubric no_skill init_skill \
  --smoke-per-group 0
```

如果只想从全体 case 中随机抽取固定数量，可以使用：

```bash
python -m reward_harness.benchmark \
  --benchmarks rewardbench2 \
  --agents no_skill \
  --smoke-per-group 0 \
  --sample-size 3000 \
  --seed 42
```

如果 vLLM 位于另一台服务器：

```bash
python -m reward_harness.benchmark \
  --base-url http://SERVER_IP:8000/v1 \
  --model Qwen/Qwen3-8B \
  --benchmarks rmbench \
  --agents no_rubric \
  --smoke-per-group 0
```

也可以直接使用仓库中的脚本：

```bash
bash run_reward_benchmarks_vllm.sh
bash run_rewardbench_vllm.sh
bash run_rewardbench2_vllm.sh
bash run_rmbench_vllm.sh
```

## 内置 Agent / Baseline

### `no_rubric`

不调用 `build_rubrics()` 中的模型生成逻辑，而是对每个候选回答直接请求一个 0～1 标量 reward。它用于衡量“指定模型直接打分”的基础能力。

### `no_skill`

先根据当前任务生成 task-specific Rubric，再使用同一个 `RubricSet` 分别评价所有候选。它不执行 Skill 选择，是标准的 rubric-based baseline。

### `init_skill`

在生成 Rubric 和评分前动态选择 workflow Skill，并把对应提示内容注入模型 Prompt，包括任务目标、约束和 pointwise evidence 等指导。Skill 本身不执行工具调用；额外成本来自 Skill 选择所需的模型请求，因此耗时和 token 用量高于 `no_skill`。

## RewardSystem 接口

自定义 Agent 需要继承 `RewardSystem`，核心接口为：

```python
class MyHarness(RewardSystem):
    def get_skill_registry(self, task: Query) -> SkillRegistry:
        ...

    def build_rubrics(self, task: Query) -> RubricSet:
        ...

    def score(
        self,
        task: Query,
        candidate: Response,
        rubrics: RubricSet,
    ) -> RewardResult:
        ...
```

`RewardSystem` 不规定 Judge 原始分数的尺度或聚合公式。Harness 在 `score()` 中自行解析 Judgment、执行聚合并生成 `RewardResult`；例如可以使用 0～5 加权平均、0～10 平均、最低项或硬约束策略。公共协议只要求每条 Rubric 恰好有一项 Judgment，并且最终 `RewardResult.reward` 归一化到 `[0,1]`。

Benchmark runner 直接编排这三个接口：每条 Query 调用一次 `build_rubrics()`，再用同一个 `RubricSet` 并发调用每个 Response 的 `score()`。这样 runner 可以统一处理分阶段重试、结果校验、完整轨迹和中断续跑。

把新的实现保存为 `reward_harness/agents/my_harness.py` 后，runner 会自动发现其中的 `RewardSystem` 子类。文件名 `my_harness` 就是 `--agents my_harness` 使用的名称。`__init__.py` 和以下划线开头的文件不会被扫描。

## Benchmark Adapter

所有数据集先转换为统一的 `BenchmarkCase`：

```text
BenchmarkCase
├── task                 # 模型可见的任务信息
├── candidates           # 待评分候选
├── group                # subset/domain 等分组
└── gold                 # 仅 evaluator 可见的正确答案或标签
```

`gold` 不会进入 Query、Response 或模型 prompt。新增数据集时，实现 `BenchmarkAdapter` 的以下方法：

- `load_cases()`：读取标准化数据并执行抽样；
- `score_outcome()`：计算单条样本的指标贡献；
- `summarize()`：按数据集规则汇总结果。

然后在 `benchmarks/__init__.py` 的 `ADAPTERS` 中注册即可，不需要修改 runner 主流程。

## 并发模型

Runner 使用两层并发：

- `--workers`：同时处理多少道 benchmark 题目，默认 4；
- `--request-workers`：全局最多允许多少个在途 LLM 请求，默认 16。

同一道题完成一次 Rubric 生成后，它的候选评分请求可以并发执行。Rubric、Skill 选择和 Judge 请求都会经过同一个全局请求上限，因此不会分别无限创建请求。

对四卡 DP=4 的 Qwen3-8B，可以从下面的配置开始调试：

```bash
--workers 16 --request-workers 64
```

实际最佳值取决于 prompt 长度、候选数量、vLLM 的 `max-num-seqs` 和 GPU 利用率。更高并发不一定更快；如果排队时间、KV cache 压力或输出长度明显上升，应降低并发。

## 结果、轨迹与断点续跑

轨迹日志和最终结果分开保存：

```text
logs/reward_agent/{benchmark}/{agent}/{model}_{signature}/
├── config.json
└── trajectories.jsonl

results/reward_agent/{benchmark}/{agent}/{model}_{signature}/
├── config.json
└── summary.json
```

- `trajectories.jsonl`：唯一的完整轨迹文件。每行独立保存 Query、Responses、evaluator-only gold、Harness metadata、Rubrics、Judgments、Reward、完整模型请求响应、token、延迟、错误和 benchmark 单题结果，可直接作为 Harness Optimization 的输入。
- `summary.json`：benchmark 指标、错误数、token/延迟统计、运行签名和轨迹文件位置。
- `config.json`：脱敏后的模型配置、数据配置、Agent 文件及源码 SHA-256。

成功的模型响应还会缓存在：

```text
logs/reward_agent/.llm_cache/
```

Runner 默认自动续跑：

- 如果同一运行签名已有完整 `summary.json`，直接跳过；
- 如果只存在部分 `trajectories.jsonl`，从未完成的 case 继续；
- 修改 Agent、数据、模型或抽样配置会生成新的运行签名和独立目录；
- `--resume` 仅作为兼容参数保留，不需要显式传入；
- 使用 `--force` 可以强制重新评测相同配置。

API 请求或 JSON/schema 解析在重试后仍失败时，该 case 会记录 `error` 并按错误计分，runner 会继续处理其他样本。

## 常用参数

```text
--benchmarks       rewardbench rewardbench2 rmbench
--agents           指定 Agent；省略时自动运行 reward_harness/agents/ 下的全部 Agent
--agents-dir       自定义 Agent 目录
--workers          题目级并发数，默认 4
--request-workers  全局 LLM 请求并发上限，默认 16
--smoke-per-group  每组抽样数，默认 2；0 表示全量
--sample-size      全局随机抽样数，默认 0，即不额外抽样
--stage-retries    Rubric/Judge 等阶段失败后的重试次数，默认 2
--base-url         vLLM OpenAI-compatible API 地址
--model            服务端模型名，默认 Qwen/Qwen3-8B
--data-dir         标准化 benchmark 数据目录
--logs-dir         轨迹目录，默认 logs/reward_agent
--results-dir      汇总结果目录，默认 results/reward_agent
--force            强制重跑相同配置
--skip-preflight   跳过启动前的 vLLM 单请求检查
```

查看完整参数：

```bash
python -m reward_harness.benchmark --help
python -m reward_harness.prepare_data --help
```

更详细的 benchmark 说明见 [BENCHMARK.md](BENCHMARK.md)。
