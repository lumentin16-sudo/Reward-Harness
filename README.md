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
- 使用顶层时间 tag 隔离每次实验；复用同一 tag 可以断点续跑。

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

标准化 benchmark 数据位于 `data/` 并随仓库提交；每次运行的轨迹和结果统一写入 `results/{run_tag}/`，该目录不会提交到 Git。

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

先根据 Query 和匿名、无标签、稳定重排的完整 Responses 生成具有区分度且可二值判断的共享 Rubric。评分时每条 Rubric 独立调用一次 Judge，只输出 0（FAIL）或 1（PASS）；A 阶段按 Rubric 权重计算加权通过率。它不执行 Skill 选择。

### `init_skill`

在生成 Rubric 和评分前动态选择 workflow Skill，并把对应提示内容注入模型 Prompt，包括任务目标、约束和 pointwise evidence 等指导。它同样采用逐 Rubric 独立的 0/1 Judge 和加权通过率聚合。Skill 本身不执行工具调用；额外成本来自 Skill 选择所需的模型请求，因此耗时和 token 用量高于 `no_skill`。

每个 Skill 包含 `name`、`stage`、`description` 和 `content`。`stage` 取 `G`、`J` 或 `A`；选择前通过 `SkillRegistry.for_stage()` 构造独立 pool。目前 `task_objective`、`constraint_analysis` 属于 G，`pointwise_evidence` 属于 J，A pool 为空。同名 Skill 可以出现在不同阶段，但同一阶段内不能重名。

## RewardSystem 接口

自定义 Agent 需要继承 `RewardSystem`，核心接口为：

```python
class MyHarness(RewardSystem):
    def get_skill_registry(self, task: Query) -> SkillRegistry:
        ...

    def build_rubrics(
        self,
        task: Query,
        responses: tuple[Response, ...],
    ) -> RubricSet:
        ...

    def score(
        self,
        task: Query,
        candidate: Response,
        rubrics: RubricSet,
    ) -> JudgmentResult:
        ...

    def aggregate(
        self,
        task: Query,
        candidate: Response,
        rubrics: RubricSet,
        judgment_result: JudgmentResult,
    ) -> RewardResult:
        ...
```

`score()` 只负责 J 阶段并返回 `JudgmentResult`；`aggregate()` 负责 A 阶段，把 Judgment 聚合成 `RewardResult`。Harness 可以使用 0～5 加权平均、0～10 平均、最低项或硬约束策略。公共协议只要求每条 Rubric 恰好有一项 Judgment，并且最终 `RewardResult.reward` 归一化到 `[0,1]`。

Benchmark runner 直接编排这四个接口：每条 Query 先将全部 Responses 移除 ID/metadata 并按内容稳定重排，再调用一次 `build_rubrics(query, responses)`；随后用同一个 `RubricSet` 为每个原始 Response 执行 `score()` 和 `aggregate()`。这样 G 可以发现候选间的实质差异，但无法利用原始位置或 gold，J 仍保持单 Response、单 Rubric 评分。

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

同一道题完成一次 Rubric 生成后，不同 Response 可以并发评分；每个 Response 内部按 Rubric 逐条调用 Judge。若一题有 R 个 Response、K 条 Rubric，`no_skill` 需要 R×K 次 Judge 调用，`init_skill` 还会增加 Skill 选择调用。所有请求共用 `--request-workers` 全局上限。

对四卡 DP=4 的 Qwen3-8B，可以从下面的配置开始调试：

```bash
--workers 16 --request-workers 64
```

实际最佳值取决于 prompt 长度、候选数量、vLLM 的 `max-num-seqs` 和 GPU 利用率。更高并发不一定更快；如果排队时间、KV cache 压力或输出长度明显上升，应降低并发。

## 结果、轨迹与断点续跑

每个模型配置的轨迹和汇总结果统一保存在同一目录：

```text
results/{run_tag}/{benchmark}/{harness}/{model}/
├── config.json
├── trajectories.jsonl
└── summary.json
```

- `trajectories.jsonl`：唯一的完整轨迹文件。每行独立保存 Query、Responses、evaluator-only gold、Harness metadata、Rubrics、JudgmentResults、RewardResults、完整模型请求响应、token、延迟、错误和 benchmark 单题结果。Average@N 时用 `repeat_index` 标记所属轮次。
- `summary.json`：顶层保存 N 轮 benchmark 指标的平均值，`repetitions` 保存每一轮的独立指标，同时记录错误数、token/延迟统计和轨迹文件位置。
- `config.json`：脱敏后的模型配置、数据配置、Agent 文件及源码 SHA-256。

成功的模型响应还会缓存在：

```text
results/{run_tag}/.llm_cache/
```

Runner 在同一个时间 tag 内自动续跑：

- 如果当前 tag 下已有完整 `summary.json`，直接跳过；
- 如果只存在部分 `trajectories.jsonl`，从未完成的 `(repeat_index, case_id)` 继续；
- 默认 tag 是启动时的本地时间 `YYYYMMDD_HHMMSS`；
- 中断后使用同一个 `--run-tag` 才会继续原目录；
- 修改 Agent、数据、模型或抽样配置时应使用新的 tag；
- 使用 `--force` 会清空同一 tag 下的已有轨迹并重新评测。

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
--average-n        完整独立运行整套评测 N 次并平均各轮 benchmark 指标，默认 1
--temperature      模型采样温度；Average@N > 1 时建议使用正数
--base-url         vLLM OpenAI-compatible API 地址
--model            服务端模型名，默认 Qwen/Qwen3-8B
--data-dir         标准化 benchmark 数据目录
--output-dir       时间目录的父目录，默认 results
--run-tag          顶层运行目录名，默认 YYYYMMDD_HHMMSS
--force            清空当前 tag 并重新运行
--skip-preflight   跳过启动前的 vLLM 单请求检查
```

查看完整参数：

```bash
python -m reward_harness.benchmark --help
python -m reward_harness.prepare_data --help
```

更详细的 benchmark 说明见 [BENCHMARK.md](BENCHMARK.md)。
