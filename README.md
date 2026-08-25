# Reward-Harness

Reward-Harness 是一个面向生成式 Reward Model / LLM-as-a-Judge 的评测框架。它使用统一的 `RewardSystem` 接口组织任务、Rubric 生成、pairwise 判断和 winner 输出，并提供 RewardBench、RewardBench 2、RM-Bench 的本地 vLLM 评测流程。

当前默认模型为 `Qwen/Qwen3-8B`。正式评测只连接本地或自建的 OpenAI-compatible vLLM 服务，不包含 SiliconFlow 等云 API 调用逻辑。

## 主要功能

- 统一的 Reward System 数据结构和调用接口。
- 每道题只生成一次共享 `RubricSet`，随后比较完整候选回答并输出唯一 winner。
- 自动发现 `reward_harness/agents/` 中的 Reward Harness，无需修改 runner。
- 支持有 Rubric、动态 Skill 和无 Rubric 直接判断等 baseline。
- 支持 RewardBench、RewardBench 2 和 RM-Bench 官方风格指标。
- 数据准备与模型评测解耦；评测阶段不会隐式下载数据。
- 双层并发、请求限流、LLM 响应缓存、错误重试和中断续跑。
- 保存逐题轨迹、原始模型响应、Rubric、WinnerResult、token、延迟和汇总指标。
- 使用顶层时间 tag 隔离每次实验；复用同一 tag 可以断点续跑。

## 目录结构

```text
Reward-Harness/
├── reward_harness/                 # 可直接 import 的 Python 包
│   ├── agents/                     # Reward Harness 实现
│   │   ├── no_rubric.py            # 官方 vanilla pairwise judging
│   │   ├── no_skill.py             # 官方 online-rubric pairwise judging
│   │   ├── init_skill_no_rubric.py # 固定 J-stage Skill，不生成 Rubric
│   │   └── init_skill.py           # G-stage Rubric Skill + response-aware Rubrics
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
- `PyYAML`：读取 Meta-Harness 的 `config.yaml`。

vLLM 请根据服务器 CUDA、PyTorch 和驱动环境单独安装。

## 1. 准备本地数据

正式评测不会联网下载数据。首次运行前，先将原始数据集转换成统一的本地 JSONL：

```bash
python -m reward_harness.prepare_data \
  --benchmarks helpsteer3 rewardbench rmbench
```

默认输出到：

```text
data/{benchmark}/{split}.jsonl
data/{benchmark}/{split}.manifest.json
```

manifest 会记录数据条数、来源 fingerprint 和内容 SHA-256，用于校验数据完整性与生成可复现的运行签名。

HelpSteer3 使用 `preference/train` 中四个 domain 的非平局样本做均衡随机抽样：

```text
data/held_in/train.jsonl             500 条，每域 125 条
```

`overall_preference < 0` 表示原 Response 1 胜，`> 0` 表示 Response 2 胜；本地格式始终把 chosen 放在 `candidate_000`，原始偏好强度和 annotator reasoning 保存在 evaluator-only gold。

如果 Hugging Face 数据已经存在于本机缓存，并且不希望程序访问网络：

```bash
python -m reward_harness.prepare_data \
  --benchmarks helpsteer3 rewardbench rmbench \
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
  --benchmarks rewardbench \
  --agents no_rubric
```

### 全量评测

`--smoke-per-group 0` 表示加载全部样本：

```bash
python -m reward_harness.benchmark \
  --benchmarks rewardbench \
  --agents no_rubric no_skill init_skill_no_rubric init_skill \
  --smoke-per-group 0
```

如果只想从全体 case 中随机抽取固定数量，可以使用：

```bash
python -m reward_harness.benchmark \
  --benchmarks rewardbench \
  --agents no_rubric \
  --smoke-per-group 0 \
  --sample-size 3000 \
  --seed 42
```

如果 vLLM 位于另一台服务器：

```bash
python -m reward_harness.benchmark \
  --base-url http://SERVER_IP:8000/v1 \
  --model Qwen/Qwen3-8B \
  --benchmarks rewardbench \
  --agents no_rubric \
  --smoke-per-group 0
```

也可以直接使用仓库中的脚本（先启动 vLLM，再运行 benchmark）：

```bash
bash start_vllm_4gpu.sh
bash run_benchmark.sh
```

## 内置 Agent / Baseline

### `no_rubric`

使用 Eval-Skill 官方 vanilla pairwise Prompt，一次查看完整的 Response A/B 并强制选择唯一 winner。

### `no_skill`

先使用官方 Rubric Generation Prompt 根据 Query 在线生成 Rubric，再使用官方 rubric pairwise Prompt 比较完整 A/B 并选择 winner。

### `init_skill_no_rubric`

把可由 Harness Optimization 编辑的离线 Skill 注入官方 skill pairwise Prompt，比较完整 A/B 并选择 winner。

### `init_skill`

把可编辑的 G-stage Rubric Skill 注入 Rubric Model；Rubric Model 同时读取 Query 和完整匿名 Responses，生成结构化、判别性的 RubricSet，再由 Judge 使用这些 Rubrics 比较完整 A/B。

接近原项目 Qwen3-8B 推理设置的运行方式为：

```bash
python -m reward_harness.benchmark \
  --benchmarks rewardbench \
  --agents no_rubric no_skill init_skill_no_rubric init_skill \
  --smoke-per-group 0 \
  --trial-num 3 \
  --temperature 0.7 \
  --max-tokens 10000
```

当前 winner-only runner 只使用 pairwise Prompt，支持 RewardBench，以及展开为 9 个 pair 的 RM-Bench。RewardBench 2 暂不参与评测。

每个 Skill 包含 `name`、`stage`、`description` 和 `content`，`stage` 取 `G` 或 `J`。`init_skill_no_rubric` 注入 J-stage comparative workflow；`init_skill` 注入 G-stage Rubric workflow。

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

    def judge(
        self,
        task: Query,
        responses: tuple[Response, ...],
        rubrics: RubricSet,
    ) -> WinnerResult:
        ...
```

Benchmark runner 会稳定重排并匿名化完整 Responses，再依次调用 `build_rubrics()` 和 `judge()`。`judge()` 直接返回绑定到匿名 Response 的 `WinnerResult`；evaluator 在模型调用结束后映射回原始 Response ID 并计算 RewardBench accuracy。

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

Runner 使用请求限流和 case 级并发：

- `--workers`：同时处理多少道 benchmark 题目，默认 4；
- `--request-workers`：全局最多允许多少个在途 LLM 请求，默认 16。

每道题把完整匿名 A/B 放进一次 comparative Judge 调用；`no_skill` 额外执行一次在线 Rubric Generation。所有请求共用 `--request-workers` 全局上限。

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

- `trajectories.jsonl`：唯一的完整轨迹文件。每行独立保存 Query、Responses、evaluator-only gold、Rubrics、WinnerResult、完整模型请求响应、token、延迟、错误和 benchmark 单题结果。多次 trial 时用 `trial_index` 标记所属轮次。
- `summary.json`：`primary_metrics` 保存关键指标，`metrics` 保存其余 benchmark 指标，`trials` 保存每轮结果，`voting` 保存 Voting@N，`counts`、`usage` 和 `artifacts` 分别保存计数、开销和轨迹路径。
- `config.json`：脱敏后的模型配置、数据配置、Agent 文件及源码 SHA-256。

成功的模型响应还会缓存在：

```text
results/{run_tag}/.llm_cache/
```

Runner 在同一个时间 tag 内自动续跑：

- 如果当前 tag 下已有完整 `summary.json`，直接跳过；
- 如果只存在部分 `trajectories.jsonl`，从未完成的 `(trial_index, case_id)` 继续；
- 默认 tag 是启动时的本地时间 `YYYYMMDD_HHMMSS`；
- 中断后使用同一个 `--run-tag` 才会继续原目录；
- 修改 Agent、数据、模型或抽样配置时应使用新的 tag；
- 使用 `--force` 会清空同一 tag 下的已有轨迹并重新评测。

API 请求或 JSON/schema 解析在重试后仍失败时，该 case 会记录 `error` 并按错误计分，runner 会继续处理其他样本。

## 常用参数

```text
--benchmarks       held_in rewardbench rmbench（rewardbench2 暂不评测）
--agents           指定 Agent；省略时自动运行 reward_harness/agents/ 下的全部 Agent
--agents-dir       自定义 Agent 目录
--workers          题目级并发数，默认 4
--request-workers  全局 LLM 请求并发上限，默认 16
--smoke-per-group  每组抽样数，默认 2；0 表示全量
--sample-size      全局随机抽样数，默认 0，即不额外抽样
--stage-retries    Rubric/Judge 等阶段失败后的重试次数，默认 2
--trial-num        完整独立运行整套评测 N 次，同时计算 Average@N 和 Voting@N，默认 1
--temperature      模型采样温度；Average@N > 1 时建议使用正数
--max-tokens       单次模型响应最大 token 数，默认 2048
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

## Harness Optimization（Codex CLI）

`meta-harness.py` 实现 propose → candidate check → benchmark → frontier update 外循环。Codex CLI 只负责读取历史轨迹、原型机制并新增 3 个候选 Harness；可信 benchmark 始终由外循环单独执行。

默认配置位于根目录 `config.yaml`，包含 run、Codex、benchmark 和路径设置。搜索阶段的 `trial_num` 不在 YAML 中配置，始终固定为 1。优先级为：

```text
命令行参数 > VLLM_BASE_URL 环境变量 > config.yaml
```

`datasets` 中统一填写逻辑数据集名称，实际目录由 `paths.data_dir` 解析。例如 `held_in: held_in` 对应 `data/held_in/`，具体 split 文件由 adapter 或读取方确定。

服务器先完成 Codex CLI 登录并启动本地 vLLM，然后从仓库根目录运行：

```bash
python meta-harness.py \
  --config config.yaml \
  --run-name reward-search \
  --iterations 5 \
  --smoke-per-group 0 \
  --sample-size 300
```

`config.yaml` 默认使用 `gpt-5.6-sol` 和 `low` reasoning effort；可通过 `--codex-model`、`--codex-reasoning-effort` 临时覆盖。继续使用同一个 `--run-name` 会根据 `evolution_summary.jsonl` 自动续跑；查看状态：

```bash
python meta-harness.py --run-name reward-search --status
```

搜索阶段固定使用 `trial_num=1`，直接读取 `metrics.domain_scores`，分别维护四个领域的最优 Harness，并以四领域宏平均更新全局 `_best`。`primary_metrics.beyond_rubric` 仍保留在正式 benchmark 结果中，但不作为 Meta-Harness 的读取入口。Average@N 和 Voting@N 应在选出最终 Harness 后，通过独立的正式 benchmark 命令评测。

Harness Optimization 使用单一的 500 条 `held_in` 搜索集。四个 baseline 和每轮所有候选都在同一集合上运行；该集合的指标用于更新 frontier，完整轨迹同时供下一轮 Codex 分析。RewardBench 和 RM-Bench 作为 held-out 数据，不参与搜索阶段。

因此 outer-loop 的调用顺序是：

```text
第 0 步：初始化 baseline
如果当前 run 还没有 frontier_val.json：
  meta-harness.py 调 benchmark.py
  在 held-in/search benchmarks 上评测 baseline harnesses，比如 no_rubric / no_skill / init_skill_no_rubric / init_skill
  然后把当前最高 avg_val 的 harness 写进 frontier_val.json
  同时把 baseline 结果写进 evolution_summary.jsonl

第 1 步：coding agent propose
每一轮 iteration：
  meta-harness.py 调 Codex CLI
  Codex CLI 加载 .claude/skills/meta-harness-reward-skill/SKILL.md
  coding agent 读取 evolution_summary.jsonl、frontier_val.json、recent trajectories、当前 benchmark/config/base harness
  它可以看全部 recent trajectories，不限于 frontier harness
  但 frontier_val.json 告诉它当前最优 harness 是谁

第 2 步：coding agent 写 candidates
coding agent 基于当前 top-performing/base harness 做修改
生成 3 个新的 reward skill harness 文件：reward_harness/agents/<candidate>.py
然后写 pending_eval.json，包含 name、file、hypothesis、axis、base_harness、components

第 3 步：candidate check
meta-harness.py 读取 pending_eval.json
对每个 candidate 做轻量检查：
  名字是否合法
  file 路径是否匹配
  文件是否存在
  Python 文件能不能 import
  是否正好定义一个 RewardSystem subclass / HARNESS_CLASS
  是否实现 get_skill_registry / build_rubrics / judge
这一步不是 benchmark，不是 held-out，不是验证集评测，只是 candidate check / import check / interface check

第 4 步：评测 candidates
通过 candidate check 的 candidates 会被传给 benchmark.py
benchmark.py 在 held-in/search benchmarks 上跑这些 candidates
生成 summary.json 和 trajectories.jsonl
meta-harness.py 从 summary.json 里读 metric，算 avg_val

第 5 步：记录和更新 frontier
meta-harness.py 把所有 evaluated candidates 的结果追加到 evolution_summary.jsonl
然后和当前 frontier 比较：
  如果 candidate avg_val > 当前 frontier avg_val：
    更新 frontier_val.json
  否则：
    结果保留在 evolution_summary.jsonl
    但不会成为 frontier

第 6 步：下一轮
下一轮 coding agent 再读取新的 evolution_summary.jsonl、frontier_val.json、recent trajectories，继续 propose 新 candidates
```

每个 run 的外循环状态位于：

```text
meta_runs/{run_name}/
├── evolution_summary.jsonl
├── frontier_val.json
├── pending_eval.json
├── reports/
├── codex_sessions/{iteration}/
└── benchmark_logs/
```

模型完整轨迹仍保存在原有路径：

```text
results/mh-{run_name}-{baseline|iter-NNN}/{benchmark}/{harness}/{model}/
```

`--fresh` 不直接删除历史，而是把原 run 状态重命名为带时间戳的 `.bak_*` 目录。`--propose-only` 可用于只验证 Codex 候选生成链路而不启动 benchmark。

更详细的 benchmark 说明见 [BENCHMARK.md](BENCHMARK.md)。
