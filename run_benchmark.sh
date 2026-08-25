#!/usr/bin/env bash
set -euo pipefail

VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

# reward_harness.benchmark 参数说明：
#   --base-url          vLLM 的 OpenAI-compatible API 地址。
#   --model             vLLM 暴露的模型名称。
#   --benchmarks        可选 rewardbench、rmbench、held_in；暂不评测 rewardbench2。
#   --agents            可选 no_rubric、no_skill、init_skill_no_rubric、init_skill。
#   --agents-dir        自定义 Agent 文件目录。
#   --workers           同时处理的 benchmark case 数量。
#   --request-workers   全局同时发送给 vLLM 的最大请求数。
#   --smoke-per-group   每个分组抽取的题数；0 表示全量评测。
#   --sample-size       从加载后的全部 case 中随机抽取 N 条；0 表示不额外抽样。
#   --seed              抽样随机种子。
#   --stage-retries     Rubric/Judge 阶段失败后的重试次数。
#   --trial-num         完整独立运行整套 benchmark N 次，同时计算 Average@N 和 Voting@N；默认 1。
#   --temperature       模型采样温度；Average@N > 1 时建议设为正数以产生独立采样。
#   --max-tokens        单次模型响应的最大 token 数，默认 2048。
#   --data-dir          标准化 benchmark 数据目录，默认 data/。
#   --output-dir        时间目录的父目录，默认 results/。
#   --run-tag           顶层时间目录名；复用同一 tag 可断点续跑。
#   --force             清空同一 tag 下已有轨迹并重新运行。
#   --skip-preflight    跳过运行前的 vLLM 健康检查。
#
python -u -m reward_harness.benchmark \
  --base-url "$VLLM_BASE_URL" \
  --model Qwen/Qwen3-8B \
  --benchmarks rewardbench \
  --agents no_rubric no_skill init_skill_no_rubric init_skill \
  --temperature 0.7 \
  --max-tokens 10000 \
  --workers 32 \
  --request-workers 32 \
  --smoke-per-group 0 \
  --trial-num 3 \
  --seed 42 \
  --output-dir results \
  --run-tag "$RUN_TAG"
