#!/usr/bin/env bash
set -euo pipefail

VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
RUN_NAME="${RUN_NAME:-reward-search-g50-code25-stem25}"
DATA_DIR="${DATA_DIR:-data_g50_code25_stem25}"
SOURCE_DATA="${SOURCE_DATA:-data/held_in/train.jsonl}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_NAME}.log}"
ITERATIONS="${ITERATIONS:-15}"
WORKERS="${WORKERS:-12}"
REQUEST_WORKERS="${REQUEST_WORKERS:-12}"
MAX_TOKENS="${MAX_TOKENS:-10000}"
SEED="${SEED:-42}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date '+%F %T')] Reward-Harness general100 search started"
echo "run_name=$RUN_NAME model=$MODEL base_url=$VLLM_BASE_URL"

echo "[$(date '+%F %T')] Checking vLLM..."
curl -fsS "$VLLM_BASE_URL/models" >/dev/null

echo "[$(date '+%F %T')] Preparing held-in data: general=50, code=25, stem=25..."
mkdir -p "$DATA_DIR/held_in"
python - <<'PY'
import json
import os
import random
from pathlib import Path
from reward_harness.benchmarks.base import BenchmarkCase, write_processed_cases
from reward_harness.reward_system import Query, Response

source = Path(os.environ.get("SOURCE_DATA", "data/held_in/train.jsonl"))
data_dir = Path(os.environ.get("DATA_DIR", "data_general100"))
seed = int(os.environ.get("SEED", "42"))
targets = {
    "general": 50,
    "code": 25,
    "stem": 25,
}

rows_by_group = {group: [] for group in targets}
for line in source.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    obj = json.loads(line)
    group = str(obj.get("group") or obj.get("task", {}).get("domain") or "").lower()
    if group in rows_by_group:
        rows_by_group[group].append(obj)

rng = random.Random(seed)
rows = []
for group, count in targets.items():
    available = rows_by_group[group]
    if len(available) < count:
        raise SystemExit(
            f"Need at least {count} {group} rows, found {len(available)} in {source}"
        )
    rows.extend(rng.sample(available, count))

rng.shuffle(rows)
cases = []
for row in rows:
    cases.append(
        BenchmarkCase(
            case_id=row["case_id"],
            group=row["group"],
            task=Query(**row["task"]),
            candidates=tuple(Response(**candidate) for candidate in row["candidates"]),
            gold=row["gold"],
        )
    )

write_processed_cases(
    data_dir,
    benchmark="held_in",
    dataset_id="helpsteer3-g50-code25-stem25",
    split="train",
    cases=cases,
    source_fingerprint=f"g50-code25-stem25-from-{source}-seed{seed}",
    force=True,
)
print(f"wrote {data_dir / 'held_in' / 'train.jsonl'} rows={len(cases)}")
PY

echo "[$(date '+%F %T')] Starting meta-harness optimization..."
python -u meta-harness.py \
  --config config.yaml \
  --run-name "$RUN_NAME" \
  --iterations "$ITERATIONS" \
  --benchmarks held_in \
  --baselines no_skill init_skill \
  --data-dir "$DATA_DIR" \
  --held-in-dataset held_in \
  --max-tokens "$MAX_TOKENS" \
  --workers "$WORKERS" \
  --request-workers "$REQUEST_WORKERS"

FRONTIER="meta_runs/$RUN_NAME/frontier_val.json"
echo "[$(date '+%F %T')] Reading best harness from $FRONTIER..."
BEST_HARNESS="$(python - <<'PY'
import json
import os
from pathlib import Path

frontier = Path("meta_runs") / os.environ.get("RUN_NAME", "reward-search-general100") / "frontier_val.json"
data = json.loads(frontier.read_text(encoding="utf-8"))
best = data.get("_best", {})
harness = best.get("harness")
if not harness:
    raise SystemExit(f"No _best.harness in {frontier}")
print(harness)
PY
)"
echo "[$(date '+%F %T')] Best harness: $BEST_HARNESS"

echo "[$(date '+%F %T')] Running RewardBench for best harness..."
python -u -m reward_harness.benchmark \
  --benchmarks rewardbench \
  --agents "$BEST_HARNESS" \
  --data-dir data \
  --base-url "$VLLM_BASE_URL" \
  --model "$MODEL" \
  --output-dir results \
  --run-tag "rewardbench-${RUN_NAME}-${BEST_HARNESS}" \
  --smoke-per-group 0 \
  --sample-size 0 \
  --max-tokens "$MAX_TOKENS" \
  --workers "$WORKERS" \
  --request-workers "$REQUEST_WORKERS" \
  --seed "$SEED"

echo "[$(date '+%F %T')] Done."
