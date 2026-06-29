#!/usr/bin/env bash
set -euo pipefail

PROVIDER="${PROVIDER:-prime}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
SPLIT="${SPLIT:-train}"
HORIZON_DAYS="${HORIZON_DAYS:-2}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1}"
ROLLOUTS_PER_EXAMPLE="${ROLLOUTS_PER_EXAMPLE:-1}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-rl/artifacts/evals/m0-qwen}"
TRACE_DIR="${TRACE_DIR:-rl/artifacts/traces/m0-qwen}"
if [[ -z "${API_KEY_VAR:-}" ]]; then
  if [[ "$PROVIDER" == "prime" ]]; then
    API_KEY_VAR="PRIME_API_KEY"
  else
    API_KEY_VAR="OPENAI_API_KEY"
  fi
fi
API_BASE_URL="${API_BASE_URL:-}"
PRIME_TEAM_ID="${PRIME_TEAM_ID:-}"
SAMPLING_ARGS="${SAMPLING_ARGS:-}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"

mkdir -p "$OUTPUT_DIR" "$TRACE_DIR"

if [[ "$PROVIDER" == "vllm" && -z "$API_BASE_URL" ]]; then
  echo "PROVIDER=vllm requires API_BASE_URL, e.g. http://<host>:8000/v1" >&2
  exit 2
fi

if [[ -z "${!API_KEY_VAR:-}" ]]; then
  echo "Missing API key environment variable: $API_KEY_VAR" >&2
  exit 2
fi

env_args=$(
  HORIZON_DAYS="$HORIZON_DAYS" SPLIT="$SPLIT" TRACE_DIR="$TRACE_DIR" uv run --extra dev --extra rl python - <<'PY'
import json
import os

print(json.dumps({
    "horizon_days": int(os.environ["HORIZON_DAYS"]),
    "split": os.environ["SPLIT"],
    "trace_dir": os.environ["TRACE_DIR"],
}))
PY
)

cmd=(
  uv run --extra dev --extra rl vf-eval entrepreneur_bench
  --provider "$PROVIDER"
  --model "$MODEL"
  --api-client-type openai_chat_completions
  --api-key-var "$API_KEY_VAR"
  --num-examples "$NUM_EXAMPLES"
  --rollouts-per-example "$ROLLOUTS_PER_EXAMPLE"
  --max-concurrent "$MAX_CONCURRENT"
  --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
  --output-dir "$OUTPUT_DIR"
  --env-args "$env_args"
  --state-columns trace_path
  --save-results
  --independent-scoring
  --disable-tui
  --timeout "$TIMEOUT_SECONDS"
)

if [[ -n "$API_BASE_URL" ]]; then
  cmd+=(--api-base-url "$API_BASE_URL")
fi

if [[ -n "$PRIME_TEAM_ID" ]]; then
  cmd+=(--header "X-Prime-Team-ID: $PRIME_TEAM_ID")
fi

if [[ -n "$SAMPLING_ARGS" ]]; then
  cmd+=(--sampling-args "$SAMPLING_ARGS")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
