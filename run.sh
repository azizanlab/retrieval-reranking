#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash run.sh {qwen3-8b|gpt-oss-20b} [instance-limit]" >&2
    exit 2
fi
case "$1" in
    qwen3-8b|gpt-oss-20b) MODEL=$1 ;;
    *) echo "Unknown model: $1" >&2; exit 2 ;;
esac
LIMIT=()
if [[ $# -eq 2 ]]; then
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || { echo "Instance limit must be positive." >&2; exit 2; }
    LIMIT=(--limit "$2")
fi

cd "$(dirname "$0")"
export RERANK_RESULTS_DIR=${RERANK_RESULTS_DIR:-"$PWD/runs"}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
if [[ -z "${GPUS:-}" ]]; then
    command -v nvidia-smi >/dev/null || { echo "No NVIDIA GPU tools found. Set GPUS for a configured CUDA environment." >&2; exit 1; }
    GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')
fi
read -r -a GPU_IDS <<< "$GPUS"
[[ ${#GPU_IDS[@]} -gt 0 ]] || { echo "No GPUs selected." >&2; exit 1; }
LOGS="$RERANK_RESULTS_DIR/$MODEL/logs"
mkdir -p "$LOGS"
exec 9>"$RERANK_RESULTS_DIR/$MODEL/.run.lock"
flock -n 9 || { echo "Another runner is using this model output directory." >&2; exit 1; }

DEFAULT_STAGES="think:fulllist:grid score:fulllist:grid score:fulllist:selection score:shortlist score:availability"
if [[ "$MODEL" == qwen3-8b ]]; then
    DEFAULT_STAGES+=" score:fulllist:omit"
fi
for stage in ${STAGES:-$DEFAULT_STAGES}; do
    step=${stage%%:*}
    experiment=${stage#*:}
    pids=()
    for i in "${!GPU_IDS[@]}"; do
        CUDA_VISIBLE_DEVICES=${GPU_IDS[$i]} python -m rerank.engine "$step" \
            --model "$MODEL" --exp "$experiment" --shard "$i/${#GPU_IDS[@]}" "${LIMIT[@]}" \
            >"$LOGS/${stage//:/-}.$i.log" 2>&1 &
        pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    [[ $status -eq 0 ]] || { echo "Stage failed: $stage (see $LOGS)" >&2; exit 1; }
    echo "Completed $stage"
done
python -m analysis.report --model "$MODEL" "${LIMIT[@]}"
