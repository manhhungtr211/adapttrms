#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────
# train.sh  —  SFT training for TinyRecursiveModel
#
# Usage:
#   bash scripts/train.sh [TASK] [NUM_EPOCHS] [BATCH_SIZE] [DIM] [N_GPU]
#
# Examples:
#   # Train on NER only, 1 GPU
#   bash scripts/train.sh NER 10 4 256 1
#
#   # Train on all finance tasks, 2 GPUs
#   bash scripts/train.sh "NER+FPB+FiQA_SA+Headline+ConvFinQA" 10 8 256 2
#
# After training completes, update inference.yaml:
#   checkpoint_path: ./checkpoints/NER_sft_v1/best_ep<N>_step<S>.pt
# ───────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TASK="${1:-NER}"
NUM_EPOCHS="${2:-10}"
BATCH_SIZE="${3:-4}"
DIM="${4:-256}"
N_GPU="${5:-1}"

CKPT_DIR="./checkpoints/${TASK}_sft_v1"
CACHE_DIR="/tmp/cache"

echo "========================================="
echo "  TinyRecursiveModel SFT Training"
echo "========================================="
echo "  Task(s)    : ${TASK}"
echo "  Epochs     : ${NUM_EPOCHS}"
echo "  Batch size : ${BATCH_SIZE}"
echo "  Dim        : ${DIM}"
echo "  GPUs       : ${N_GPU}"
echo "  Output     : ${CKPT_DIR}"
echo "========================================="

if [ "${N_GPU}" -gt 1 ]; then
    LAUNCHER="accelerate launch --num_processes ${N_GPU} --multi_gpu"
else
    LAUNCHER="accelerate launch --num_processes 1"
fi

${LAUNCHER} train.py \
    task_name="${TASK}" \
    num_epochs="${NUM_EPOCHS}" \
    batch_size="${BATCH_SIZE}" \
    dim="${DIM}" \
    output_dir="${CKPT_DIR}" \
    cache_dir="${CACHE_DIR}" \
    hydra.run.dir=/tmp

echo ""
echo "Training complete. Checkpoints saved to: ${CKPT_DIR}"
echo ""
echo "To run NER inference with the trained model, use:"
echo "  bash scripts/inference.sh NER tiny_recursive False False 1 custom 50257 \\"
echo "    checkpoint_path=${CKPT_DIR}/best_*.pt \\"
echo "    n_tokens=1024 generate_max_len=128"
