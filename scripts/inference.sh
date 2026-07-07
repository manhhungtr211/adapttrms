#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# inference.sh  —  Unified inference + eval for TinyRecursiveModel / HF models
#
# Positional args (must keep this order):
#   $1  DOMAIN        : 'finance' | 'biomedicine' | 'law' | single task name
#   $2  MODEL         : HF model name OR 'tiny_recursive' for custom model
#   $3  ADD_BOS       : True | False
#   $4  MODEL_PARALLEL: True | False
#   $5  N_GPU         : 1 | 2 | 4 | 8
#   $6  MODEL_TYPE    : 'huggingface' | 'custom'   (default: huggingface)
#
# Extra key=value args (any order, passed as Hydra overrides):
#   vocab_size=<N>          vocabulary size (default: 50000)
#   checkpoint_path=<path>  trained .pt checkpoint (custom model only)
#   output_dir=<path>       prediction output dir  (default: /tmp/output)
#   res_dir=<path>          eval results dir       (default: /tmp/res)
#   cache_dir=<path>        HF cache dir           (default: /tmp/cache)
#   dim=<N>                 model hidden dim (custom only, default: 256)
#   n_layers=<N>            (custom only, default: 2)
#   n_heads=<N>             (custom only, default: 4)
#   n_latent_recursions=<N> (custom only, default: 3)
#   n_improvement_cycles=<N>(custom only, default: 2)
#   ... any other Hydra override
#
# Examples:
#   # Minimal — your existing Kaggle command (unchanged):
#   DOMAIN='finance' MODEL_NAME='tiny_recursive' ADD_BOS_TOKEN=False \
#   MODEL_PARALLEL=True N_GPU=2 MODEL_TYPE='custom' \
#   bash scripts/inference.sh $DOMAIN $MODEL_NAME $ADD_BOS_TOKEN \
#       $MODEL_PARALLEL $N_GPU $MODEL_TYPE \
#       vocab_size=50000 output_dir=/tmp/output res_dir=/tmp/res cache_dir=/tmp/cache
#
#   # With a trained checkpoint (add one extra arg):
#   ... checkpoint_path=/kaggle/working/checkpoints/NER_sft_v1/best_ep10_step400.pt
# ───────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Positional arguments ─────────────────────────────────────────────────────
DOMAIN="${1:?'ERROR: DOMAIN (arg 1) is required'}"
MODEL="${2:?'ERROR: MODEL (arg 2) is required'}"
ADD_BOS="${3:?'ERROR: ADD_BOS_TOKEN (arg 3) is required'}"
MODEL_PARALLEL="${4:?'ERROR: MODEL_PARALLEL (arg 4) is required'}"
N_GPU="${5:?'ERROR: N_GPU (arg 5) is required'}"
MODEL_TYPE="${6:-huggingface}"

# ── Defaults for extractable extra args ─────────────────────────────────────
VOCAB_SIZE="50000"
CHECKPOINT_PATH="null"
OUTPUT_DIR="/tmp/output"
RES_DIR="/tmp/res"
CACHE_DIR="/tmp/cache"

# ── Parse remaining extra args ───────────────────────────────────────────────
# Known keys are extracted into shell variables so they can be printed/logged.
# Everything else passes through as raw Hydra overrides.
shift 6
EXTRA_ARGS=()
for arg in "$@"; do
    case "${arg}" in
        vocab_size=*)       VOCAB_SIZE="${arg#vocab_size=}" ;;
        checkpoint_path=*)  CHECKPOINT_PATH="${arg#checkpoint_path=}" ;;
        output_dir=*)       OUTPUT_DIR="${arg#output_dir=}" ;;
        res_dir=*)          RES_DIR="${arg#res_dir=}" ;;
        cache_dir=*)        CACHE_DIR="${arg#cache_dir=}" ;;
        *)                  EXTRA_ARGS+=("${arg}") ;;
    esac
done

# ── Resolve checkpoint_path to absolute ──────────────────────────────────────
# CRITICAL: hydra.run.dir=/tmp changes cwd BEFORE Python code runs.
# Relative paths like ./checkpoints/... would be resolved as /tmp/checkpoints/...
# We must convert to absolute path HERE in the shell, before that happens.
if [ "${CHECKPOINT_PATH}" != 'null' ] && [ "${CHECKPOINT_PATH}" != 'None' ] && [ -n "${CHECKPOINT_PATH}" ]; then
    if [[ "${CHECKPOINT_PATH}" != /* ]]; then
        # It's a relative path — resolve it from the repo root
        CHECKPOINT_PATH="$(realpath "${CHECKPOINT_PATH}" 2>/dev/null || echo "${REPO_ROOT}/${CHECKPOINT_PATH}")"
        echo "[inference.sh] Resolved checkpoint to absolute path: ${CHECKPOINT_PATH}"
    fi
fi

# ── Task mapping ─────────────────────────────────────────────────────────────
if   [ "${DOMAIN}" == 'biomedicine' ]; then TASK='MQP+PubMedQA+RCT+USMLE+ChemProt'
elif [ "${DOMAIN}" == 'finance' ];     then TASK='NER+FPB+FiQA_SA+Headline+ConvFinQA'
elif [ "${DOMAIN}" == 'law' ];         then TASK='CaseHOLD+SCOTUS+UNFAIR_ToS'
else                                        TASK="${DOMAIN}"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo "========================================="
echo "  AdaptTRMs Inference + Eval"
echo "========================================="
echo "  Domain / Task(s) : ${DOMAIN} → ${TASK}"
echo "  Model            : ${MODEL}"
echo "  Model type       : ${MODEL_TYPE}"
echo "  Add BOS token    : ${ADD_BOS}"
echo "  Model parallel   : ${MODEL_PARALLEL}"
echo "  N_GPU            : ${N_GPU}"
echo "  Vocab size       : ${VOCAB_SIZE}"
echo "  Checkpoint       : ${CHECKPOINT_PATH}"
echo "  Output dir       : ${OUTPUT_DIR}"
echo "  Res dir          : ${RES_DIR}"
echo "  Cache dir        : ${CACHE_DIR}"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "  Extra Hydra args : ${EXTRA_ARGS[*]}"
fi
echo "========================================="

if [ "${MODEL_TYPE}" == 'custom' ] && [ "${CHECKPOINT_PATH}" == 'null' ]; then
    echo ""
    echo "[WARNING] model_type=custom but no checkpoint_path provided."
    echo "          The model will run with RANDOM WEIGHTS — predictions will be meaningless."
    echo "          Add 'checkpoint_path=<path>.pt' to use a trained checkpoint."
    echo ""
fi

# ── Build the shared Hydra command array ──────────────────────────────────────
# All launch variants reuse this array, only the accelerate prefix changes.
HYDRA_CMD=(
    inference.py
    "task_name=${TASK}"
    "model_name=${MODEL}"
    "model_type=${MODEL_TYPE}"
    "add_bos_token=${ADD_BOS}"
    "vocab_size=${VOCAB_SIZE}"
    "checkpoint_path=${CHECKPOINT_PATH}"
    "output_dir=${OUTPUT_DIR}"
    "res_dir=${RES_DIR}"
    "cache_dir=${CACHE_DIR}"
    "model_parallel=${MODEL_PARALLEL}"
    "hydra.run.dir=/tmp"
)
# Append any remaining extra Hydra overrides after the known keys
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    HYDRA_CMD+=("${EXTRA_ARGS[@]}")
fi

# ── GPU device mapping ────────────────────────────────────────────────────────
case "${N_GPU}" in
    8) DEVICES='0,1,2,3,4,5,6,7' ;;
    4) DEVICES='0,1,2,3' ;;
    2) DEVICES='0,1' ;;
    1) DEVICES='0' ;;
    *) echo "ERROR: N_GPU must be 1, 2, 4, or 8 (got: ${N_GPU})"; exit 1 ;;
esac

# ── Launch ────────────────────────────────────────────────────────────────────
if [ "${MODEL_PARALLEL}" == 'True' ] || [ "${MODEL_PARALLEL}" == 'true' ]; then
    # Model-parallel: single process that shards across all visible GPUs
    CUDA_VISIBLE_DEVICES="${DEVICES}" \
        accelerate launch --num_processes 1 "${HYDRA_CMD[@]}"
else
    if [ "${N_GPU}" == '1' ]; then
        CUDA_VISIBLE_DEVICES="${DEVICES}" \
            accelerate launch --num_processes 1 "${HYDRA_CMD[@]}"
    else
        CUDA_VISIBLE_DEVICES="${DEVICES}" \
            accelerate launch --num_processes "${N_GPU}" --multi_gpu "${HYDRA_CMD[@]}"
    fi
fi

# ── Post-inference: print saved outputs ───────────────────────────────────────
echo ""
echo "Post-inference: checking output and res dirs..."

if compgen -G "${OUTPUT_DIR}*" > /dev/null 2>&1; then
    for f in "${OUTPUT_DIR}"*; do
        echo "saved pred to:  $f"
        sed -n '1,200p' "$f" || true
    done
else
    echo "No files in ${OUTPUT_DIR}"
fi

if compgen -G "${RES_DIR}*" > /dev/null 2>&1; then
    for f in "${RES_DIR}"*; do
        echo "saved eval res to:  $f"
        sed -n '1,200p' "$f" || true
    done
else
    echo "No files in ${RES_DIR}"
fi

# ── Optional HF dataset API call ─────────────────────────────────────────────
if [ -n "${HF_DATASET_URL:-}" ]; then
    echo ""
    echo "HTTP Request: GET ${HF_DATASET_URL}"
    echo "Response:"
    curl -s -i "${HF_DATASET_URL}" || true
fi