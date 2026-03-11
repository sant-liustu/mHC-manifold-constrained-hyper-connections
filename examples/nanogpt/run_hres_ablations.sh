#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Default to 8 GPUs on a single node; override with env if needed.
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# Fast ablation defaults (intended for short comparative runs).
COMMON_ARGS=(
  max_iters=3000
  lr_decay_iters=3000
  gradient_accumulation_steps=8
  batch_size=8
  block_size=512
  n_layer=6
  n_embd=288
  n_head=6
  eval_interval=1000
  eval_iters=50
  compile_model=True
)

# Compare three H_res variants for mHC:
# 1) learnable doubly-stochastic (sinkhorn)
# 2) fixed identity
# 3) orthogonal manifold (Newton-Schulz projection)

run_cfg() {
  local cfg="$1"
  shift
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train.py "${cfg}" \
    "${COMMON_ARGS[@]}" "$@"
}

run_cfg config/train_fineweb10B_mhc.py "$@"
run_cfg config/train_fineweb10B_mhc_identity.py "$@"
run_cfg config/train_fineweb10B_mhc_orthogonal.py "$@"
