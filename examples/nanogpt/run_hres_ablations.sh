#!/usr/bin/env bash
set -euo pipefail

# Compare three H_res variants for mHC:
# 1) learnable doubly-stochastic (sinkhorn)
# 2) fixed identity
# 3) orthogonal manifold (Newton-Schulz projection)

python train.py config/train_fineweb10B_mhc.py "$@"
python train.py config/train_fineweb10B_mhc_identity.py "$@"
python train.py config/train_fineweb10B_mhc_orthogonal.py "$@"
