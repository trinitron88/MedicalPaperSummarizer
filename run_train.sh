#!/bin/bash
# Workaround script for macOS mutex issues

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export KMP_DUPLICATE_LIB_OK=TRUE
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Use exec to replace shell with python process
exec /usr/local/bin/python3 -u train_model.py
