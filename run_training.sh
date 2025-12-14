#!/bin/bash
# Ultimate workaround for macOS PyTorch mutex bug

# Set all threading env vars
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export KMP_DUPLICATE_LIB_OK=TRUE
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Find libomp if it exists
if [ -f "/usr/local/lib/libomp.dylib" ]; then
    export DYLD_INSERT_LIBRARIES=/usr/local/lib/libomp.dylib
fi

# Run with unbuffered output
exec /usr/local/bin/python3 -u train_model.py
