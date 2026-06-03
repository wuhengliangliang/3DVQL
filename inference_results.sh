#!/bin/bash

# =========================
# IQ-VLA / evaluation runner
# =========================

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export CUDA_MODULE_LOADING=LAZY

# >>> 4. 避免线程爆炸
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# >>> 5. 日志目录
LOG_DIR="./log"
mkdir -p ${LOG_DIR}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/eval_${TIMESTAMP}.log"

echo "=================================================="
echo "Starting VQL3D evaluation"
echo "Log file: ${LOG_FILE}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=================================================="

# >>> 6. 运行程序
python -u inference_results.py \
    --cfg config/eval.yaml \
    --eval \
    --data /mnt/data_2/pl/3DVQL_Test_V2 \
    2>&1 | tee ${LOG_FILE}