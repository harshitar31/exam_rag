#!/bin/bash
# finetune.sh — Fine-tune Qwen2.5-3B-Instruct on Apple Silicon with MLX
#
# Prerequisites:
#   pip install mlx-lm pdfplumber requests tqdm
#
# Run:
#   bash finetune.sh

set -e

MODEL="mlx-community/Qwen2.5-3B-Instruct-4bit"   # 4-bit quant, ~2GB download
ADAPTER_DIR="adapters"
DATA_DIR="data_v3"       # built by fix_training_data.py (needs train.jsonl + valid.jsonl)
BATCH_SIZE=2
LEARNING_RATE=2e-5
ITERS=600           # ~1-2 hrs on M2/M3 Air, adjust up if you have more data
STEPS_PER_EVAL=100
# mlx_lm defaults to 2048 tokens; ~17% of question examples are longer and
# would be cut off mid-answer, teaching the model to stop early.
MAX_SEQ_LENGTH=4096

echo "=============================="
echo " EXAM RAG — MLX Fine-tune"
echo "=============================="
echo "Model   : $MODEL"
echo "Iters   : $ITERS"
echo "Data    : $DATA_DIR"
echo ""

# Download base model if not cached
python -c "
from mlx_lm import load
print('Downloading/verifying model...')
load('$MODEL')
print('Model ready.')
"

# Run LoRA fine-tuning
mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data "$DATA_DIR" \
  --adapter-path "$ADAPTER_DIR" \
  --batch-size $BATCH_SIZE \
  --iters $ITERS \
  --learning-rate $LEARNING_RATE \
  --steps-per-eval $STEPS_PER_EVAL \
  --val-batches 10 \
  --max-seq-length $MAX_SEQ_LENGTH \
  --grad-checkpoint

echo ""
echo "[DONE] Adapters saved to $ADAPTER_DIR/"
echo "Next: run 'bash fuse.sh' to merge adapters into the base model"