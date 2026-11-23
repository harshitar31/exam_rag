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
LEARNING_RATE=1e-4  # usual LoRA rate; 2e-5 barely moves the adapters in ~1 epoch
ITERS=1100          # ~2 epochs over data_v3 (1103 examples / batch 2)
STEPS_PER_EVAL=100
SAVE_EVERY=100      # keep checkpoints so an overfit final one can be skipped
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
PYTHON="${PYTHON:-python}"
$PYTHON -c "
from mlx_lm import load
print('Downloading/verifying model...')
load('$MODEL')
print('Model ready.')
"

# Run LoRA fine-tuning.
# --mask-prompt: only the JSON answer counts toward the loss. Without it most
# of the loss is the model predicting the PDF text it was given.
$PYTHON -m mlx_lm lora \
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
  --save-every $SAVE_EVERY \
  --mask-prompt \
  --grad-checkpoint

echo ""
echo "[DONE] Adapters saved to $ADAPTER_DIR/"
echo "Next: python evaluate.py --model $MODEL --adapter $ADAPTER_DIR"