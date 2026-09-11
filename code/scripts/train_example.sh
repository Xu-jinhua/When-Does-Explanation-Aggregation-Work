#!/usr/bin/env bash
# train_example.sh — end-to-end Phase 0 example: build the BreastMNIST
# manifest, register the reference (full train split) partition, and
# fine-tune a ResNet-18 reference model that a simple config can then bind
# with `init_mode: checkpoint` + `checkpoint_path`.
#
# Usage (from anywhere):
#   bash scripts/train_example.sh
#
# Environment overrides:
#   PYTHON               python interpreter      (default: python)
#   HF_HOME              Hugging Face cache root (default: ~/.cache/huggingface)
#   CUDA_VISIBLE_DEVICES GPU selection           (default: 0)
#   EPOCHS               training epochs         (default: 30)
#
# The dataset download is cached under ./cache/medmnist (relative to code/).
# After training, point your config at the exported inference weights:
#   init_mode: checkpoint
#   checkpoint_path: configs/simple/assets/breastmnist-resnet18/checkpoints/inference.pt
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE_ROOT"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="src"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EPOCHS="${EPOCHS:-30}"

ASSETS="configs/simple/assets"
MANIFEST="$ASSETS/breastmnist.manifest.json"
PARTITION="$ASSETS/breastmnist-reference-partition.json"
CACHE_DIR="./cache/medmnist"
OUTPUT="$ASSETS/breastmnist-resnet18/checkpoints"

# 1. Dataset manifest (downloads BreastMNIST once into $CACHE_DIR).
"$PYTHON" -m xai_ensemble.cli phase0 build-manifest \
  --dataset breastmnist \
  --output "$MANIFEST" \
  --cache-dir "$CACHE_DIR"

# 2. Reference partition: the full train split behind source-id reference-full.
"$PYTHON" -m xai_ensemble.cli phase0 build-partitions \
  --manifest "$MANIFEST" \
  --output "$PARTITION" \
  --kind reference \
  --split train

# 3. Fine-tune the reference model from ImageNet-1K weights.
"$PYTHON" -m xai_ensemble.cli phase0 train \
  --dataset breastmnist \
  --manifest "$MANIFEST" \
  --train-partition "$PARTITION" \
  --source-id reference-full \
  --model resnet18 \
  --recipe timm_finetune \
  --initialization imagenet1k \
  --epochs "$EPOCHS" \
  --batch-size 128 \
  --validation-batch-size 128 \
  --workers 4 \
  --precision auto \
  --seed 20260714 \
  --device cuda \
  --cache-dir "$CACHE_DIR" \
  --output "$OUTPUT" \
  --resume auto \
  --run-id quickstart \
  --task-id train-breastmnist-resnet18-reference \
  --project-root .

echo
echo "Checkpoints written under $OUTPUT (latest.pt / best.pt / inference.pt)."
echo "To use the trained model in a simple config, set:"
echo "  init_mode: checkpoint"
echo "  checkpoint_path: $OUTPUT/inference.pt"
