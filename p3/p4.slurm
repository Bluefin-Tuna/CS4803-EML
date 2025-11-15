#!/bin/bash

SCRIPT_SLURM="fsdp.sh"
RESULTS_DIR="./results"

GBS=64
WARMUP=20
MEASURE=100
SEEDS="1,2,3"

echo "Submitting FSDP/DDP experiments (num_gpus=4)..."

OUT_DDP="${RESULTS_DIR}/part4_ddp.json"
echo "  - Step 1 (DDP) -> ${OUT_DDP}"
sbatch --export=ALL,WORLD_SIZE=4,MODE=ddp,GBS=${GBS},WARMUP=${WARMUP},MEASURE=${MEASURE},SEEDS="${SEEDS}",OUT="${OUT_DDP}" \
  "${SCRIPT_SLURM}"

OUT_FSDP="${RESULTS_DIR}/part4_fsdp.json"
echo "  - Step 2 (FSDP full-shard) -> ${OUT_FSDP}"
sbatch --export=ALL,WORLD_SIZE=4,MODE=fsdp_full,GBS=${GBS},WARMUP=${WARMUP},MEASURE=${MEASURE},SEEDS="${SEEDS}",OUT="${OUT_FSDP}" \
  "${SCRIPT_SLURM}"

echo
echo "All FSDP/DDP jobs submitted. Check with:  squeue -u $USER"
