#!/bin/bash

SCRIPT_SLURM="pipeline.sh"
RESULTS_DIR="./results"

GBS=64
WARMUP=20
MEASURE=100

echo "Submitting Step 1 jobs (world_size 2, 3, 4)..."
for g in 2 3 4; do
  OUT="${RESULTS_DIR}/part3_step1__g${g}.json"
  echo "  sbatch (Step 1, gpus=${g}) -> ${OUT}"
  sbatch --export=ALL,WORLD_SIZE=${g},PARTITION=even,GBS=${GBS},CHUNKS=4,WARMUP=${WARMUP},MEASURE=${MEASURE},OUT="${OUT}" \
    "${SCRIPT_SLURM}"
done

echo
echo "Submitting Step 3 jobs (g=2,4; chunks=4,8,16)..."
for g in 2 4; do
  for c in 4 8 16; do
    OUT="${RESULTS_DIR}/part3_step3__c${c}_g${g}.json"
    echo "  sbatch (Step 3, gpus=${g}, chunks=${c}) -> ${OUT}"
    sbatch --export=ALL,WORLD_SIZE=${g},PARTITION=even,GBS=${GBS},CHUNKS=${c},WARMUP=${WARMUP},MEASURE=${MEASURE},OUT="${OUT}" \
      "${SCRIPT_SLURM}"
  done
done

echo
echo "Submitting Step 4 jobs (balanced vs unbalanced)..."

# Balanced: 3-3-3-3
OUT_B="${RESULTS_DIR}/part3_step4__b.json"
echo "  sbatch (Step 4, balanced 3-3-3-3) -> ${OUT_B}"
sbatch --export=ALL,WORLD_SIZE=4,PARTITION=3-3-3-3,GBS=${GBS},CHUNKS=4,WARMUP=${WARMUP},MEASURE=${MEASURE},OUT="${OUT_B}" \
  "${SCRIPT_SLURM}"

# Unbalanced: 6-2-2-2
OUT_UB="${RESULTS_DIR}/part3_step4__ub.json"
echo "  sbatch (Step 4, unbalanced 6-2-2-2) -> ${OUT_UB}"
sbatch --export=ALL,WORLD_SIZE=4,PARTITION=6-2-2-2,GBS=${GBS},CHUNKS=4,WARMUP=${WARMUP},MEASURE=${MEASURE},OUT="${OUT_UB}" \
  "${SCRIPT_SLURM}"

echo
echo "All jobs submitted. Use:  squeue -u $USER"
