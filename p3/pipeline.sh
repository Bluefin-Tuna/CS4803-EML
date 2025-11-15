#!/bin/bash
#SBATCH -J p3                     # Job name (you'll see this in squeue)
#SBATCH -o ./output/out.%j.log    # Stdout (%j = job ID)
#SBATCH -e ./error/err.%j.log     # Stderr
#SBATCH -N 1                      # One node
#SBATCH --gres=gpu:2080_ti:4      # Request up to 4 GPUs on the node
#SBATCH --mem=128G                # Memory


source /coc/flash5/tchopra32/eml/p3/.venv/bin/activate

echo "Host: $(hostname)"
echo "GPUs visible: $CUDA_VISIBLE_DEVICES"

SCRIPT="pipeline.py"

WORLD_SIZE=${WORLD_SIZE:-4}
PARTITION=${PARTITION:-even}
GBS=${GBS:-64}
CHUNKS=${CHUNKS:-4}
WARMUP=${WARMUP:-20}
MEASURE=${MEASURE:-100}
SEEDS=${SEEDS:-"1,2,3"}
OUT=${OUT:-"./results/default.json"}

echo "Running config:"
echo "  WORLD_SIZE = ${WORLD_SIZE}"
echo "  PARTITION  = ${PARTITION}"
echo "  GBS        = ${GBS}"
echo "  CHUNKS     = ${CHUNKS}"
echo "  WARMUP     = ${WARMUP}"
echo "  MEASURE    = ${MEASURE}"
echo "  SEEDS      = ${SEEDS}"
echo "  OUT        = ${OUT}"

mkdir -p "$(dirname "${OUT}")"

torchrun --standalone --nproc-per-node="${WORLD_SIZE}" "${SCRIPT}" \
  --world_size "${WORLD_SIZE}" \
  --partition "${PARTITION}" \
  --global_batch_size "${GBS}" \
  --chunks "${CHUNKS}" \
  --steps_warmup "${WARMUP}" \
  --steps_measure "${MEASURE}" \
  --seeds "${SEEDS}" \
  --out "${OUT}"
