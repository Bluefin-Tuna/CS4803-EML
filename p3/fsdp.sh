#!/bin/bash
#SBATCH -J p4                   # Job name
#SBATCH -o ./output/out.%j.log       # Stdout
#SBATCH -e ./error/err.%j.log        # Stderr
#SBATCH -N 1                         # One node
#SBATCH --gres=gpu:h100:4         # Request up to 4 GPUs
#SBATCH --mem=128G                   # Memory


source /coc/flash5/tchopra32/eml/p3/.venv/bin/activate

echo "Host: $(hostname)"
echo "GPUs visible: $CUDA_VISIBLE_DEVICES"

SCRIPT="fsdp.py"

# -------- Config from environment (with sensible defaults) --------
WORLD_SIZE=${WORLD_SIZE:-4}       # num_gpus=4
MODE=${MODE:-ddp}                 # "ddp" or "fsdp_full" (adjust to your script)
GBS=${GBS:-64}                    # global batch size
WARMUP=${WARMUP:-20}              # warmup steps
MEASURE=${MEASURE:-100}           # measure steps
SEEDS=${SEEDS:-"1,2,3"}           # seeds string
OUT=${OUT:-"./fsdp_results/default.json"}

echo "Running FSDP/DDP config:"
echo "  WORLD_SIZE = ${WORLD_SIZE}"
echo "  MODE       = ${MODE}"
echo "  GBS        = ${GBS}"
echo "  WARMUP     = ${WARMUP}"
echo "  MEASURE    = ${MEASURE}"
echo "  SEEDS      = ${SEEDS}"
echo "  OUT        = ${OUT}"

mkdir -p "$(dirname "${OUT}")"

torchrun --standalone --nproc-per-node="${WORLD_SIZE}" "${SCRIPT}" \
  --world_size "${WORLD_SIZE}" \
  --mode "${MODE}" \
  --global_batch_size "${GBS}" \
  --steps_warmup "${WARMUP}" \
  --steps_measure "${MEASURE}" \
  --seeds "${SEEDS}" \
  --out "${OUT}"
