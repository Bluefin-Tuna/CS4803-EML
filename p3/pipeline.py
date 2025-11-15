# pipeline.py
# Distributed Pipeline Parallelism — Student Template
#
# Quick sanity check (one step):
#   torchrun --standalone --nproc-per-node=4 pipeline.py  --world_size 4 --global_batch_size 8 --chunks 4 --quick_demo
   
#
# For the assignment:
#   - Implement ALL TODOs below (measurement, seeds aggregation, custom partitioning, optional stage timing).
#   - Then run your experiments for Steps 1–4 and produce tables/plots.
#
# Notes:
#   - Use mean ± std over 3 seeds across Steps 1–4 (same protocol as the pipeline task).
#   - Warm up 20 steps, then measure >= 100 steps (configurable via CLI).

import argparse
import os
import time
import json

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, ScheduleGPipe, SplitPoint
from transformers import BertModel, BertConfig

from hf_utils import generate_inputs_for_model, get_number_of_params


# ---------------------------
# Split spec helpers
# ---------------------------

def make_split_spec_even(model: BertModel, world_size: int):
    """
    Build split points at 'encoder.layer.{k}' to roughly balance blocks per rank.

    """
    n_layers = model.config.num_hidden_layers
    per = (n_layers + world_size - 1) // world_size  # ceil
    if per == 0:
        raise ValueError(f"world_size ({world_size}) cannot exceed num_hidden_layers ({n_layers})")
    return {
        f"encoder.layer.{i * per}": SplitPoint.BEGINNING
        for i in range(1, world_size)
    }


def make_split_spec_custom(model: BertModel, pattern: str, world_size: int):
    """
    TODO (Step 4): parse a custom pattern like "6-2-2-2" and produce split points.
      - Validate: number of parts == world_size
      - Validate: sum(parts) == num_hidden_layers
      - Compute cumulative sums c1, c2, c3,... and place SplitPoint.BEGINNING
        at 'encoder.layer.{c1}', 'encoder.layer.{c2}', ...
    Return a dict[str, SplitPoint].
    """
    try:
        parts = [int(x) for x in pattern.split('-')]
    except ValueError:
        raise ValueError(f"Invalid partition pattern: {pattern}")
    assert len(parts) == world_size, f"Number of parts in pattern ({len(parts)}) must equal world_size ({world_size})"
    n_layers = model.config.num_hidden_layers
    assert sum(parts) == n_layers, f"Sum of parts ({sum(parts)}) must equal num_hidden_layers ({n_layers})"
    split_spec = {}
    cumulative = 0
    for part in parts[:-1]:
        cumulative += part
        split_spec[f"encoder.layer.{cumulative}"] = SplitPoint.BEGINNING
    return split_spec

# ---------------------------
# Optional: simple per-stage timing (hint)
# ---------------------------

def wrap_stage_forward_for_timing(module: torch.nn.Module, device: torch.device):
    """
    OPTIONAL (Step 4): wrap stage forward to collect avg forward time per step.
    Use CUDA events; store accumulator in module._acc_ms.
    """
    if not torch.cuda.is_available() or device.type != "cuda":
        return module
    forward = module.forward
    module._acc_ms = 0.0
    def timed(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(device):
            start.record()
            out = forward(*args, **kwargs)
            end.record()
        torch.cuda.synchronize(device)
        module._acc_ms += start.elapsed_time(end)
        return out
    module.forward = timed
    return module


# ---------------------------
# Warmup + measurement loop
# ---------------------------

def warmup_and_measure(schedule: ScheduleGPipe,
                       model: BertModel,
                       device: torch.device,
                       global_batch_size: int,
                       chunks: int,
                       steps_warmup: int,
                       steps_measure: int,
                       seed: int):
    """
    TODO (Step 1, Step 2, Step 3): implement full measurement.

    Requirements:
      - Assert global_batch_size % chunks == 0 (integer microbatch size).
      - Warm up for `steps_warmup` steps (do not time).
      - Measure for `steps_measure` steps:
          * Generate inputs each step on rank 0's device
          * Rank 0: schedule.step(**inputs); other ranks: schedule.step()
          * Use dist.barrier() + torch.cuda.synchronize() before/after timing
      - Throughput (samples/s) = (global_batch_size * steps_measure) / elapsed_seconds
      - Peak memory per rank via torch.cuda.max_memory_allocated(rank)
      - Return: (throughput: float, peaks_bytes: int)

    Hints:
      - Reset memory stats before timing on each rank: torch.cuda.reset_peak_memory_stats(rank)
      - Use time.time() wall clock for elapsed_seconds
    """
    assert global_batch_size % chunks == 0, "global_batch_size must be divisible by chunks"

    rank = dist.get_rank()

    def step():
        if rank == 0:
            inputs = generate_inputs_for_model(BertModel, model, "BertModel", global_batch_size, device)
            schedule.step(**inputs)
        else:
            _ = schedule.step()

    dist.barrier()
    for i in range(steps_warmup):
        step()
    dist.barrier()

    # Measurement
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        torch.cuda.reset_peak_memory_stats(idx)
        torch.cuda.synchronize(idx)

    start = time.time()

    for _ in range(steps_measure):
        step()

    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        torch.cuda.synchronize(idx)

    end = time.time()
    dist.barrier()

    elapsed = end - start
    throughput = (global_batch_size * steps_measure) / elapsed

    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        peak_bytes = torch.cuda.max_memory_allocated(idx)
    else:
        peak_bytes = 0

    return float(throughput), int(peak_bytes)


# ---------------------------
# One run for a given seed
# ---------------------------

def run_one_seed(args, pipe_ir, model):
    """
    TODO: If you want per-stage forward timing (Step 4), wrap your stage module first.

    Return a dict with at least:
      - 'seed': int
      - 'throughput_samples_per_s': float
      - 'mem_peak_per_rank_bytes': List[int]
      - (optional) 'stage_forward_time_ms_per_rank': List[float or None]
    """
    rank, world_size = dist.get_rank(), dist.get_world_size()
    stage = pipe_ir.build_stage(rank, device=args.device)
    schedule = ScheduleGPipe(stage, args.chunks)
    stage_module = pipe_ir.get_stage_module(rank)
    wrap_stage_forward_for_timing(stage_module, args.device)
    thr, peaks = warmup_and_measure(
        schedule=schedule,
        model=model,
        device=args.device,
        global_batch_size=args.global_batch_size,
        chunks=args.chunks,
        steps_warmup=args.steps_warmup,
        steps_measure=args.steps_measure,
        seed=args.seed,
    )
    stage_ms = getattr(stage_module, "_acc_ms", None)
    if stage_ms is not None and args.steps_measure > 0:
        stage_ms = float(stage_ms) / args.steps_measure
    peaks_all = [None for _ in range(world_size)]
    dist.all_gather_object(peaks_all, peaks)
    stage_ms_all = None
    if stage_ms is not None:
        stage_ms_all = [None for _ in range(world_size)]
        dist.all_gather_object(stage_ms_all, stage_ms)
    out = {
        "seed": args.seed,
        "throughput_samples_per_s": float(thr),
        "mem_peak_per_rank_bytes": peaks_all,
        "stage_forward_time_ms_per_rank": stage_ms_all,
    }
    return out


# ---------------------------
# Main
# ---------------------------

def run(args):
    rank = args.rank
    world_size = args.world_size

    config = BertConfig()
    config.return_dict = False

    # Create model
    model = BertModel(config)
    model.to(args.device)
    model.eval()

    if rank == 0:
        print(model.config)
        print(f"Total params = {get_number_of_params(model) // 10 ** 6}M")

    # Example microbatch
    assert args.global_batch_size % args.chunks == 0, "global_batch_size must be divisible by chunks"
    example_mb = generate_inputs_for_model(BertModel, model, "BertModel",
                                           args.global_batch_size // args.chunks, args.device)

    # Split points
    if args.partition == "even":
        split_spec = make_split_spec_even(model, world_size)
    else:
        split_spec = make_split_spec_custom(model, args.partition, world_size)

    if rank == 0:
        print("Split points:", list(split_spec.keys()))

    # Build pipeline IR
    pipe_ir = pipeline(
        model,
        mb_args=(),
        mb_kwargs=example_mb,
        split_spec=split_spec,
    )
    assert pipe_ir.num_stages == world_size, f"nstages={pipe_ir.num_stages} != world_size={world_size}"

    # Quick test (one step)
    if args.quick_demo:
        stage = pipe_ir.build_stage(args.rank, device=args.device)
        schedule = ScheduleGPipe(stage, args.chunks)
        full_inputs = generate_inputs_for_model(BertModel, model, "BertModel",
                                                args.global_batch_size, args.device)
        if rank == 0:
            schedule.step(**full_inputs)
        else:
            _ = schedule.step()
        dist.barrier()
        print(f"[Rank {rank}] quick_demo complete")
        return

    # ===== Multi-seed measurements (Step 1 / Step 2 / Step 3 / Step 4) =====
    # TODO: loop over args.seeds, set torch.manual_seed/torch.cuda.manual_seed_all,
    #       call run_one_seed each time, aggregate mean ± std on rank 0,
    #       and optionally save JSON to args.out for plotting.
    #
    # Hints:
    #   - Parse seeds from comma-separated string (e.g., "1,2,3").
    #   - On rank 0, compute mean/std for throughput; for memory, compute mean/std per rank.
    #   - If args.out is set, write a dict containing spec + per_seed + agg results.
    #
    # >>> YOUR CODE HERE <<<
    seeds = [int(s) for s in args.seeds.split(',')]
    results = []
    for seed in seeds:
        args.seed = seed
        torch.manual_seed(seed)
        if args.cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        result = run_one_seed(args, pipe_ir, model)
        if rank == 0:
            results.append(result)
    if rank == 0:
        throughputs = [res['throughput_samples_per_s'] for res in results]
        throughput_mean = float(np.mean(throughputs))
        throughput_std = float(np.std(throughputs))
        
        memories = [res['mem_peak_per_rank_bytes'] for res in results]
        memory_stats = []
        for ridx in range(world_size):
            mems_rank = [memories[sidx][ridx] for sidx in range(len(seeds))]
            mean_mem = float(np.mean(mems_rank))
            std_mem = float(np.std(mems_rank))
            memory_stats.append({'rank': ridx, 'mean': mean_mem, 'std': std_mem})

        aggregated = {
            'throughput_samples_per_s': {
                'mean': throughput_mean,
                'std': throughput_std
            },
            'mem_peak_per_rank_bytes': memory_stats
        }

        final_output = {
            'spec': {
                'world_size': world_size,
                'global_batch_size': args.global_batch_size,
                'chunks': args.chunks,
                'steps_warmup': args.steps_warmup,
                'steps_measure': args.steps_measure,
                'partition': args.partition,
                'seeds': seeds
            },
            'per_seed': results,
            'aggregated': aggregated
        }

        print("Aggregated results:", json.dumps(aggregated, indent=2))

        if args.out:
            with open(args.out, 'w') as f:
                json.dump(final_output, f, indent=2)
            print(f"Results saved to {args.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    p.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    p.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    p.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))

    p.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    p.add_argument('--global_batch_size', type=int, default=64)
    p.add_argument('--chunks', type=int, default=4)
    p.add_argument('--steps_warmup', type=int, default=20)
    p.add_argument('--steps_measure', type=int, default=100)
    p.add_argument('--partition', type=str, default="even",
                   help='either "even" or a custom pattern like "3-1-2-2" (Step 4)')
    p.add_argument('--seeds', type=str, default="1,2,3",
                   help='comma-separated list, e.g., "1,2,3"')
    p.add_argument('--out', type=str, default="",
                   help="optional JSON path to save results (for tables/plots)")
    p.add_argument('--quick_demo', action='store_true', help="run a single untimed step to sanity-check PiPPy")

    args = p.parse_args()

    lrank = int(os.getenv("LOCAL_RANK", 0))

    if args.cuda:
        torch.cuda.set_device(lrank)
        args.device = torch.device(f"cuda:{lrank}")
    else:
        args.device = torch.device("cpu")

    backend = "nccl" if args.cuda else "gloo"
    dist.init_process_group(backend=backend, rank=args.rank, world_size=args.world_size, device_id=lrank)

    args.world_size = dist.get_world_size()

    try:
        run(args)
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
