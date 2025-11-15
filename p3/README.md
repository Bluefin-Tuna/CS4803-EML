# Lab 3 - Tanush Chopra (tchopra32)
```

## Environment

1. Run `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Run `uv sync`
3. Run `source .venv/bin/activate`

## Task 2

### Assumptions

- You have activated the environment. If not refer to step 3 of the "Environment" section

### Steps

1. Run `python broadcast.py`
  a. Will save the outputs to ./results/part2_broadcast.json

## Task 3

### Assumptions

- You are on sky1.cc.gatech.edu
  - SLURM config settings in pipeline.sh rely on GPUs that are found here

### Steps

1. Run `bash p3.slurm`
  a. Error logs will be stored in ./error/ 
  b. Output logs will be stored in ./output/
  c. Results (JSON) will be stored in ./results/

## Task 4

### Assumptions

- You are on login-ice.pace.gatech.edu
  - SLURM config settings in pipeline.sh rely on GPUs that are found here

### Steps

- Run `bash p4.slurm`
  a. Error logs will be stored in ./error/ 
  b. Output logs will be stored in ./output/
  c. Results (JSON) will be stored in ./results/

