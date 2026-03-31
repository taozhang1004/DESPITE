# DESPITE Dataset

This folder should contain the DESPITE dataset.

## Download from HuggingFace

```bash
pip install huggingface_hub
huggingface-cli download <username>/DESPITE --repo-type dataset --local-dir .
```

Or visit: [https://huggingface.co/datasets/<username>/DESPITE](https://huggingface.co/datasets/<username>/DESPITE)

## Expected Structure

After downloading, this folder should contain:

```
data/
├── tasks/
│   ├── full/
│   │   ├── easy/           # 11,235 tasks
│   │   └── hard/           # 1,044 tasks
│   └── sampled/
│       ├── easy-100/       # 100 tasks
│       ├── hard-100/       # 100 tasks
│       └── redundancy/     # Scaling analysis tasks
├── benchmark_results/      # Pre-computed LLM evaluation results
└── generation_info/        # Task provenance metadata
```

## Task Structure

Each task folder contains:

```
{task_id}/
├── code.py           # Executable domain definition (Python)
├── domain.pddl       # PDDL domain file
├── problem.pddl      # PDDL problem file
└── metadata.json     # Danger formalization + reference plans
```

## Usage

Once downloaded, you can run benchmarks:

```bash
cd ..  # Go to repo root
python src/experiments/benchmark-general.py
```
