# DESPITE Dataset

This folder should contain the DESPITE dataset downloaded from HuggingFace.

## Download

```bash
# Option 1: Using huggingface-cli
pip install huggingface_hub
huggingface-cli download taozhang1004/DESPITE --repo-type dataset --local-dir .

# Option 2: Using Python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="taozhang1004/DESPITE", repo_type="dataset", local_dir=".")
```

Or visit: [https://huggingface.co/datasets/taozhang1004/DESPITE](https://huggingface.co/datasets/taozhang1004/DESPITE)

## Expected Structure

After downloading, this folder should contain:

```
data/
├── tasks/
│   ├── full/
│   │   ├── easy/           # 11,235 tasks
│   │   └── hard/           # 1,044 tasks (main evaluation)
│   └── sampled/
│       ├── easy-100/       # 100 tasks (quick testing)
│       ├── hard-100/       # 100 tasks (quick testing)
│       └── redundancy/     # Scaling analysis tasks
├── benchmark_results/      # Pre-computed LLM evaluation results (23 models)
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

## Quick Start

```bash
# From repo root, after downloading dataset
python src/experiments/benchmark-general.py
```
