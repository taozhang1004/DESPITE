# DESPITE
***D**eterministic **E**valuation of **S**afe **P**lanning **I**n embodied **T**ask **E**xecution*

This repository contains the official implementation for:

> **Using large language models for embodied planning introduces systematic safety risks**
>
> Tao Zhang, Kaixian Qu, Zhibin Li, Jiajun Wu, Marco Hutter, Manling Li, Fan Shi

DESPITE is a benchmark framework for evaluating task planning safety, with deterministic validation and scalable task generation.

## Getting Started

### Installation

```bash
git clone https://github.com/Lennittus/DESPITE.git
cd DESPITE
conda create -n despite python=3.10 && conda activate despite
pip install -e ".[planning]"
```

Requires Java 17+ for the planning engine.

### Dataset

```bash
mkdir -p data && cd data
huggingface-cli download Lennittus/DESPITE --repo-type dataset --local-dir .
tar -xzf tasks.tar.gz && cd ..
```

### Usage

```bash
# Generate pddls and reference plans
python {task_folder}/code.py

# Validate a plan
python {task_folder}/code.py -plan "(action1) (action2) ..."

# Run LLM benchmark
cp .env.example .env  # Add API keys
python src/experiments/benchmark.py
```

See `[src/data_generator/README.md](src/data_generator/README.md)` for task generation.

## Core Structure

```
├── data/{task_folder}/
│   └── code.py           # Entry point
└── src/
    ├── planner/          # LLM planner and deterministic solver
    ├── experiments/      # Benchmark scripts
    │   └── analysis/     # Result analysis
    └── data_generator/   # Task generation pipeline
```

## Citation

```bibtex
@article{zhang2025despite,
  title={Using large language models for embodied planning introduces systematic safety risks},
  author={Zhang, Tao and Qu, Kaixian and Li, Zhibin and Wu, Jiajun and Hutter, Marco and Li, Manling and Shi, Fan},
  year={2025}
}
```

## License

This project is licensed under the MIT License.