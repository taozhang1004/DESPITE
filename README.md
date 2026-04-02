# DESPITE: Deterministic Evaluation of Safe Planning In embodied Task Execution

Official implementation for the paper: **"Using large language models for embodied planning introduces systematic safety risks"**

**Authors:** Tao Zhang, Kaixian Qu, Zhibin Li, Jiajun Wu, Marco Hutter, Manling Li, Fan Shi

## Overview

DESPITE is a benchmark for evaluating LLM safety in embodied task planning. It tests four planning abilities:

1. **Comprehensive Planning** - Generate valid, safe plans
2. **Danger Identification** - Identify dangerous actions in a domain
3. **Danger Condition Inference** - Infer conditions that make actions dangerous
4. **Safe Alternative Discovery** - Generate safe alternative plans

## Installation

```bash
git clone https://github.com/Lennittus/DESPITE.git
cd DESPITE

# Create conda environment
conda create -n despite python=3.10
conda activate despite

# Install package with all dependencies
pip install -e ".[planning]"

# Install Java 17+ for ENHSP planning engine
# macOS:
brew install openjdk@17
# Ubuntu:
sudo apt install openjdk-17-jdk
```

**Requirements:**
- Python 3.8+
- Java 17+ (for ENHSP numeric planning engine)

### Apple Silicon (M1/M2/M3) Support

Both planning engines work on ARM-based Macs:

| Engine | ARM Support | Notes |
|--------|-------------|-------|
| `up-enhsp` | ✅ Yes | Requires Java 17+ (install via `brew install openjdk@17`) |
| `up-fast-downward` | ✅ Yes | Native ARM wheel available |
| `up-pyperplan` | ⚠️ Limited | Pure Python, but doesn't support numeric planning features |

**Plan Validation** (testing LLM-generated plans) works on all platforms.

**Plan Generation** (automatic planning) requires `up-enhsp` or `up-fast-downward`:
```bash
pip install up-enhsp  # Recommended for numeric planning
```

## Dataset

Download the DESPITE dataset from HuggingFace:

```bash
cd data
huggingface-cli download Lennittus/DESPITE --repo-type dataset --local-dir .
tar -xzf tasks.tar.gz
cd ..
```

Or visit: [https://huggingface.co/datasets/Lennittus/DESPITE](https://huggingface.co/datasets/Lennittus/DESPITE)

### Dataset Statistics

| Split | Subset | Tasks | Description |
|-------|--------|-------|-------------|
| `full` | `easy` | 11,235 | Standard difficulty |
| `full` | `hard` | 1,044 | Complex tasks (main evaluation) |
| `sampled` | `easy-100` | 100 | Quick evaluation subset |
| `sampled` | `hard-100` | 100 | Quick evaluation subset |

### Data Sources

Tasks derived from [ALFRED](https://askforalfred.com/), [BDDL](https://behavior.stanford.edu/), [VirtualHome](http://virtual-home.org/), [NormBank](https://github.com/SALT-NLP/normbank), and [NEISS](https://www.cpsc.gov/Research--Statistics/NEISS-Injury-Data).

## Quick Start

```bash
# 1. Activate environment
conda activate despite

# 2. Set up API keys
cp .env.example .env
# Edit .env with your API keys

# 3. Download dataset
pip install huggingface_hub
cd data && huggingface-cli download Lennittus/DESPITE --repo-type dataset --local-dir . && tar -xzf tasks.tar.gz && cd ..

# 4. Run benchmark
python src/experiments/benchmark.py
```

## Configuration

Edit the bottom of `src/experiments/benchmark.py` to configure:

```python
DRY_RUN = True              # Set False to run actual benchmark
FORCE_RERUN = False         # Set True to rerun completed tasks

abilities_to_test = [
    "comprehensive_planning",
    # "danger_identification",
    # "danger_condition_inference",
    # "safe_alternative_discovery",
]

models = [
    ("openai", "gpt-4o"),
    # ("anthropic", "claude-3-5-sonnet-20241022"),
]
```

## Project Structure

```
DESPITE/
├── src/
│   ├── planner/           # Core LLM planning module
│   │   ├── llm.py         # Multi-provider LLM interface
│   │   ├── solver.py      # BasePlanner + planning logic
│   │   └── prompts.py     # Prompt templates
│   ├── experiments/       # Benchmark scripts
│   │   ├── benchmark.py   # Main benchmark script
│   │   └── analysis/      # Visualization scripts
│   ├── data_generator/    # Data pipeline tools
│   └── utils/             # Utilities
├── data/                  # Dataset (download from HuggingFace)
└── pyproject.toml         # Package configuration
```

## Reproducing Paper Results

```bash
# Run benchmark (edit config at bottom of file)
python src/experiments/benchmark.py

# Generate analysis figures
python src/experiments/analysis/general_analysis.py
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

MIT License. See [LICENSE](LICENSE) for details.

See original dataset repositories for their respective terms.
