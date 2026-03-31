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
git clone https://github.com/<username>/DESPITE.git
cd DESPITE
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- Java 17+ (for ENHSP numeric planning engine)

## Dataset

Download the DESPITE dataset from HuggingFace:

```bash
pip install huggingface_hub
huggingface-cli download <username>/DESPITE --repo-type dataset --local-dir data/
```

Or visit: [https://huggingface.co/datasets/<username>/DESPITE](https://huggingface.co/datasets/<username>/DESPITE)

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
# 1. Set up API keys
cp .env.example .env
# Edit .env with your API keys

# 2. Run benchmark
python src/experiments/benchmark-general.py
```

## Configuration

Edit the top of benchmark scripts to configure:

```python
MODELS = ["gpt-4o", "claude-3-5-sonnet", ...]  # Models to evaluate
FOLDERS = ["data/full/hard"]                    # Dataset splits
DRY_RUN = False                                 # Set True for testing
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
│   │   ├── benchmark-general.py
│   │   ├── benchmark-abilities.py
│   │   └── analysis/      # Visualization scripts
│   ├── data_converter/    # Data pipeline tools
│   └── utils/             # Utilities
├── data/                  # Dataset (download from HuggingFace)
└── requirements.txt
```

## Reproducing Paper Results

```bash
# Main benchmark (Table 1 in paper)
python src/experiments/benchmark-general.py

# Ability-specific evaluation (Table 2)
python src/experiments/benchmark-abilities.py

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
