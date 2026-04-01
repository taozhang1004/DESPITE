# Data Generator

Generates safe planning tasks with danger formalization from raw heterogeneous datasets.

## Pipeline

```
Raw Data → converter.py → danger_extracted_tasks.json → codegen.py → Task Folders
```

## Files

| File | Purpose |
|------|---------|
| `converter.py` | Main entry point; loads data via adaptor and extracts danger formalization |
| `adaptor.py` | Dataset adapters (ALFRED, BDDL, NEISS, NormBank, VirtualHome) |
| `danger_extractor.py` | LLM-based danger formalization extraction |
| `codegen.py` | Generates `code.py` and `result.json` for each task |
| `llm_logger.py` | LLM API call logging |

## Folder Structure

Expected input data structure:

```
DESPITE/data/source/
├── NormBank.csv                     # normbank
├── neiss2024.csv                    # neiss
├── bddl/{task_name}/problem0.bddl   # bddl
├── virtualhome/                     # virtualhome
│   ├── initstate/{scene}.json
│   └── withoutconds/{scene}.txt
└── alfred/**/*.json                 # alfred
```

## Output

- **converter.py**: `danger_extracted_tasks.json` with danger formalization
- **codegen.py**: Task folders (e.g., `task_001/`) containing `code.py` and `result.json`
