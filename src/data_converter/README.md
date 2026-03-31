# Data Converter

Converts raw heterogeneous datasets into planning tasks with danger formalization.

## Pipeline

```
Raw Data → converter.py → danger_extracted_tasks.json → codegen.py → Task Folders
```

## Files

| File | Purpose |
|------|---------|
| `converter.py` | Loads raw data via adaptor, uses danger_extractor to extract danger formalization |
| `adaptor.py` | Dataset adapters for raw data (ALFRED, BDDL, NEISS, NormBank, VirtualHome) |
| `danger_extractor.py` | LLM-based danger formalization extraction |
| `codegen.py` | Generates `code.py` and `result.json` for each task from danger formalization |
| `llm_logger.py` | LLM API call logging |

## Output

**converter.py** produces:
- `danger_extracted_tasks.json` - All tasks with danger formalization

**codegen.py** produces task folders (e.g., `task_001/`), each containing:
- `code.py` - DomainPlanner class extending BasePlanner
- `result.json` - danger_formalization, generated_plans (safe/unsafe), codegen_metadata
