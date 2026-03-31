#!/usr/bin/env python3
"""
Add safety_intention to existing redundancy benchmark results.
Runs code.py -si for each plan and updates the validation_result.
Uses asyncio for high concurrency of subprocess calls.
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from tqdm import tqdm

MAX_CONCURRENT = 64  # Number of concurrent subprocess calls
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # sp-exp/


async def validate_safety_intention(sem: asyncio.Semaphore, task_dir: Path, plan_str: str) -> bool:
    """Run code.py -si to check safety intention for a plan."""
    code_py = task_dir / "code.py"
    if not code_py.exists():
        return False

    # Convert file path to module path for correct imports
    # e.g. data/sampled/redundancy/experiments/obj8/003022/code.py -> data.sampled.redundancy.experiments.obj8.003022.code
    try:
        rel = code_py.resolve().relative_to(REPO_ROOT)
        module_path = str(rel.with_suffix("")).replace("/", ".")
    except ValueError:
        return False

    async with sem:
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", module_path, "-si", plan_str,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(REPO_ROOT),
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode()
            return "Safety Intention: Yes" in output
        except Exception:
            return False


async def process_benchmark_file(sem: asyncio.Semaphore, benchmark_file: Path) -> dict:
    """Process a single benchmark file and add safety_intention to all models."""
    task_dir = benchmark_file.parent

    try:
        with open(benchmark_file) as f:
            data = json.load(f)
    except Exception as e:
        return {"file": str(benchmark_file), "status": "error", "message": str(e)}

    models = data.get("models", {})

    # Collect all SI evaluation tasks for this file
    si_tasks = []  # list of (model_key, validation_result, coroutine)

    for model_key, model_data in models.items():
        task_types = model_data.get("task_types", {})
        comprehensive = task_types.get("comprehensive_planning", {})

        if not comprehensive:
            continue

        validation_result = comprehensive.get("validation_result", {})

        # Get the plan from the result
        cp_result = comprehensive.get("comprehensive_planning_result", {})
        actions = cp_result.get("actions", [])

        if not actions:
            response = cp_result.get("response", "")
            if response:
                actions = re.findall(r'\([^)]+\)', response)

        if not actions:
            validation_result["safety_intention"] = False
        else:
            plan_str = " ".join(actions)
            si_tasks.append((validation_result, validate_safety_intention(sem, task_dir, plan_str)))

    if not si_tasks and not any(
        model_data.get("task_types", {}).get("comprehensive_planning", {})
        for model_data in models.values()
    ):
        return {"file": str(benchmark_file), "status": "skipped"}

    # Run all SI checks for this file concurrently
    if si_tasks:
        results = await asyncio.gather(*(t[1] for t in si_tasks))
        for (validation_result, _), si_val in zip(si_tasks, results):
            validation_result["safety_intention"] = si_val

    with open(benchmark_file, 'w') as f:
        json.dump(data, f, indent=2)
    return {"file": str(benchmark_file), "status": "updated"}


async def main():
    redundancy_base = Path("data/sampled/redundancy")

    benchmark_files = []
    for subdir in ["baseline", "experiments", "experiment_base"]:
        d = redundancy_base / subdir
        if d.exists():
            benchmark_files.extend(d.rglob("benchmark_results_*.json"))

    benchmark_files = list(benchmark_files)
    print(f"Found {len(benchmark_files)} benchmark files to process")

    sem = asyncio.Semaphore(MAX_CONCURRENT)

    updated_count = 0
    error_count = 0
    skipped_count = 0

    # Process all files concurrently (semaphore limits subprocess concurrency)
    tasks = [process_benchmark_file(sem, f) for f in benchmark_files]

    pbar = tqdm(total=len(tasks), desc="Adding SI")
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result["status"] == "updated":
            updated_count += 1
        elif result["status"] == "error":
            error_count += 1
        else:
            skipped_count += 1
        pbar.update(1)
    pbar.close()

    print(f"\nDone!")
    print(f"  Updated: {updated_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")


if __name__ == "__main__":
    asyncio.run(main())
