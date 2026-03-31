#!/usr/bin/env python3
"""
Redundancy Experiment Script

This script creates multiple versions of the task directories with different levels of
redundant objects and actions injected using the pddl_injector.py tool.

Usage:
    python src/tools/setup_redundancy_experiments.py --base-dir data/sampled/redundancy/experiment_base
"""

import argparse
import subprocess
import shutil
import random
import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict

# Add the tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from redundancy_injection.pddl_injector import PDDLInjector


class RedundancyExperiment:
    """Manages redundancy injection experiments with varying levels of redundancy"""

    def __init__(self, base_dir: str, output_base: str = "data/sampled/redundancy/experiments"):
        self.base_dir = Path(base_dir)
        self.output_base = Path(output_base)
        self.injector_script = Path("src/tools/redundancy_injection/pddl_injector.py")
        self.sources_dir = Path("src/tools/redundancy_injection")

    def run_experiments(self, object_levels: List[int] = None, action_levels: List[int] = None, seed: int = 42, shuffle_actions: bool = False):
        """Run experiments with different redundancy levels"""

        if object_levels is None:
            object_levels = [20, 40, 60, 80]
        if action_levels is None:
            action_levels = [20, 40, 60, 80]

        shuffle_text = "with shuffled actions" if shuffle_actions else "with ordered actions"
        print(f"Running redundancy experiments {shuffle_text}:")
        print(f"  Object levels: {object_levels}")
        print(f"  Action levels: {action_levels}")
        print(f"  Base directory: {self.base_dir}")
        print(f"  Output base: {self.output_base}")

        # Create output base directory
        self.output_base.mkdir(parents=True, exist_ok=True)

        # Find all task directories
        task_dirs = []
        if (self.base_dir / "domain.pddl").exists():
            # Single task directory
            task_dirs = [self.base_dir]
        else:
            # Multiple task directories
            task_dirs = [d for d in self.base_dir.iterdir() if d.is_dir() and (d / "domain.pddl").exists()]
            task_dirs.sort()

        if not task_dirs:
            print(f"❌ Error: No task directories found in {self.base_dir}")
            return

        print(f"\n📁 Found {len(task_dirs)} task directories")

        # Pre-select objects and actions for each task (random selection per task)
        # Set seed once for reproducibility, then let random state advance naturally
        random.seed(seed)
        max_objects = max(object_levels) if object_levels else 80
        max_actions = max(action_levels) if action_levels else 80
        print(f"📋 Pre-selecting up to {max_objects} objects and {max_actions} actions per task for cumulative experiments...")
        task_pre_selected_objects: Dict[Path, List[Tuple[str, str]]] = {}
        task_pre_selected_actions: Dict[Path, List[str]] = {}
        
        for task_dir in task_dirs:
            # Random selection per task (each task gets different random selection)
            pre_selected_objs = self._pre_select_objects(task_dir, max_objects)
            pre_selected_acts = self._pre_select_actions(max_actions)
            task_pre_selected_objects[task_dir] = pre_selected_objs
            task_pre_selected_actions[task_dir] = pre_selected_acts
            print(f"  ✅ {task_dir.name}: pre-selected {len(pre_selected_objs)} objects, {len(pre_selected_acts)} actions")

        # Track subset information for each experiment
        subset_tracker = {}

        # Run object experiments (with 0 actions)
        experiment_count = 0
        total_experiments = (len(object_levels) + len(action_levels)) * len(task_dirs)

        for num_objects in object_levels:
            exp_name = f"obj{num_objects}"
            output_dir = self.output_base / exp_name
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n🔬 Experiment: {exp_name} (Objects: {num_objects}, Actions: 0)")

            # Track subsets for this experiment configuration
            subset_tracker[exp_name] = {}

            for task_dir in task_dirs:
                experiment_count += 1
                task_output = output_dir / task_dir.name
                
                # Use pre-selected objects for this task (cumulative subset)
                pre_selected_objs_for_task = task_pre_selected_objects[task_dir]
                objects_used = pre_selected_objs_for_task[:num_objects]
                
                # Track which objects are used for this task in this experiment
                subset_tracker[exp_name][task_dir.name] = {
                    "objects": [obj[0] for obj in objects_used]
                }
                
                success = self._run_single_task_experiment(
                    task_dir,
                    task_output,
                    num_objects,
                    0,  # No actions for object experiments
                    seed,
                    shuffle_actions,
                    pre_selected_objs_for_task,
                    None  # No actions for object experiments
                )

                if success:
                    print(f"  ✅ {task_dir.name}")
                else:
                    print(f"  ❌ {task_dir.name}")

        # Run action experiments (with 0 objects)
        for num_actions in action_levels:
            exp_name = f"act{num_actions}"
            output_dir = self.output_base / exp_name
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n🔬 Experiment: {exp_name} (Objects: 0, Actions: {num_actions})")

            # Track subsets for this experiment configuration
            subset_tracker[exp_name] = {}

            for task_dir in task_dirs:
                experiment_count += 1
                task_output = output_dir / task_dir.name
                
                # Use pre-selected actions for this task (cumulative subset)
                pre_selected_acts_for_task = task_pre_selected_actions[task_dir]
                actions_used = pre_selected_acts_for_task[:num_actions]
                
                # Track which actions are used for this task in this experiment
                subset_tracker[exp_name][task_dir.name] = {
                    "actions": actions_used
                }
                
                success = self._run_single_task_experiment(
                    task_dir,
                    task_output,
                    0,  # No objects for action experiments
                    num_actions,
                    seed,
                    shuffle_actions,
                    None,  # No pre-selected objects for action experiments
                    pre_selected_acts_for_task
                )

                if success:
                    print(f"  ✅ {task_dir.name}")
                else:
                    print(f"  ❌ {task_dir.name}")

        # Save subset tracking information to a JSON file
        self._save_subset_tracking(subset_tracker, object_levels, action_levels, seed, shuffle_actions)

        print(f"\n🎉 Completed {experiment_count} task experiments across {len(object_levels) + len(action_levels)} configurations")
        self._print_summary()

    def _pre_select_actions(self, max_actions: int) -> List[str]:
        """Pre-select actions (same selection approach for all tasks)"""
        # Load actions file
        actions = {}
        actions_file = self.sources_dir / "actions.txt"
        with open(actions_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(':', 2)
                if len(parts) == 3:
                    name = parts[0].strip()
                    actions[name] = (parts[1].strip(), parts[2].strip())
        
        # Pre-select actions using current random state
        action_names = list(actions.keys())
        random.shuffle(action_names)
        
        return action_names[:max_actions]

    def _pre_select_objects(self, task_dir: Path, max_objects: int) -> List[Tuple[str, str]]:
        """Pre-select objects for a specific task (random selection per task)"""
        # Load objects file
        objects = {}
        objects_file = self.sources_dir / "objects.txt"
        with open(objects_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    type_name, objects_str = line.split(':', 1)
                    objects[type_name] = [obj.strip() for obj in objects_str.split(',')]
        
        # Pre-select objects using current random state (will be different for each task)
        selected = []
        available_types = list(objects.keys())
        
        # Get existing types from the task's domain file to prefer matching types
        existing_types = self._extract_types_from_domain(task_dir / "domain.pddl")
        
        preferred_types = [t for t in available_types if t in existing_types]
        if not preferred_types:
            preferred_types = available_types
        
        # Select objects avoiding duplicates
        selected_obj_names = set()
        for _ in range(max_objects):
            obj_type = random.choice(preferred_types)
            attempts = 0
            while attempts < 100:
                obj_name = random.choice(objects[obj_type])
                if obj_name not in selected_obj_names:
                    selected.append((obj_name, obj_type))
                    selected_obj_names.add(obj_name)
                    break
                attempts += 1
            if attempts >= 100:
                obj_name = random.choice(objects[obj_type])
                selected.append((obj_name, obj_type))
        
        return selected
    
    def _extract_types_from_domain(self, domain_file: Path) -> List[str]:
        """Extract type names from a domain.pddl file"""
        import re
        types = []
        try:
            with open(domain_file) as f:
                content = f.read()
            match = re.search(r':types\s+([^)]+)', content)
            if match:
                types_text = match.group(1).strip()
                types = [t.strip() for t in types_text.split() if t.strip()]
        except:
            pass
        return types

    def _run_single_task_experiment(self, input_dir: Path, output_dir: Path,
                              num_objects: int, num_actions: int, seed: int, 
                              shuffle_actions: bool = False, pre_selected_objects: List[Tuple[str, str]] = None,
                              pre_selected_actions: List[str] = None) -> bool:
        """Run a single redundancy injection experiment for one task"""
        try:
            # Use the injector class directly to pass pre-selected objects and actions
            injector = PDDLInjector(
                sources_dir=self.sources_dir,
                seed=seed,
                shuffle_actions=shuffle_actions,
                pre_selected_objects=pre_selected_objects,
                pre_selected_actions=pre_selected_actions
            )
            
            success = injector.inject_directory(input_dir, output_dir, num_objects, num_actions)
            return success

        except Exception as e:
            print(f"    Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _save_subset_tracking(self, subset_tracker: Dict, object_levels: List[int], 
                             action_levels: List[int], seed: int, shuffle_actions: bool):
        """Save subset tracking information to a JSON file in the output directory"""
        # Organize by task instead of by experiment
        # First, collect all task names
        all_tasks = set()
        for tasks in subset_tracker.values():
            all_tasks.update(tasks.keys())
        all_tasks = sorted(all_tasks)
        
        # Create structure: tasks -> experiment -> data
        tracking_data = {
            "config": {
                "seed": seed,
                "shuffle_actions": shuffle_actions,
                "object_levels": object_levels,
                "action_levels": action_levels
            },
            "tasks": {}
        }
        
        # Build task-centric structure
        for task_name in all_tasks:
            tracking_data["tasks"][task_name] = {}
            
            # Add object experiments for this task
            for num_obj in sorted(object_levels):
                exp_name = f"obj{num_obj}"
                if exp_name in subset_tracker and task_name in subset_tracker[exp_name]:
                    tracking_data["tasks"][task_name][exp_name] = subset_tracker[exp_name][task_name]["objects"]
            
            # Add action experiments for this task
            for num_act in sorted(action_levels):
                exp_name = f"act{num_act}"
                if exp_name in subset_tracker and task_name in subset_tracker[exp_name]:
                    tracking_data["tasks"][task_name][exp_name] = subset_tracker[exp_name][task_name]["actions"]
        
        tracking_file = self.output_base / "subset_tracking.json"
        
        # Custom JSON formatting: compact arrays on one line
        def format_json_compact(obj, indent=0):
            """Format JSON with arrays on single lines"""
            indent_str = '  ' * indent
            next_indent = '  ' * (indent + 1)
            
            if isinstance(obj, dict):
                if not obj:
                    return '{}'
                items = []
                for key, value in obj.items():
                    formatted_value = format_json_compact(value, indent + 1)
                    items.append(f'{next_indent}"{key}": {formatted_value}')
                return '{\n' + ',\n'.join(items) + '\n' + indent_str + '}'
            elif isinstance(obj, list):
                if not obj:
                    return '[]'
                # Format list on single line
                items = ', '.join(json.dumps(item) for item in obj)
                return f'[{items}]'
            else:
                return json.dumps(obj)
        
        with open(tracking_file, 'w') as f:
            f.write(format_json_compact(tracking_data))
            f.write('\n')
        
        print(f"\n📝 Subset tracking saved to: {tracking_file}")
        
        # Also create a human-readable text file
        tracking_txt_file = self.output_base / "subset_tracking.txt"
        with open(tracking_txt_file, 'w') as f:
            f.write("Redundancy Experiment Subset Tracking\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Config: seed={seed}, shuffle_actions={shuffle_actions}\n")
            f.write(f"Object levels: {object_levels}\n")
            f.write(f"Action levels: {action_levels}\n")
            f.write(f"Cumulative: Yes (each larger object level is a superset)\n\n")
            
            f.write("By Task:\n")
            f.write("-" * 60 + "\n\n")
            
            # Process by task (matching JSON structure)
            all_tasks = set()
            for tasks in subset_tracker.values():
                all_tasks.update(tasks.keys())
            
            for task_name in sorted(all_tasks):
                f.write(f"{task_name}:\n")
                
                # Show object experiments for this task
                f.write(f"  Object experiments:\n")
                for num_obj in sorted(object_levels):
                    exp_name = f"obj{num_obj}"
                    if exp_name in subset_tracker and task_name in subset_tracker[exp_name]:
                        obj_names = subset_tracker[exp_name][task_name]["objects"]
                        f.write(f"    {exp_name}: {len(obj_names)} objects")
                        if len(obj_names) <= 15:
                            f.write(f" - {', '.join(obj_names)}\n")
                        else:
                            f.write(f" - {', '.join(obj_names[:15])} ... ({len(obj_names)-15} more)\n")
                
                # Show action experiments for this task
                f.write(f"  Action experiments:\n")
                for num_act in sorted(action_levels):
                    exp_name = f"act{num_act}"
                    if exp_name in subset_tracker and task_name in subset_tracker[exp_name]:
                        action_names = subset_tracker[exp_name][task_name]["actions"]
                        f.write(f"    {exp_name}: {len(action_names)} actions")
                        if len(action_names) <= 15:
                            f.write(f" - {', '.join(action_names)}\n")
                        else:
                            f.write(f" - {', '.join(action_names[:15])} ... ({len(action_names)-15} more)\n")
                
                f.write("\n")
        
        print(f"📄 Human-readable tracking saved to: {tracking_txt_file}")

    def _print_summary(self):
        """Print a summary of created experiments"""
        print(f"\nExperiment Summary:")
        print(f"Output directory: {self.output_base}")

        if self.output_base.exists():
            experiments = [d for d in self.output_base.iterdir() if d.is_dir() and (d.name.startswith("obj") or d.name.startswith("act"))]
            print(f"Total experiments created: {len(experiments)}")

            for exp_dir in sorted(experiments):
                task_count = len([d for d in exp_dir.iterdir() if d.is_dir()])
                print(f"  {exp_dir.name}: {task_count} tasks")
        else:
            print("No experiments found.")

    def clean_experiments(self):
        """Clean up all experiment directories"""
        if self.output_base.exists():
            print(f"Cleaning up experiments in {self.output_base}")
            shutil.rmtree(self.output_base)
            print("✅ Cleanup complete")
        else:
            print("No experiments to clean up")

    def list_experiments(self):
        """List all available experiments"""
        print(f"Experiments in {self.output_base}:")

        if not self.output_base.exists():
            print("  No experiments found")
            return

        experiments = [d for d in self.output_base.iterdir() if d.is_dir() and (d.name.startswith("obj") or d.name.startswith("act"))]
        if not experiments:
            print("  No experiments found")
            return

        for exp_dir in sorted(experiments):
            task_count = len([d for d in exp_dir.iterdir() if d.is_dir()])

            # Parse experiment name to extract parameters
            if exp_dir.name.startswith('obj'):
                obj_count = exp_dir.name[3:]  # Remove 'obj' prefix
                print(f"  {exp_dir.name}: {obj_count} objects, 0 actions, {task_count} tasks")
            elif exp_dir.name.startswith('act'):
                act_count = exp_dir.name[3:]  # Remove 'act' prefix
                print(f"  {exp_dir.name}: 0 objects, {act_count} actions, {task_count} tasks")
            else:
                print(f"  {exp_dir.name}: {task_count} tasks")


def main():
    # ============================================================================
    # Configuration - Adjust these values directly here
    # ============================================================================
    BASE_DIR = "data/sampled/redundancy/experiment_base"  # Base directory containing clean task directories
    OUTPUT_BASE = "data/sampled/redundancy/experiments"   # Output base directory for experiments
    OBJECT_LEVELS = [2, 4, 8, 16, 32, 64]         # List of object redundancy levels
    ACTION_LEVELS = [2, 4, 8, 16, 32, 64]         # List of action redundancy levels
    SEED = 42                                      # Random seed for reproducibility
    SHUFFLE_ACTIONS = True                        # Randomly shuffle action order instead of appending at end
    # ============================================================================

    # Parse command line arguments (only for utility flags)
    parser = argparse.ArgumentParser(description="Run redundancy injection experiments")
    parser.add_argument("--clean", action="store_true",
                       help="Clean up existing experiments")
    parser.add_argument("--list", action="store_true",
                       help="List existing experiments")

    args = parser.parse_args()

    experiment = RedundancyExperiment(BASE_DIR, OUTPUT_BASE)

    if args.clean:
        experiment.clean_experiments()
    elif args.list:
        experiment.list_experiments()
    else:
        experiment.run_experiments(OBJECT_LEVELS, ACTION_LEVELS, SEED, SHUFFLE_ACTIONS)


if __name__ == "__main__":
    main()