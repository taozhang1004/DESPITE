#!/usr/bin/env python3
"""
Fixed PDDL redundancy injector that properly respects PDDL structure.
"""

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class PDDLInjector:
    def __init__(self, sources_dir: Path, seed: int = 42, shuffle_actions: bool = False, 
                 pre_selected_objects: List[Tuple[str, str]] = None, pre_selected_actions: List[str] = None):
        self.sources_dir = Path(sources_dir)
        self.seed = seed
        self.shuffle_actions = shuffle_actions
        self.pre_selected_objects = pre_selected_objects
        self.pre_selected_actions = pre_selected_actions
        random.seed(seed)

        # Load redundancy sources
        self.objects = self._load_objects()
        self.actions = self._load_actions()

    def _load_objects(self) -> Dict[str, List[str]]:
        objects = {}
        with open(self.sources_dir / "objects.txt") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    type_name, objects_str = line.split(':', 1)
                    objects[type_name] = [obj.strip() for obj in objects_str.split(',')]
        return objects

    def _load_actions(self) -> Dict[str, Tuple[str, str]]:
        actions = {}
        with open(self.sources_dir / "actions.txt") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                parts = line.split(':', 2)
                if len(parts) == 3:
                    name, precond, effect = parts
                    actions[name] = (precond.strip(), effect.strip())
        return actions

    def _extract_action_dependencies(self, actions: List[str]) -> Dict[str, List[str]]:
        """Extract objects and predicates needed by actions"""
        dependencies = {
            'objects': [],
            'predicates': []
        }

        for action_name in actions:
            if action_name in self.actions:
                precond, effect = self.actions[action_name]

                # Extract predicates and objects from preconditions and effects
                for text in [precond, effect]:
                    # Find all predicate patterns in PDDL expressions
                    # This matches patterns like (predicate_name ...)
                    predicate_matches = re.findall(r'\(([a-zA-Z_][a-zA-Z0-9_]*)', text)

                    # Filter out PDDL logical operators
                    pddl_operators = {'and', 'not', 'or', 'when', 'forall', 'exists', 'imply'}
                    predicates = [p for p in predicate_matches if p not in pddl_operators]
                    dependencies['predicates'].extend(predicates)

                    # Find object references that aren't variables (don't start with ?)
                    objects = re.findall(r'\b(?!\?)[a-zA-Z_]+(?=\s|\)|$)', text)
                    # Filter out common PDDL keywords and predicates
                    keywords = {'and', 'not', 'or', 'robot_at', 'object_at', 'robot_has'}
                    filtered_objects = [obj for obj in objects if obj not in keywords and obj not in predicates and len(obj) > 1]
                    dependencies['objects'].extend(filtered_objects)

        # Remove duplicates
        dependencies['objects'] = list(set(dependencies['objects']))
        dependencies['predicates'] = list(set(dependencies['predicates']))

        return dependencies

    def inject_directory(self, input_dir: Path, output_dir: Path, num_objects: int = 5, num_actions: int = 3) -> bool:
        try:
            input_dir = Path(input_dir)
            output_dir = Path(output_dir)

            domain_file = input_dir / "domain.pddl"
            problem_file = input_dir / "problem.pddl"

            if not domain_file.exists() or not problem_file.exists():
                return False

            output_dir.mkdir(parents=True, exist_ok=True)

            # Process files
            self._process_domain(domain_file, output_dir / "domain.pddl", num_objects, num_actions)
            self._process_problem(problem_file, output_dir / "problem.pddl", num_objects)

            # Copy additional files if they exist
            import shutil
            files_to_copy = ["code.py", "result.json"]
            for filename in files_to_copy:
                source_file = input_dir / filename
                if source_file.exists():
                    shutil.copy2(source_file, output_dir / filename)

            return True

        except Exception as e:
            print(f"Error: {e}")
            return False

    def _process_domain(self, input_file: Path, output_file: Path, num_objects: int, num_actions: int):
        with open(input_file) as f:
            content = f.read()

        # Parse existing structure
        existing_types = self._extract_types(content)

        # Select redundant elements
        self.selected_objects = self._select_objects(num_objects, existing_types)
        selected_actions = self._select_actions(num_actions)

        # Inject into content
        modified_content = self._inject_domain(content, self.selected_objects, selected_actions)

        with open(output_file, 'w') as f:
            f.write(modified_content)

    def _process_problem(self, input_file: Path, output_file: Path, num_objects: int):
        with open(input_file) as f:
            content = f.read()

        # Use the same objects selected for domain
        selected_objects = getattr(self, 'selected_objects', [])

        # Inject objects
        modified_content = self._inject_problem(content, selected_objects)

        with open(output_file, 'w') as f:
            f.write(modified_content)

    def _extract_types(self, content: str) -> List[str]:
        match = re.search(r':types\s+([^)]+)', content)
        if match:
            types_text = match.group(1).strip()
            return [t.strip() for t in types_text.split() if t.strip()]
        return []

    def _select_objects(self, num_objects: int, existing_types: List[str]) -> List[Tuple[str, str]]:
        # If pre-selected objects are provided, use the first N objects from that list
        # This ensures cumulative selection within a task
        if self.pre_selected_objects is not None:
            if len(self.pre_selected_objects) >= num_objects:
                return self.pre_selected_objects[:num_objects]
            else:
                # If we need more objects than pre-selected, fall back to random selection
                # but start with the pre-selected ones
                selected = list(self.pre_selected_objects)
                remaining = num_objects - len(selected)
        else:
            selected = []
            remaining = num_objects

        # Random selection for remaining objects or if no pre-selected objects
        available_types = list(self.objects.keys())

        # Prefer existing types
        preferred_types = [t for t in available_types if t in existing_types]
        if not preferred_types:
            preferred_types = available_types

        # Get already selected objects to avoid duplicates
        selected_obj_names = {obj_name for obj_name, _ in selected}

        for _ in range(remaining):
            obj_type = random.choice(preferred_types)
            # Avoid duplicates by checking if object already selected
            attempts = 0
            while attempts < 100:  # Limit attempts to avoid infinite loop
                obj_name = random.choice(self.objects[obj_type])
                if obj_name not in selected_obj_names:
                    selected.append((obj_name, obj_type))
                    selected_obj_names.add(obj_name)
                    break
                attempts += 1
            if attempts >= 100:
                # If we can't find a unique object, just add it anyway (unlikely but safe)
                obj_name = random.choice(self.objects[obj_type])
                selected.append((obj_name, obj_type))

        return selected


    def _select_actions(self, num_actions: int) -> List[str]:
        # If pre-selected actions are provided, use the first N actions from that list
        # This ensures cumulative selection within a task
        if self.pre_selected_actions is not None:
            if len(self.pre_selected_actions) >= num_actions:
                return self.pre_selected_actions[:num_actions]
            else:
                # If we need more actions than pre-selected, fall back to random selection
                selected = list(self.pre_selected_actions)
                remaining = num_actions - len(selected)
        else:
            selected = []
            remaining = num_actions

        # Random selection for remaining actions or if no pre-selected actions
        action_names = list(self.actions.keys())
        selected_action_names = set(selected)
        
        # Get actions not yet selected
        available_actions = [a for a in action_names if a not in selected_action_names]
        random.shuffle(available_actions)
        selected.extend(available_actions[:remaining])
        
        return selected[:num_actions]

    def _inject_domain(self, content: str, objects: List[Tuple[str, str]], actions: List[str]) -> str:
        lines = content.split('\n')
        result = []

        for i, line in enumerate(lines):
            result.append(line)

            # Inject types after :types line
            if ':types' in line:
                needed_types = set(obj_type for _, obj_type in objects)
                # Extract existing types from the line
                existing_part = line.split('(:types')[1].strip()
                if existing_part.endswith(')'):
                    existing_part = existing_part[:-1].strip()
                existing_types = set(existing_part.split()) if existing_part else set()

                new_types = needed_types - existing_types
                if new_types:
                    # Rebuild the types line properly
                    all_types = list(existing_types) + list(sorted(new_types))
                    result[-1] = f" (:types {' '.join(all_types)})"

            # Inject constants after :constants line
            elif ':constants' in line:
                # Group objects by type
                by_type = {}
                for obj_name, obj_type in objects:
                    if obj_type not in by_type:
                        by_type[obj_type] = []
                    by_type[obj_type].append(obj_name)

                # Add lines for each type
                for obj_type, obj_names in by_type.items():
                    obj_str = ' '.join(obj_names)
                    result.append(f"   {obj_str} - {obj_type}")


        # Add complex domain-aware actions
        if actions:
            domain_context = self._extract_domain_context(result)
            action_dependencies = self._extract_action_dependencies(actions)

            # Add any missing objects/predicates needed by actions to the domain
            result = self._add_action_dependencies(result, action_dependencies, domain_context)

            complex_actions = self._generate_domain_aware_actions(actions, domain_context)

            if self.shuffle_actions:
                # Random insertion: find existing actions and mix with new ones
                result = self._insert_actions_randomly(result, complex_actions)
            else:
                # Original behavior: append at the end before final closing parenthesis
                for i in range(len(result) - 1, -1, -1):
                    if result[i].strip() == ')':
                        # Insert complex actions before this line
                        for action_lines in reversed(complex_actions):
                            for line in reversed(action_lines):
                                result.insert(i, line)
                        break

        return '\n'.join(result)

    def _insert_actions_randomly(self, lines: List[str], new_actions: List[List[str]]) -> List[str]:
        """Randomly insert new actions among existing actions to shuffle the order"""
        # Find all existing action blocks
        action_blocks = []
        current_action = []
        in_action = False
        action_start_idx = -1

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Detect start of action
            if stripped.startswith('(:action'):
                if in_action and current_action:
                    # Save previous action
                    action_blocks.append((action_start_idx, current_action.copy()))
                in_action = True
                action_start_idx = i
                current_action = [line]
            elif in_action:
                current_action.append(line)
                # Detect end of action (closing parenthesis at same indentation level as (:action)
                if stripped == ')' and len(line) - len(line.lstrip()) <= 1:  # Top-level closing
                    action_blocks.append((action_start_idx, current_action.copy()))
                    current_action = []
                    in_action = False

        if in_action and current_action:
            # Handle case where last action doesn't have proper closing
            action_blocks.append((action_start_idx, current_action.copy()))

        if not action_blocks:
            # No existing actions found, fall back to original behavior
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == ')':
                    for action_lines in reversed(new_actions):
                        for line in reversed(action_lines):
                            lines.insert(i, line)
                    break
            return lines

        # Create a list of all actions (existing + new) and shuffle them
        all_actions = []

        # Add existing actions
        for _, action_lines in action_blocks:
            all_actions.append(action_lines)

        # Add new actions
        for action_lines in new_actions:
            all_actions.append(action_lines)

        # Shuffle all actions together
        random.shuffle(all_actions)

        # Build result by replacing action blocks with shuffled actions
        result = []
        action_block_idx = 0
        skip_until_idx = -1

        for i, line in enumerate(lines):
            if i <= skip_until_idx:
                continue

            # Check if this is the start of an action block we need to replace
            is_action_start = False
            for start_idx, _ in action_blocks:
                if i == start_idx:
                    is_action_start = True
                    # Find the end of this action block
                    for end_idx, action_lines in action_blocks:
                        if start_idx == end_idx:
                            skip_until_idx = start_idx + len(action_lines) - 1
                            break
                    break

            if is_action_start:
                # Insert the next shuffled action instead
                if action_block_idx < len(all_actions):
                    result.extend(all_actions[action_block_idx])
                    action_block_idx += 1
            else:
                result.append(line)

        # Add any remaining shuffled actions that weren't inserted yet
        final_closing_idx = -1
        for i in range(len(result) - 1, -1, -1):
            if result[i].strip() == ')':
                final_closing_idx = i
                break

        if final_closing_idx != -1:
            while action_block_idx < len(all_actions):
                for line in reversed(all_actions[action_block_idx]):
                    result.insert(final_closing_idx, line)
                action_block_idx += 1

        return result

    def _extract_domain_context(self, lines: List[str]) -> Dict:
        """Extract domain context (locations, items, types) for action generation"""
        context = {
            'locations': [],
            'items': [],
            'objects_by_type': {},
            'available_predicates': []
        }

        in_constants = False
        for line in lines:
            if ':constants' in line:
                in_constants = True
            elif line.strip() == ')' and in_constants:
                in_constants = False
            elif in_constants and ' - ' in line:
                # Parse constants like "customer_area staff_area - location"
                parts = line.split(' - ')
                if len(parts) == 2:
                    obj_type = parts[1].strip()
                    obj_names = [name.strip() for name in parts[0].split() if name.strip()]

                    context['objects_by_type'][obj_type] = context['objects_by_type'].get(obj_type, []) + obj_names

                    if obj_type == 'location':
                        context['locations'].extend(obj_names)
                    elif obj_type in ['item', 'object', 'objecttype']:
                        context['items'].extend(obj_names)

        # Extract predicates
        for line in lines:
            if ':predicates' in line:
                # Simple extraction of predicate names
                if 'robot_at' in line:
                    context['available_predicates'].append('robot_at')
                if 'object_at' in line:
                    context['available_predicates'].append('object_at')
                if 'robot_has' in line:
                    context['available_predicates'].append('robot_has')
                break

        return context

    def _add_action_dependencies(self, lines: List[str], dependencies: Dict, context: Dict) -> List[str]:
        """Add missing objects/predicates needed by actions to the domain"""
        result = []

        for i, line in enumerate(lines):
            result.append(line)

            # Add missing predicates to the :predicates section
            if ':predicates' in line:
                # Extract existing predicates from the line
                pred_match = re.match(r'(\s*\(:predicates\s*)(.+?)(\)\s*)?$', line)
                if pred_match and dependencies['predicates']:
                    prefix = pred_match.group(1)
                    existing_preds = pred_match.group(2)

                    # Extract predicate names that are already defined
                    existing_pred_names = set()
                    for pred in existing_preds.split():
                        if pred.startswith('(') and not pred.startswith('(?'):
                            pred_name = pred.replace('(', '').split()[0]
                            existing_pred_names.add(pred_name)

                    # Add missing predicates as simple zero-arity predicates
                    new_predicates = []
                    for pred in dependencies['predicates']:
                        if pred not in existing_pred_names and pred not in ['and', 'not', 'or']:
                            new_predicates.append(f"({pred})")

                    if new_predicates:
                        all_preds = existing_preds + ' ' + ' '.join(new_predicates)
                        result[-1] = f"{prefix}{all_preds})"

        return result

    def _generate_domain_aware_actions(self, action_names: List[str], context: Dict) -> List[str]:
        """Generate actions from actions.txt using actual domain constants"""
        complex_actions = []

        locations = context['locations'] if context['locations'] else ['base_location', 'work_area']
        items = context['items'] if context['items'] else ['tool', 'object']

        for action_name in action_names:
            if action_name in self.actions:
                precond_template, effect_template = self.actions[action_name]

                # Substitute placeholders with actual domain elements
                precond = self._substitute_placeholders(precond_template, locations, items)
                effect = self._substitute_placeholders(effect_template, locations, items)

                action_lines = [
                    f" (:action {action_name}",
                    f"  :parameters ()",
                    f"  :precondition {precond}",
                    f"  :effect {effect})"
                ]

                complex_actions.append(action_lines)

        return complex_actions

    def _substitute_placeholders(self, template: str, locations: List[str], items: List[str]) -> str:
        """Replace placeholder variables with actual domain constants"""
        result = template

        # Replace location placeholders
        if '?target_area' in result:
            result = result.replace('?target_area', locations[0] if locations else 'work_area')
        if '?source_area' in result:
            result = result.replace('?source_area', locations[-1] if len(locations) > 1 else 'base_location')
        if '?pickup_location' in result:
            result = result.replace('?pickup_location', locations[0] if locations else 'storage_area')
        if '?delivery_location' in result:
            result = result.replace('?delivery_location', locations[-1] if len(locations) > 1 else 'delivery_area')
        if '?work_area' in result:
            result = result.replace('?work_area', locations[-1] if len(locations) > 1 else 'work_area')
        if '?storage_area' in result:
            result = result.replace('?storage_area', locations[0] if locations else 'storage_area')
        if '?inventory_location' in result:
            result = result.replace('?inventory_location', locations[0] if locations else 'inventory_room')
        if '?delivery_point' in result:
            result = result.replace('?delivery_point', locations[-1] if len(locations) > 1 else 'delivery_point')
        if '?coordination_hub' in result:
            result = result.replace('?coordination_hub', locations[0] if locations else 'control_center')
        if '?processing_station' in result:
            result = result.replace('?processing_station', locations[-1] if len(locations) > 1 else 'processing_area')

        # Replace object placeholders
        if '?object' in result:
            result = result.replace('?object', items[0] if items else 'tool')

        return result

    def _inject_problem(self, content: str, objects: List[Tuple[str, str]]) -> str:
        lines = content.split('\n')
        result = []

        for line in lines:
            result.append(line)

            # Inject objects after :objects line
            if ':objects' in line:
                # Group objects by type
                by_type = {}
                for obj_name, obj_type in objects:
                    if obj_type not in by_type:
                        by_type[obj_type] = []
                    by_type[obj_type].append(obj_name)

                # Add lines for each type
                for obj_type, obj_names in by_type.items():
                    obj_str = ' '.join(obj_names)
                    result.append(f"   {obj_str} - {obj_type}")

        return '\n'.join(result)


def main():
    parser = argparse.ArgumentParser(description='Inject PDDL redundancy')
    parser.add_argument('input', help='Input directory or parent directory')
    parser.add_argument('output', help='Output directory')
    parser.add_argument('--sources', default=None, help='Redundancy sources directory')
    parser.add_argument('--num-objects', type=int, default=5, help='Number of objects to inject')
    parser.add_argument('--num-actions', type=int, default=3, help='Number of actions to inject')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--shuffle-actions', action='store_true',
                       help='Randomly shuffle action order instead of appending at end')

    args = parser.parse_args()

    # Use same directory as script if no sources specified
    sources_dir = Path(args.sources) if args.sources else Path(__file__).parent
    injector = PDDLInjector(sources_dir, seed=args.seed, shuffle_actions=args.shuffle_actions)

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Injecting {args.num_objects} objects and {args.num_actions} actions...")

    if (input_path / "domain.pddl").exists():
        # Single directory
        success = injector.inject_directory(input_path, output_path, args.num_objects, args.num_actions)
        print(f"✅ Success: {1 if success else 0}")
    else:
        # Multiple directories
        task_dirs = [d for d in input_path.iterdir() if d.is_dir() and (d / "domain.pddl").exists()]
        print(f"Found {len(task_dirs)} directories to process")

        success_count = 0
        for task_dir in task_dirs:
            task_output = output_path / task_dir.name
            if injector.inject_directory(task_dir, task_output, args.num_objects, args.num_actions):
                success_count += 1
                print(f"✅ {task_dir.name}")

        print(f"✅ Success: {success_count}")
        print(f"❌ Failed: {len(task_dirs) - success_count}")


if __name__ == "__main__":
    main()