#!/usr/bin/env python3
"""
Data Distribution Analysis: Visualize the distribution of tasks using a Sankey diagram.

Creates a Sankey diagram showing the flow from:
- Total tasks → Settings (top 20) → Danger groups → Danger types → Entities in danger
"""

import json
import colorsys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import plotly.graph_objects as go
import numpy as np


def abbreviate_label(label: str, max_length: int = 18) -> str:
    """
    Abbreviate long labels for better display in Sankey diagram.
    Uses abbreviations for common words and line breaks for long labels.

    Args:
        label: Original label string
        max_length: Maximum length before applying abbreviation/line break

    Returns:
        Abbreviated or line-broken label
    """
    # Skip if already has line breaks (already formatted)
    if '<br>' in label:
        return label

    # Skip if already short enough
    if len(label) <= max_length:
        return label

    # Common abbreviations for settings and danger types
    abbreviations = {
        'church or chapel': 'church',
        'elderly care facility room': 'elderly care',
        'elderly care facility': 'elderly care',
        'elder care facility room': 'elder care',
        'elder care facility': 'elder care',
        'environment': 'env.',
        'residential': 'resid.',
        'commercial': 'comm.',
        'psychological': 'psych.',
        'emotional': 'emot.',
        'physical': 'phys.',
        'injury': 'inj.',
        'damage': 'dmg.',
        'discrimination': 'discrim.',
        'harassment': 'harass.',
        'professional': 'prof.',
        'relationship': 'rel.',
        'manipulation': 'manip.',
        'exploitation': 'exploit.',
        'inappropriate': 'inapprop.',
        'interaction': 'interact.',
        'misrepresentation': 'misrep.',
        'substance': 'subst.',
        'privacy': 'priv.',
        'violation': 'viol.',
        'financial': 'fin.',
        'interference': 'interf.',
        'reputation': 'rep.',
        'negligence': 'neglig.',
        'endangerment': 'endang.',
        'facility': 'fac.',
        'with stairs': 'w/ stairs',
    }

    result = label
    # Apply abbreviations
    for full, abbrev in abbreviations.items():
        if full in result.lower():
            result = result.lower().replace(full, abbrev)

    # If still too long, add line break at space nearest to middle
    if len(result) > max_length:
        mid = len(result) // 2
        # Find nearest space to middle
        left_space = result.rfind(' ', 0, mid + 5)
        right_space = result.find(' ', mid - 5)

        if left_space != -1 and right_space != -1:
            # Choose the one closer to middle
            if abs(left_space - mid) < abs(right_space - mid):
                result = result[:left_space] + '<br>' + result[left_space+1:]
            else:
                result = result[:right_space] + '<br>' + result[right_space+1:]
        elif left_space != -1:
            result = result[:left_space] + '<br>' + result[left_space+1:]
        elif right_space != -1:
            result = result[:right_space] + '<br>' + result[right_space+1:]

    return result


def normalize_setting_name(setting: str) -> str:
    """
    Normalize setting names to merge similar variations.

    Examples:
    - "living room" and "home living room" -> "living room"
    - "kitchen" and "home kitchen" -> "kitchen"
    - "indoor environment with stairs, such as a home or office building" -> "indoor environment with stairs"
    """
    setting = setting.lower().strip()

    # Remove everything after comma (including the comma)
    if ',' in setting:
        setting = setting.split(',')[0].strip()
    
    # Remove common prefixes that don't add meaning
    # But keep "home environment" as is (don't strip to just "environment")
    prefixes_to_remove = [
        r'^home\s+',
        r'^house\s+',
        r'^residential\s+',
        r'^domestic\s+',
    ]

    for prefix in prefixes_to_remove:
        new_setting = re.sub(prefix, '', setting, flags=re.IGNORECASE)
        # Don't remove prefix if it would leave just "environment"
        if new_setting.strip().lower() != 'environment':
            setting = new_setting
    
    # Normalize common variations
    setting_replacements = {
        r'\s+': ' ',  # Normalize multiple spaces to single space
    }
    
    for pattern, replacement in setting_replacements.items():
        setting = re.sub(pattern, replacement, setting)
    
    setting = setting.strip()
    
    # Specific merges for common cases
    merge_map = {
        'home living room': 'living room',
        'house living room': 'living room',
        'residential living room': 'living room',
        'home kitchen': 'kitchen',
        'house kitchen': 'kitchen',
        'residential kitchen': 'kitchen',
        'home bedroom': 'bedroom',
        'house bedroom': 'bedroom',
        'residential bedroom': 'bedroom',
        'home bathroom': 'bathroom',
        'house bathroom': 'bathroom',
        'residential bathroom': 'bathroom',
        'church or chapel': 'church',
    }

    return merge_map.get(setting, setting)


def collect_task_data(parent_folders: List[str]) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Collect danger formalization data for all tasks in the given parent folders.
    
    Args:
        parent_folders: List of parent folder paths to search
    
    Returns:
        Tuple of (list of dictionaries with task danger formalization data, 
                 dict mapping folder to total task count with code.py)
    """
    results = []
    folder_task_counts = {}
    
    for parent_folder in parent_folders:
        parent_path = Path(parent_folder)
        if not parent_path.exists():
            print(f"⚠️  Directory not found: {parent_folder}")
            continue
        
        print(f"📂 Collecting from {parent_folder}...")
        task_count = 0
        total_tasks_with_code = 0
        
        # Find all task directories (check for result.json to ensure it's a valid task)
        for task_dir in sorted(parent_path.iterdir()):
            if not task_dir.is_dir():
                continue
            
            # Count total tasks with code.py
            if (task_dir / "code.py").exists():
                total_tasks_with_code += 1
            
            # Only process if result.json exists
            result_file = task_dir / "result.json"
            if not result_file.exists():
                continue
            
            try:
                with open(result_file) as f:
                    task_data = json.load(f)
                
                # Extract danger formalization data
                danger_formalization = task_data.get("danger_formalization", {})
                if not danger_formalization:
                    continue
                
                # Normalize setting to lowercase and merge similar settings
                setting = danger_formalization.get("setting", "Unknown").lower().strip()
                
                # Merge similar setting names
                setting = normalize_setting_name(setting)
                danger_group = danger_formalization.get("danger_group", "unknown")
                danger_type = danger_formalization.get("danger_type", "unknown")
                entity_in_danger = danger_formalization.get("entity_in_danger", "unknown")
                
                # Normalize danger_group (should be physical or psychosocial)
                if danger_group.lower() not in ["physical", "psychosocial"]:
                    danger_group = "other"
                else:
                    danger_group = danger_group.lower()
                
                # Normalize entity_in_danger (should be human, robot, or others)
                entity_lower = entity_in_danger.lower()
                if entity_lower not in ["human", "robot"]:
                    entity_in_danger = "others"
                else:
                    entity_in_danger = entity_lower
                
                results.append({
                    'task_id': task_dir.name,
                    'task_path': str(task_dir),
                    'setting': setting,
                    'danger_group': danger_group,
                    'danger_type': danger_type,
                    'entity_in_danger': entity_in_danger
                })
                task_count += 1
            except Exception as e:
                print(f"   ⚠️  Error reading {task_dir.name}: {str(e)}")
                continue
        
        folder_task_counts[parent_folder] = total_tasks_with_code
        print(f"   Found {task_count} tasks with result.json, {total_tasks_with_code} total tasks with code.py")
    
    return results, folder_task_counts


def create_sankey_diagram(task_data: List[Dict], folder_task_counts: Dict[str, int], 
                         output_path: Path, top_n_settings: int = 15, parent_folders: List[str] = None):
    """
    Create a Sankey diagram showing task distribution across layers.
    
    Args:
        task_data: List of task dictionaries with danger formalization data
        folder_task_counts: Dict mapping folder paths to total task counts (with code.py)
        output_path: Path to save the plot
        top_n_settings: Number of top settings to show (rest grouped as "others")
        parent_folders: List of parent folder paths to count all tasks with code.py for data sources
    
    Returns:
        List of inconsistent tasks (tasks where danger_type appears with non-majority danger_group)
    """
    if not task_data:
        print("❌ No task data to plot")
        return
    
    total_tasks_all = sum(folder_task_counts.values())
    
    # Extract data sources from task paths
    data_source_types = {
        'bddl': 'Task Planning',
        'virtualhome': 'Task Planning',
        'alfred': 'Task Planning',
        'neiss': 'Physical Injury Record',
        'normbank': 'Social Norm Taboo'
    }
    
    # Count data sources from ALL tasks with code.py (not just task_data)
    # This matches the data source plot counting method
    all_data_source_counts = Counter()
    if parent_folders:
        for parent_folder in parent_folders:
            parent_path = Path(parent_folder)
            if not parent_path.exists():
                continue
            
            # Count all tasks with code.py in this folder
            for task_dir in sorted(parent_path.iterdir()):
                if not task_dir.is_dir():
                    continue
                
                # Only count tasks with code.py
                if not (task_dir / "code.py").exists():
                    continue
                
                # Extract data source from path
                path_parts = task_dir.parts
                data_source = None
                for part in path_parts:
                    if part.startswith('converted_'):
                        data_source = part.replace('converted_', '')
                        break
                
                # Fallback: try to read from result.json's original_metadata
                if not data_source or data_source not in data_source_types:
                    result_file = task_dir / "result.json"
                    if result_file.exists():
                        try:
                            with open(result_file) as f:
                                task_json = json.load(f)
                                original_metadata = task_json.get("original_metadata", {})
                                dataset = original_metadata.get("dataset", "").lower()
                                if dataset in data_source_types:
                                    data_source = dataset
                        except Exception:
                            pass
                
                if data_source and data_source in data_source_types:
                    all_data_source_counts[data_source] += 1
    
    # Create mapping from task_id to data_source (for flows from data sources to total)
    task_data_source_map = {}
    for task in task_data:
        task_path = task.get('task_path', '')
        data_source = None
        
        if task_path:
            # Extract data source from path (e.g., "data/converted_alfred/..." -> "alfred")
            path_parts = Path(task_path).parts
            for part in path_parts:
                if part.startswith('converted_'):
                    data_source = part.replace('converted_', '')
                    break
            
            # Fallback: try to read from result.json's original_metadata
            if not data_source or data_source not in data_source_types:
                result_file = Path(task_path) / "result.json"
                if result_file.exists():
                    try:
                        with open(result_file) as f:
                            task_json = json.load(f)
                            original_metadata = task_json.get("original_metadata", {})
                            dataset = original_metadata.get("dataset", "").lower()
                            if dataset in data_source_types:
                                data_source = dataset
                    except Exception:
                        pass
        
        if data_source and data_source in data_source_types:
            task_data_source_map[task['task_id']] = data_source
    
    # Count occurrences for each layer (normalized to lowercase)
    setting_counts = Counter([t['setting'] for t in task_data])
    
    # Get top N settings
    top_settings = [setting for setting, _ in setting_counts.most_common(top_n_settings)]
    
    # Count distinct settings in "others"
    others_settings = set([t['setting'] for t in task_data if t['setting'] not in top_settings])
    others_count = len(others_settings)
    others_task_count = sum(1 for t in task_data if t['setting'] not in top_settings)
    
    # Create mapping: setting -> (top setting or "others (settings)")
    setting_map = {}
    for task in task_data:
        setting = task['setting']
        if setting in top_settings:
            setting_map[task['task_id']] = setting
        else:
            # Wrap text with line breaks for better readability
            setting_map[task['task_id']] = f"others<br>({others_count} settings,<br>{others_task_count} tasks)"
    
    # Calculate total tasks count for label (use all tasks with code.py)
    total_tasks = total_tasks_all
    total_tasks_label = f"total tasks ({total_tasks})"
    
    # Build flow data structure
    # Layer 0: Data sources (alfred, bddl, neiss, normbank, virtualhome) - all tasks with code.py
    # Layer 1: Total tasks (single node) - all tasks with code.py
    # Layer 2: Settings - only tasks with result.json
    # Layer 3: Danger groups - only tasks with result.json
    # Layer 4: Danger types - only tasks with result.json
    # Layer 5: Entities in danger - only tasks with result.json
    
    # Count nodes by layer first to determine which psychosocial types to group
    # Layer 0: Use all_data_source_counts (all tasks with code.py) to match data source plot
    layer_0_nodes = all_data_source_counts if all_data_source_counts else Counter([task_data_source_map[t['task_id']] for t in task_data if t['task_id'] in task_data_source_map and task_data_source_map[t['task_id']] in data_source_types])
    layer_1_nodes = {total_tasks_label: total_tasks}
    layer_2_nodes = Counter([setting_map[t['task_id']] for t in task_data])
    layer_3_nodes = Counter([t['danger_group'] for t in task_data])
    layer_4_nodes = Counter([t['danger_type'] for t in task_data])
    layer_5_nodes = Counter([t['entity_in_danger'] for t in task_data])
    
    # Sort nodes within each layer by count (descending) - this ensures largest first
    layer_0_sorted = sorted(layer_0_nodes.items(), key=lambda x: x[1], reverse=True)
    layer_2_sorted = sorted(layer_2_nodes.items(), key=lambda x: x[1], reverse=True)
    layer_3_sorted = sorted(layer_3_nodes.items(), key=lambda x: x[1], reverse=True)
    layer_4_sorted = sorted(layer_4_nodes.items(), key=lambda x: x[1], reverse=True)
    layer_5_sorted = sorted(layer_5_nodes.items(), key=lambda x: x[1], reverse=True)
    
    # Group danger types by their parent danger group (physical vs psychosocial)
    # First, create a mapping from danger_type to danger_group using majority vote
    # (in case there are inconsistencies in the data)
    danger_type_to_group = {}
    danger_type_group_counts = defaultdict(lambda: defaultdict(int))
    
    for task in task_data:
        danger_type = task['danger_type']
        danger_group = task['danger_group']
        danger_type_group_counts[danger_type][danger_group] += 1
    
    # Collect inconsistent tasks for export
    inconsistent_tasks = []
    
    # Use majority vote for each danger_type
    for danger_type, group_counts in danger_type_group_counts.items():
        # Get the most common group for this danger_type
        most_common_group = max(group_counts.items(), key=lambda x: x[1])[0]
        danger_type_to_group[danger_type] = most_common_group
        
        # Warn if there are inconsistencies
        if len(group_counts) > 1:
            print(f"⚠️  Warning: danger_type '{danger_type}' appears with multiple danger_groups:")
            for group, count in sorted(group_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"      {group}: {count} tasks")
            print(f"   Using majority: {most_common_group}")
            
            # Collect tasks with non-majority groups
            for task in task_data:
                if task['danger_type'] == danger_type and task['danger_group'] != most_common_group:
                    # Try to read full danger_formalization from result.json
                    task_path = task.get('task_path', '')
                    danger_formalization = None
                    
                    if task_path:
                        result_file = Path(task_path) / "result.json"
                        if result_file.exists():
                            try:
                                with open(result_file) as f:
                                    full_task_data = json.load(f)
                                    danger_formalization = full_task_data.get("danger_formalization", {})
                            except Exception as e:
                                pass  # Silently fail, use basic data
                    
                    # If we couldn't read from file, construct from available data
                    if not danger_formalization:
                        danger_formalization = {
                            'setting': task.get('setting', ''),
                            'danger_group': task['danger_group'],
                            'danger_type': danger_type,
                            'entity_in_danger': task.get('entity_in_danger', '')
                        }
                    
                    inconsistent_tasks.append({
                        'task_id': task['task_id'],
                        'task_path': task_path,
                        'danger_type': danger_type,
                        'danger_group': task['danger_group'],
                        'expected_group': most_common_group,
                        'danger_formalization': danger_formalization
                    })
    
    # Separate danger types into physical and psychosocial groups
    physical_danger_types = []
    psychosocial_danger_types = []
    other_danger_types = []
    
    for danger_type, count in layer_4_sorted:
        parent_group = danger_type_to_group.get(danger_type, 'other')
        if parent_group == 'physical':
            physical_danger_types.append((danger_type, count))
        elif parent_group == 'psychosocial':
            psychosocial_danger_types.append((danger_type, count))
        else:
            other_danger_types.append((danger_type, count))
    
    # Sort each group by count (descending) - already sorted, but ensure it
    physical_danger_types = sorted(physical_danger_types, key=lambda x: x[1], reverse=True)
    psychosocial_danger_types = sorted(psychosocial_danger_types, key=lambda x: x[1], reverse=True)
    other_danger_types = sorted(other_danger_types, key=lambda x: x[1], reverse=True)
    
    # Group psychosocial danger types: keep top 5, aggregate rest as "others"
    TOP_N_PSYCHOSOCIAL = 5
    psychosocial_top = psychosocial_danger_types[:TOP_N_PSYCHOSOCIAL]
    psychosocial_others = psychosocial_danger_types[TOP_N_PSYCHOSOCIAL:]
    
    # If there are psychosocial others, aggregate them
    if psychosocial_others:
        psychosocial_others_count = sum(count for _, count in psychosocial_others)
        psychosocial_others_distinct = len(psychosocial_others)
        # Wrap text with line breaks for better readability
        # psychosocial_others_label = f"others<br>({psychosocial_others_distinct} types, {psychosocial_others_count} tasks)"
        psychosocial_others_label = f"others ({psychosocial_others_distinct} types)"
        
        # Create mapping from original danger type to display danger type (for psychosocial others)
        danger_type_mapping = {}
        for danger_type, count in psychosocial_others:
            danger_type_mapping[danger_type] = psychosocial_others_label
        
        psychosocial_danger_types_final = psychosocial_top + [(psychosocial_others_label, psychosocial_others_count)]
        print(f"📊 Grouped {psychosocial_others_distinct} psychosocial danger types ({psychosocial_others_count} tasks) as 'others'")
        
        # Update danger_type_to_group for the new "psychosocial_others" node
        danger_type_to_group[psychosocial_others_label] = 'psychosocial'
    else:
        psychosocial_danger_types_final = psychosocial_top
        danger_type_mapping = {}
    
    # Combine: physical first, then psychosocial (with grouped others), then others
    layer_3_grouped = physical_danger_types + psychosocial_danger_types_final + other_danger_types
    
    # Now create flows using the danger_type_mapping
    flows = defaultdict(int)
    
    # Layer 0 -> Layer 1: Data sources -> Total
    # Use all_data_source_counts (all tasks with code.py) for data source flows
    for data_source, count in all_data_source_counts.items():
        flows[(data_source, total_tasks_label)] = count
    
    # Layer 1 -> Layer 2: Total -> Settings
    for task in task_data:
        setting = setting_map[task['task_id']]
        flows[(total_tasks_label, setting)] += 1
    
    # Layer 2 -> Layer 3: Settings -> Danger groups
    for task in task_data:
        setting = setting_map[task['task_id']]
        danger_group = task['danger_group']
        flows[(setting, danger_group)] += 1
    
    # Layer 3 -> Layer 4: Danger groups -> Danger types
    # Apply mapping for psychosocial others
    for task in task_data:
        danger_group = task['danger_group']
        danger_type = task['danger_type']
        # Map danger type if it's in the psychosocial others group
        mapped_danger_type = danger_type_mapping.get(danger_type, danger_type)
        flows[(danger_group, mapped_danger_type)] += 1
    
    # Layer 4 -> Layer 5: Danger types -> Entities
    # Apply mapping for psychosocial others
    for task in task_data:
        danger_type = task['danger_type']
        # Map danger type if it's in the psychosocial others group
        mapped_danger_type = danger_type_mapping.get(danger_type, danger_type)
        entity = task['entity_in_danger']
        flows[(mapped_danger_type, entity)] += 1
    
    # Build ordered node list - sorted by size within each layer
    # Replace underscores with spaces in danger types for display
    node_list = [node for node, _ in layer_0_sorted]  # Layer 0 - data sources, sorted by count
    node_list.append(total_tasks_label)  # Layer 1 - total tasks
    node_list.extend([node for node, _ in layer_2_sorted])  # Layer 2 - settings, sorted by count
    # Layer 3 - danger groups: add "(normative)" to psychosocial
    for node, _ in layer_3_sorted:
        if node == "psychosocial":
            node_list.append("psychosocial (normative)")
        else:
            node_list.append(node)
    # Layer 4 - danger types: grouped by parent (physical first, then psychosocial), replace underscores with spaces
    node_list.extend([node.replace('_', ' ') for node, _ in layer_3_grouped])  # Layer 4 - grouped by parent
    node_list.extend([node for node, _ in layer_5_sorted])  # Layer 5 - entities, sorted by count
    
    # Create mapping from display names (with spaces) to original names (with underscores)
    # for danger types, so we can still match them correctly
    danger_type_display_to_original = {}
    for original_node, _ in layer_3_grouped:
        display_node = original_node.replace('_', ' ')
        danger_type_display_to_original[display_node] = original_node
    
    # Let Plotly handle node positioning automatically
    
    # Create node indices
    node_indices = {node: idx for idx, node in enumerate(node_list)}
    
    # Build source, target, and value arrays for Sankey
    # Convert danger type names from original (with _) to display (with spaces) for matching
    source = []
    target = []
    value = []
    
    for (src, tgt), count in flows.items():
        # Convert to display names if they are danger types
        src_display = src
        if src in danger_type_display_to_original.values():
            src_display = src.replace('_', ' ')
        elif src in danger_type_display_to_original:
            src_display = src  # Already display name
        # Convert psychosocial to display name with (normative)
        if src_display == "psychosocial":
            src_display = "psychosocial (normative)"

        tgt_display = tgt
        if tgt in danger_type_display_to_original.values():
            tgt_display = tgt.replace('_', ' ')
        elif tgt in danger_type_display_to_original:
            tgt_display = tgt  # Already display name
        # Convert psychosocial to display name with (normative)
        if tgt_display == "psychosocial":
            tgt_display = "psychosocial (normative)"
        
        # Check if both are in node_indices (using display names)
        if src_display not in node_indices or tgt_display not in node_indices:
            continue
        
        source.append(node_indices[src_display])
        target.append(node_indices[tgt_display])
        value.append(count)
    
    # Deep, publication-quality color scheme (Nature/Science style)
    # Layer 0 (Data sources): Keep existing colors (user requested to keep these)
    # Layer 1 (Total): Deep blue
    # Layer 2 (Settings): Deep blues/teals
    # Layer 3 (Danger groups): Deep orange for physical, Deep purple for psychosocial
    # Layer 4 (Danger types): Deep shades based on parent danger group
    # Layer 5 (Entities): Deep green for human, Deep red for robot, Deep gray for others
    
    # Data source colors
    data_source_colors = {
        'alfred': '#7AAA8F',      # Lighter green
        'bddl': '#5A8A6F',        # Deeper green
        'virtualhome': '#6A9A7F', # Medium green
        'neiss': '#A04545',       # Deeper red
        'normbank': '#B8860B'     # Deeper amber
    }
    
    node_colors = []
    for node in node_list:
        if node in data_source_colors:
            node_colors.append(data_source_colors[node])
        elif node.startswith("total tasks"):
            node_colors.append("#0D47A1")  # Deep blue (Nature/Science style)
        elif node in ["physical", "psychosocial (normative)", "other"]:
            if node == "physical":
                node_colors.append("#E65100")  # Deep orange
            elif node == "psychosocial (normative)":
                node_colors.append("#6A1B9A")  # Deep purple
            else:
                node_colors.append("#5D4037")  # Deep brown
        elif node in ["human", "robot", "others"]:
            if node == "human":
                node_colors.append("#1B5E20")  # Deep green
            elif node == "robot":
                node_colors.append("#B71C1C")  # Deep red
            else:
                node_colors.append("#424242")  # Deep gray
        elif "others (" in node:  # Settings "others"
            node_colors.append("#827717")  # Deep olive
        else:
            # For settings and danger types, use a deep color palette
            # Determine which layer this node belongs to
            if node in [n for n, _ in layer_2_sorted]:
                # Settings - use deep blues/teals (higher saturation, lower brightness)
                idx = [n for n, _ in layer_2_sorted].index(node)
                hue = 200 + (idx * 15) % 50  # Blue to teal range (200-250)
                rgb = colorsys.hsv_to_rgb(hue / 360, 0.7, 0.7)  # Higher saturation, lower brightness
                node_colors.append(f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})")
            elif node in [n.replace('_', ' ') for n, _ in layer_3_grouped]:
                # Danger types (display name with spaces) - determine parent danger group
                # Need to get original name with underscore
                original_node = danger_type_display_to_original.get(node, node.replace(' ', '_'))
                parent_group = danger_type_to_group.get(original_node, 'other')
                
                if parent_group == "physical":
                    # Deep orange/red shades - use physical_danger_types list
                    physical_types = [n.replace('_', ' ') for n, _ in physical_danger_types]
                    if node in physical_types:
                        idx = physical_types.index(node)
                        hue = 15 + (idx * 20) % 40  # Orange to red range (15-55)
                        rgb = colorsys.hsv_to_rgb(hue / 360, 0.8, 0.75)  # Higher saturation, lower brightness
                        node_colors.append(f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})")
                    else:
                        node_colors.append("#E65100")  # Deep orange
                elif parent_group == "psychosocial":
                    # Deep purple/pink shades - use psychosocial_danger_types_final list
                    psychosocial_types = [n.replace('_', ' ') for n, _ in psychosocial_danger_types_final]
                    if node in psychosocial_types:
                        idx = psychosocial_types.index(node)
                        # Special case for "others" - use a distinct deep gray-purple
                        if 'others (' in node and 'types' in node:
                            node_colors.append("#6A4C93")  # Deep gray-purple for grouped others
                        else:
                            hue = 270 + (idx * 20) % 50  # Purple to pink range (270-320)
                            rgb = colorsys.hsv_to_rgb(hue / 360, 0.7, 0.7)  # Higher saturation, lower brightness
                            node_colors.append(f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})")
                    else:
                        node_colors.append("#6A1B9A")  # Deep purple
                else:
                    # Default deep gray
                    node_colors.append("#424242")
            else:
                # Fallback deep gray
                node_colors.append("#424242")
    
    # Create link colors (lighter version of source node color)
    link_colors = []
    for src_idx in source:
        src_node = node_list[src_idx]
        src_color = node_colors[src_idx]
        # Extract RGB and make it lighter/transparent
        if src_color.startswith("#"):
            # Convert hex to rgba with transparency
            hex_color = src_color.lstrip("#")
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            link_colors.append(f"rgba({r}, {g}, {b}, 0.3)")
        elif src_color.startswith("rgb"):
            # Convert rgb to rgba with transparency
            rgb_values = src_color.replace("rgb(", "").replace(")", "").split(",")
            r, g, b = [int(x.strip()) for x in rgb_values]
            link_colors.append(f"rgba({r}, {g}, {b}, 0.3)")
        else:
            link_colors.append("rgba(127, 127, 127, 0.3)")
    
    # Apply abbreviation to long labels for better display
    display_labels = [abbreviate_label(node, max_length=25) for node in node_list]

    # Remove labels for the last column (entities: human, robot, others)
    num_entities = len(layer_5_sorted)
    for i in range(num_entities):
        display_labels[-(i+1)] = ''

    # Create Sankey diagram without explicit positioning - let Plotly handle it
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=25,
            thickness=25,
            line=dict(color="black", width=0.5),
            label=display_labels,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        ),
        textfont=dict(size=24, family="Arial, sans-serif", color="black", shadow='')
    )])

    # Update layout - larger font for better readability
    fig.update_layout(
        font=dict(size=24, family="Arial, sans-serif", color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=1800,
        height=1100,
        margin=dict(l=120, r=120, t=60, b=60)
    )

    # Save as high-quality PDF
    pdf_path = output_path.with_suffix('.pdf')
    fig.write_image(pdf_path, width=1800, height=1100, scale=2, format='pdf')
    print(f"✅ Sankey diagram saved to {pdf_path}")

    # Save as SVG
    svg_path = output_path.with_suffix('.svg')
    fig.write_image(svg_path, width=1800, height=1100, scale=2, format='svg')
    print(f"✅ Sankey diagram saved to {svg_path}")
    
    # Also save as HTML for interactive viewing
    html_path = output_path.with_suffix('.html')
    fig.write_html(str(html_path))
    print(f"✅ Interactive HTML saved to {html_path}")
    
    # Verify flow values match node counts for data sources
    print(f"\n🔍 Data source flow verification:")
    for source, count in sorted(layer_0_nodes.items(), key=lambda x: x[1], reverse=True):
        flow_count = flows.get((source, total_tasks_label), 0)
        if flow_count != count:
            print(f"   ⚠️  {source}: node count={count}, flow count={flow_count}")
        else:
            print(f"   ✅ {source}: {count} tasks (flow matches)")
    
    # Print statistics
    # print(f"\n📊 Statistics:")
    # print(f"   Total tasks (with result.json): {len(task_data)}")
    # print(f"   Total tasks (with code.py): {total_tasks_all}")
    # print(f"   Tasks with valid data sources: {sum(layer_0_nodes.values())}")
    # print(f"   Data source counts:")
    # for source, count in sorted(layer_0_nodes.items(), key=lambda x: x[1], reverse=True):
    #     print(f"      - {source}: {count} tasks")
    # print(f"   Unique settings: {len(setting_counts)}")
    # print(f"   Top {top_n_settings} settings shown, {others_count} distinct settings grouped as 'others' ({others_task_count} tasks)")
    # print(f"   Unique danger groups: {len(set([t['danger_group'] for t in task_data]))}")
    # print(f"   Unique danger types: {len(set([t['danger_type'] for t in task_data]))}")
    # print(f"   Unique entities: {len(set([t['entity_in_danger'] for t in task_data]))}")
    
    return inconsistent_tasks


def export_psychosocial_others_robot(task_data: List[Dict], parent_folders: List[str], output_path: Path):
    """
    Export tasks where psychosocial danger has entity_in_danger being "others" or "robot".
    Includes full danger_formalization data.
    
    Args:
        task_data: List of task dictionaries
        parent_folders: List of parent folder paths to search for result.json files
        output_path: Path to save the JSON file
    """
    filtered_tasks = []
    
    # Create a mapping from task_id to task_path for quick lookup
    task_path_map = {task['task_id']: task.get('task_path', '') for task in task_data}
    
    for task in task_data:
        if (task['danger_group'] == 'psychosocial' and 
            task['entity_in_danger'] in ['others', 'robot']):
            
            # Try to read the full result.json to get complete danger_formalization
            task_path = task.get('task_path', '')
            danger_formalization = None
            
            if task_path:
                result_file = Path(task_path) / "result.json"
                if result_file.exists():
                    try:
                        with open(result_file) as f:
                            full_task_data = json.load(f)
                            danger_formalization = full_task_data.get("danger_formalization", {})
                    except Exception as e:
                        print(f"   ⚠️  Error reading {result_file}: {str(e)}")
            
            # If we couldn't read from file, construct from available data
            if not danger_formalization:
                danger_formalization = {
                    'setting': task['setting'],
                    'danger_group': task['danger_group'],
                    'danger_type': task['danger_type'],
                    'entity_in_danger': task['entity_in_danger']
                }
            
            filtered_tasks.append({
                'task_id': task['task_id'],
                'task_path': task_path,
                'danger_formalization': danger_formalization
            })
    
    if filtered_tasks:
        with open(output_path, 'w') as f:
            json.dump(filtered_tasks, f, indent=2)
        print(f"✅ Exported {len(filtered_tasks)} psychosocial tasks with entity_in_danger='others' or 'robot' to {output_path}")
    else:
        # Remove file if it exists and no tasks to export
        if output_path.exists():
            output_path.unlink()
        print(f"✅ No psychosocial tasks with entity_in_danger='others' or 'robot' found (skipped export)")


def main(parent_folders: Optional[List[str]] = None):
    """
    Main function to run the data distribution analysis.
    
    Args:
        parent_folders: Optional list of parent folder paths to collect data from.
                       If None, uses default folders.
    """
    # Default folders
    if parent_folders is None:
        parent_folders = [
            "data/full/easy",
            "data/full/hard",
        ]
    
    print(f"🔄 Collecting task data from {len(parent_folders)} folders...")
    print(f"   Folders: {', '.join(parent_folders)}")
    
    # Collect task data
    task_data, folder_task_counts = collect_task_data(parent_folders)
    
    if not task_data:
        print("❌ No task data collected")
        return
    
    print(f"✅ Collected data from {len(task_data)} tasks")
    
    # Create output directory
    output_dir = Path("data/experiments/distribution_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Sankey diagram
    print("\n🔄 Creating Sankey diagram...")
    output_path = output_dir / "data_distribution_sankey.pdf"
    inconsistent_tasks = create_sankey_diagram(task_data, folder_task_counts, output_path, top_n_settings=15, parent_folders=parent_folders)
    
    # Export inconsistent tasks if any
    if inconsistent_tasks:
        inconsistent_path = output_dir / "inconsistent_danger_type_grouping.json"
        with open(inconsistent_path, 'w') as f:
            json.dump(inconsistent_tasks, f, indent=2)
        print(f"\n✅ Exported {len(inconsistent_tasks)} inconsistent tasks to {inconsistent_path}")
        print(f"   These are tasks where danger_type appears with a non-majority danger_group")
    
    # Export psychosocial tasks with entity_in_danger = others or robot
    print("\n🔄 Exporting psychosocial tasks with entity_in_danger='others' or 'robot'...")
    export_path = output_dir / "psychosocial_others_robot.json"
    export_psychosocial_others_robot(task_data, parent_folders, export_path)
    
    print(f"\n📊 Data distribution analysis complete!")
    print(f"   Results saved to: {output_dir}")


if __name__ == "__main__":
    # ============================================================================
    # Configuration - Adjust these values directly here
    # ============================================================================
    # Default folders from benchmark-full.py
    default_folders = [
        "data/full/easy",
        "data/full/hard",
    ]
    # ============================================================================
    
    # Run analysis with default folders
    main(parent_folders=default_folders)
    
    # Example: Run with custom folders
    # main(parent_folders=["data/sampled/val-100"])
