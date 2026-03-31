#!/usr/bin/env python3
"""
Factor Analysis with Violin Plots: Plan Metrics and Category Analysis

Analyzes plan metrics (length, delta) and categorical factors using violin plots
to show distributions across difficulty levels (Hard, Medium, Easy).
Publication-quality compact visualizations suitable for Science/Nature journals.
"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import glob
import re
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def apply_axis_style(ax, fontsize=12):
    """Apply consistent Nature/Science style to an axis.

    - Black spines with 0.5 linewidth
    - Visible tick marks (length=4, width=0.5) pointing outward
    - All spines visible
    """
    # All spines visible with black color
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(0.5)

    # Tick marks: short, pointing outward, on bottom and left only
    ax.tick_params(axis='both', which='both', direction='out', length=1, width=0.5,
                   labelsize=fontsize, pad=1, colors='black',
                   bottom=True, left=True, top=False, right=False)

    # No grid lines
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', visible=False)


def parse_plan(plan_str: str) -> List[str]:
    """Parse a plan string into a list of actions."""
    if not plan_str or not isinstance(plan_str, str):
        return []
    actions = re.findall(r'\(([^)]+)\)', plan_str)
    return actions


def count_actions(plan_str: str) -> int:
    """Count the number of actions in a plan string."""
    return len(parse_plan(plan_str))


def calculate_plan_length(task_dir: Path, plan_type: str = 'unsafe') -> int:
    """
    Calculate plan length from result.json.

    Args:
        task_dir: Path to task directory
        plan_type: 'unsafe' or 'safe'

    Returns:
        Number of actions in the plan, or 0 if not found
    """
    result_file = task_dir / "result.json"
    if not result_file.exists():
        return 0

    try:
        with open(result_file) as f:
            task_data = json.load(f)

        if 'generated_plans' not in task_data:
            return 0

        plans = task_data['generated_plans']
        if plan_type == 'unsafe':
            plan_str = plans.get('unsafe_plan', '')
        else:
            plan_str = plans.get('safe_plan', '')

        return count_actions(plan_str)
    except Exception:
        return 0


def calculate_plan_delta(task_dir: Path) -> int:
    """
    Calculate plan delta (safe_plan_length - unsafe_plan_length).

    Args:
        task_dir: Path to task directory

    Returns:
        Plan delta, or 0 if not found
    """
    unsafe_len = calculate_plan_length(task_dir, 'unsafe')
    safe_len = calculate_plan_length(task_dir, 'safe')
    return safe_len - unsafe_len


def calculate_danger_group(task_dir: Path) -> str:
    """
    Extract danger_group from result.json.

    Args:
        task_dir: Path to task directory

    Returns:
        Danger group string, or 'unknown' if not found
    """
    result_file = task_dir / "result.json"
    if not result_file.exists():
        return 'unknown'

    try:
        with open(result_file) as f:
            task_data = json.load(f)

        if 'danger_formalization' not in task_data:
            return 'unknown'

        formalization = task_data['danger_formalization']
        return formalization.get('danger_group', 'unknown')
    except Exception:
        return 'unknown'


def normalize_entity_in_danger(value: str) -> str:
    """
    Normalize entity_in_danger value to one of three categories.

    Maps various entity_in_danger values to canonical categories:
    - human: "Human", "human", "child"
    - robot: "robot"
    - others: "Others", "others", "dog", "unknown", and any other values
    """
    if not value or not isinstance(value, str):
        return 'others'

    value_lower = value.lower().strip()

    if value_lower in ['human', 'child']:
        return 'human'
    if value_lower == 'robot':
        return 'robot'
    return 'others'


def calculate_entity_in_danger(task_dir: Path) -> str:
    """
    Extract and normalize entity_in_danger from result.json.

    Args:
        task_dir: Path to task directory

    Returns:
        Normalized entity in danger string: 'human', 'robot', or 'others'
    """
    result_file = task_dir / "result.json"
    if not result_file.exists():
        return 'others'

    try:
        with open(result_file) as f:
            task_data = json.load(f)

        if 'danger_formalization' not in task_data:
            return 'others'

        formalization = task_data['danger_formalization']
        raw_value = formalization.get('entity_in_danger', 'unknown')
        return normalize_entity_in_danger(raw_value)
    except Exception:
        return 'others'


def calculate_data_source(task_dir: Path) -> str:
    """
    Extract data source from result.json.

    Args:
        task_dir: Path to task directory

    Returns:
        Data source string (e.g., 'alfred', 'bddl', 'neiss', etc.) or 'unknown'
    """
    result_file = task_dir / "result.json"
    if not result_file.exists():
        return 'unknown'

    try:
        with open(result_file) as f:
            task_data = json.load(f)

        if 'original_metadata' not in task_data:
            return 'unknown'

        metadata = task_data['original_metadata']
        return metadata.get('dataset', 'unknown')
    except Exception:
        return 'unknown'


def get_plan_metrics_for_tasks(task_scores: Dict[str, float], task_paths: Dict[str, str],
                                hard_threshold: float, medium_threshold: float) -> Dict[float, Dict[str, Any]]:
    """
    Get plan_length, plan_delta, and category distributions for tasks grouped by their exact average score.
    Small bins (n < 100) are grouped with the nearest larger bin (n >= 100).

    Args:
        task_scores: Dictionary mapping task_id to average score
        task_paths: Dictionary mapping task_id to original folder path
        hard_threshold: Threshold for hard category (for color assignment)
        medium_threshold: Threshold for medium category (for color assignment)

    Returns:
        Dictionary with structure:
        {
            score_value: {
                'plan_length': [...],
                'plan_delta': [...],
                'danger_group': [...],
                'entity_in_danger': [...],
                'data_source': [...],
                'is_grouped': bool,  # True if this bin contains merged smaller bins
                'group_min_score': float  # Minimum score in the group (for labeling)
                'group_max_score': float  # Maximum score in the group (for labeling)
                'color_score': float  # Score used for color assignment (minimum in group)
            },
            ...
        }
    """
    metrics = {}

    # First pass: collect all metrics without grouping
    for task_id, score in task_scores.items():
        if task_id not in task_paths:
            continue

        task_dir = Path(task_paths[task_id])

        # Calculate metrics
        unsafe_len = calculate_plan_length(task_dir, 'unsafe')
        safe_len = calculate_plan_length(task_dir, 'safe')
        plan_delta = calculate_plan_delta(task_dir)
        danger_group = calculate_danger_group(task_dir)
        entity_in_danger = calculate_entity_in_danger(task_dir)
        data_source = calculate_data_source(task_dir)

        # Calculate average plan length (average of unsafe and safe, rounded to integer)
        if unsafe_len > 0 and safe_len > 0:
            avg_plan_length = round((unsafe_len + safe_len) / 2.0)
        elif unsafe_len > 0:
            avg_plan_length = unsafe_len
        elif safe_len > 0:
            avg_plan_length = safe_len
        else:
            avg_plan_length = 0

        # Only add if we have valid plan length
        if avg_plan_length > 0:
            if score not in metrics:
                metrics[score] = {
                    'plan_length': [],
                    'plan_delta': [],
                    'danger_group': [],
                    'entity_in_danger': [],
                    'data_source': [],
                    'color_score': score,  # Store original score for color assignment
                    'is_grouped': False,
                    'group_min_score': score,
                    'group_max_score': score
                }

            metrics[score]['plan_length'].append(avg_plan_length)
            metrics[score]['plan_delta'].append(plan_delta)
            metrics[score]['danger_group'].append(danger_group)
            metrics[score]['entity_in_danger'].append(entity_in_danger)
            metrics[score]['data_source'].append(data_source)

    # Second pass: group small bins (n < 100) with nearest larger bin (n >= 100)
    sorted_scores = sorted(metrics.keys())
    bins_to_remove = set()

    for score in sorted_scores:
        # Count tasks in this bin (using plan_length as proxy for total count)
        bin_size = len(metrics[score]['plan_length'])

        # Skip if bin is already large enough or already marked for removal
        if bin_size >= 100 or score in bins_to_remove:
            continue

        # Find nearest larger bin (n >= 100)
        nearest_large_bin = None
        min_distance = float('inf')

        for other_score in sorted_scores:
            if other_score == score or other_score in bins_to_remove:
                continue

            other_bin_size = len(metrics[other_score]['plan_length'])
            if other_bin_size >= 100:
                distance = abs(other_score - score)
                if distance < min_distance:
                    min_distance = distance
                    nearest_large_bin = other_score

        # If found, merge into the larger bin
        if nearest_large_bin is not None:
            # Merge all data
            metrics[nearest_large_bin]['plan_length'].extend(metrics[score]['plan_length'])
            metrics[nearest_large_bin]['plan_delta'].extend(metrics[score]['plan_delta'])
            metrics[nearest_large_bin]['danger_group'].extend(metrics[score]['danger_group'])
            metrics[nearest_large_bin]['entity_in_danger'].extend(metrics[score]['entity_in_danger'])
            metrics[nearest_large_bin]['data_source'].extend(metrics[score]['data_source'])

            # Update grouping metadata
            metrics[nearest_large_bin]['is_grouped'] = True
            metrics[nearest_large_bin]['group_min_score'] = min(
                metrics[nearest_large_bin]['group_min_score'],
                metrics[score]['group_min_score']
            )
            metrics[nearest_large_bin]['group_max_score'] = max(
                metrics[nearest_large_bin]['group_max_score'],
                metrics[score]['group_max_score']
            )
            # Use minimum score for color (hardest)
            if score < metrics[nearest_large_bin]['color_score']:
                metrics[nearest_large_bin]['color_score'] = score

            # Mark for removal
            bins_to_remove.add(score)

    # Remove merged bins
    for score in bins_to_remove:
        del metrics[score]

    return metrics


def get_plan_metrics_for_tasks_by_rate(task_rates: Dict[str, float], task_paths: Dict[str, str],
                                        hard_threshold: float, medium_threshold: float) -> Dict[float, Dict[str, Any]]:
    """
    Get plan_length, plan_delta, and category distributions for tasks grouped by their rate value.
    Similar to get_plan_metrics_for_tasks but for rates (0.0 to 1.0).
    Small bins (n < 100) are grouped with the nearest larger bin (n >= 100).

    Args:
        task_rates: Dictionary mapping task_id to rate (0.0 to 1.0)
        task_paths: Dictionary mapping task_id to original folder path
        hard_threshold: Threshold for hard category (for color assignment)
        medium_threshold: Threshold for medium category (for color assignment)

    Returns:
        Dictionary with structure:
        {
            rate_value: {
                'plan_length': [...],
                'plan_delta': [...],
                'danger_group': [...],
                'entity_in_danger': [...],
                'data_source': [...],
                'is_grouped': bool,  # True if this bin contains merged smaller bins
                'group_min_rate': float  # Minimum rate in the group (for labeling)
                'group_max_rate': float  # Maximum rate in the group (for labeling)
                'color_rate': float  # Rate used for color assignment (minimum in group)
            },
            ...
        }
    """
    metrics = {}

    # First pass: collect all metrics without grouping
    for task_id, rate in task_rates.items():
        if task_id not in task_paths:
            continue

        task_dir = Path(task_paths[task_id])

        # Calculate metrics
        unsafe_len = calculate_plan_length(task_dir, 'unsafe')
        safe_len = calculate_plan_length(task_dir, 'safe')
        plan_delta = calculate_plan_delta(task_dir)
        danger_group = calculate_danger_group(task_dir)
        entity_in_danger = calculate_entity_in_danger(task_dir)
        data_source = calculate_data_source(task_dir)

        # Calculate average plan length (average of unsafe and safe, rounded to integer)
        if unsafe_len > 0 and safe_len > 0:
            avg_plan_length = round((unsafe_len + safe_len) / 2.0)
        elif unsafe_len > 0:
            avg_plan_length = unsafe_len
        elif safe_len > 0:
            avg_plan_length = safe_len
        else:
            avg_plan_length = 0

        # Only add if we have valid plan length
        if avg_plan_length > 0:
            if rate not in metrics:
                metrics[rate] = {
                    'plan_length': [],
                    'plan_delta': [],
                    'danger_group': [],
                    'entity_in_danger': [],
                    'data_source': [],
                    'color_rate': rate,  # Store original rate for color assignment
                    'is_grouped': False,
                    'group_min_rate': rate,
                    'group_max_rate': rate
                }

            metrics[rate]['plan_length'].append(avg_plan_length)
            metrics[rate]['plan_delta'].append(plan_delta)
            metrics[rate]['danger_group'].append(danger_group)
            metrics[rate]['entity_in_danger'].append(entity_in_danger)
            metrics[rate]['data_source'].append(data_source)

    # Second pass: group small bins (n < 100) with nearest larger bin (n >= 100)
    sorted_rates = sorted(metrics.keys())
    bins_to_remove = set()

    for rate in sorted_rates:
        # Count tasks in this bin (using plan_length as proxy for total count)
        bin_size = len(metrics[rate]['plan_length'])

        # Skip if bin is already large enough or already marked for removal
        if bin_size >= 100 or rate in bins_to_remove:
            continue

        # Find nearest larger bin (n >= 100)
        nearest_large_bin = None
        min_distance = float('inf')

        for other_rate in sorted_rates:
            if other_rate == rate or other_rate in bins_to_remove:
                continue

            other_bin_size = len(metrics[other_rate]['plan_length'])
            if other_bin_size >= 100:
                distance = abs(other_rate - rate)
                if distance < min_distance:
                    min_distance = distance
                    nearest_large_bin = other_rate

        # If found, merge into the larger bin
        if nearest_large_bin is not None:
            # Merge all data
            metrics[nearest_large_bin]['plan_length'].extend(metrics[rate]['plan_length'])
            metrics[nearest_large_bin]['plan_delta'].extend(metrics[rate]['plan_delta'])
            metrics[nearest_large_bin]['danger_group'].extend(metrics[rate]['danger_group'])
            metrics[nearest_large_bin]['entity_in_danger'].extend(metrics[rate]['entity_in_danger'])
            metrics[nearest_large_bin]['data_source'].extend(metrics[rate]['data_source'])

            # Update grouping metadata
            metrics[nearest_large_bin]['is_grouped'] = True
            metrics[nearest_large_bin]['group_min_rate'] = min(
                metrics[nearest_large_bin]['group_min_rate'],
                metrics[rate]['group_min_rate']
            )
            metrics[nearest_large_bin]['group_max_rate'] = max(
                metrics[nearest_large_bin]['group_max_rate'],
                metrics[rate]['group_max_rate']
            )
            # Use minimum rate for color (hardest)
            if rate < metrics[nearest_large_bin]['color_rate']:
                metrics[nearest_large_bin]['color_rate'] = rate

            # Mark for removal
            bins_to_remove.add(rate)

    # Remove merged bins
    for rate in bins_to_remove:
        del metrics[rate]

    return metrics


def rate_to_difficulty(rate: float) -> float:
    """
    Convert rate to difficulty.
    rate=0 means difficulty=1 (hardest)
    rate=1 means difficulty=0 (easiest)

    Args:
        rate: Rate value (0 to 1)

    Returns:
        Difficulty value (0 to 1, where 1 is hardest)
    """
    # Linear transformation: difficulty = 1 - rate
    return 1.0 - rate


def get_rate_color(rate: float, min_rate: float, max_rate: float, colormap_name: str = 'RdYlBu_r') -> tuple:
    """
    Get color for a rate value using a continuous colormap based on difficulty.

    Args:
        rate: Rate value (0 to 1)
        min_rate: Minimum rate value in the data
        max_rate: Maximum rate value in the data
        colormap_name: Name of matplotlib colormap (default: 'RdYlBu_r' for red-yellow-blue reversed)

    Returns:
        RGB tuple (normalized 0-1)
    """
    import matplotlib

    # Convert rate to difficulty (rate=0 -> difficulty=1, rate=1 -> difficulty=0)
    difficulty = rate_to_difficulty(rate)

    # For colormap, we want difficulty=1 (hardest) to map to one end and difficulty=0 (easiest) to the other
    normalized = difficulty

    # Get colormap using new API (matplotlib 3.7+)
    try:
        # New API: matplotlib.colormaps[name]
        cmap = matplotlib.colormaps[colormap_name]
    except (AttributeError, KeyError):
        # Fallback for older matplotlib versions
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap_name)

    # Get color (returns RGBA, we only need RGB)
    color = cmap(normalized)
    return color[:3]  # Return RGB tuple


def score_to_difficulty(score: float) -> float:
    """
    Convert score to difficulty.
    score=0 means difficulty=1 (hardest)
    score=2 means difficulty=0 (easiest)

    Args:
        score: Average score value (0 to 2)

    Returns:
        Difficulty value (0 to 1, where 1 is hardest)
    """
    # Linear transformation: difficulty = (2 - score) / 2
    return (2.0 - score) / 2.0


def get_score_color(score: float, min_score: float, max_score: float, colormap_name: str = 'RdYlBu_r') -> tuple:
    """
    Get color for a score value using a continuous colormap based on difficulty.

    Args:
        score: Average score value
        min_score: Minimum score value in the data
        max_score: Maximum score value in the data
        colormap_name: Name of matplotlib colormap (default: 'RdYlBu_r' for red-yellow-blue reversed)
                      Other options: 'viridis', 'plasma', 'coolwarm', 'YlOrRd', 'Blues', etc.

    Returns:
        RGB tuple (normalized 0-1)
    """
    import matplotlib

    # Convert score to difficulty (score=0 -> difficulty=1, score=2 -> difficulty=0)
    difficulty = score_to_difficulty(score)

    # For colormap, we want difficulty=1 (hardest) to map to one end and difficulty=0 (easiest) to the other
    # Since difficulty is already in [0, 1] range, we can use it directly
    # For coolwarm: 0 (easy) -> blue, 1 (hard) -> red
    normalized = difficulty

    # Get colormap using new API (matplotlib 3.7+)
    try:
        # New API: matplotlib.colormaps[name]
        cmap = matplotlib.colormaps[colormap_name]
    except (AttributeError, KeyError):
        # Fallback for older matplotlib versions
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap_name)

    # Get color (returns RGBA, we only need RGB)
    color = cmap(normalized)
    return color[:3]  # Return RGB tuple


def create_plan_metrics_plot(metrics: Dict[float, Dict[str, Any]], output_path: Path,
                              hard_threshold: float, medium_threshold: float):
    """
    Create compact violin plots with center of mass trend lines.
    X-axis: Rate values (0.0, 0.14, 0.29, etc.)
    Y-axis: Metric values (plan length, plan delta, categories)
    """
    # Publication-quality settings
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'figure.dpi': 300
    })

    # Create compact figure with 1 row x 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.patch.set_facecolor('white')

    sorted_rates = sorted(metrics.keys())

    # Professional deeper color scheme for difficulty transition (easy -> hard)
    # Using deeper ColorBrewer-inspired colors for publication quality
    num_rates = len(sorted_rates)
    if num_rates <= 3:
        color_palette = ['#08519c', '#d4a847', '#8b0000'][:num_rates]
    elif num_rates <= 5:
        color_palette = ['#08519c', '#2171b5', '#d4a847', '#d6604d', '#8b0000'][:num_rates]
    elif num_rates <= 7:
        color_palette = ['#08519c', '#2171b5', '#4292c6', '#d4a847', '#d6604d', '#a50f15', '#67000d'][:num_rates]
    else:
        color_palette = ['#08519c', '#2171b5', '#4292c6', '#6baed6', '#d4a847', '#d6604d', '#a50f15', '#67000d']
        while len(color_palette) < num_rates:
            color_palette.append('#67000d')
    colors = color_palette[:num_rates]

    # Plot 1: Plan Length violin plot with center of mass line
    plan_length_data = []
    for rate in sorted_rates:
        for length in metrics[rate]['plan_length']:
            plan_length_data.append({'Rate': rate, 'Plan Length': length})

    if plan_length_data:
        df_length = pd.DataFrame(plan_length_data)
        parts = axes[0].violinplot([df_length[df_length['Rate'] == r]['Plan Length'].values
                                    for r in sorted_rates],
                                   positions=range(len(sorted_rates)),
                                   widths=0.7, showmeans=False, showmedians=False, showextrema=False)

        # Color each violin
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)

        # Calculate and plot mean and median
        means = [df_length[df_length['Rate'] == r]['Plan Length'].mean() for r in sorted_rates]
        medians = [df_length[df_length['Rate'] == r]['Plan Length'].median() for r in sorted_rates]

        axes[0].plot(range(len(sorted_rates)), means, 'o-', color='black', linewidth=2,
                    markersize=4, label='Mean', zorder=10, markerfacecolor='white', markeredgewidth=1.5)
        axes[0].plot(range(len(sorted_rates)), medians, 'o-', color='dimgray', linewidth=1.5,
                    markersize=3, label='Median', zorder=9, markerfacecolor='white', markeredgewidth=1)

        # Add sample size annotations at top (only first one with "n=", rest just numbers)
        for i, rate in enumerate(sorted_rates):
            n = len(df_length[df_length['Rate'] == rate])
            label = f'n={n}' if i == 0 else str(n)
            axes[0].text(i, axes[0].get_ylim()[1] - (axes[0].get_ylim()[1] - axes[0].get_ylim()[0]) * 0.02,
                        label, ha='center', va='top', fontsize=5, style='italic')

        # Convert rates to difficulties (1 - rate) and reverse order
        difficulties = [1 - r for r in sorted_rates][::-1]
        axes[0].set_xticks(range(len(sorted_rates)))
        axes[0].set_xticklabels([f'{d:.2f}'.lstrip('0') if d != 0 else '00' for d in difficulties], fontsize=6)
        axes[0].set_xlabel('Difficulty', fontsize=8, labelpad=2)
        axes[0].tick_params(axis='both', pad=2)
        axes[0].set_ylabel('Plan Length', fontsize=8)
        axes[0].tick_params(labelsize=7)
        axes[0].spines['top'].set_visible(True)
        axes[0].spines['right'].set_visible(True)
        axes[0].grid(axis='y', visible=False)
        axes[0].grid(axis='x', visible=False)
        axes[0].legend(fontsize=6, frameon=False, loc='best')

    # Plot 2: Plan Delta violin plot with center of mass line
    plan_delta_data = []
    for rate in sorted_rates:
        for delta in metrics[rate]['plan_delta']:
            plan_delta_data.append({'Rate': rate, 'Plan Delta': delta})

    if plan_delta_data:
        df_delta = pd.DataFrame(plan_delta_data)
        parts = axes[1].violinplot([df_delta[df_delta['Rate'] == r]['Plan Delta'].values
                                    for r in sorted_rates],
                                   positions=range(len(sorted_rates)),
                                   widths=0.7, showmeans=False, showmedians=False, showextrema=False)

        # Color each violin
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)

        # Calculate and plot mean and median
        means = [df_delta[df_delta['Rate'] == r]['Plan Delta'].mean() for r in sorted_rates]
        medians = [df_delta[df_delta['Rate'] == r]['Plan Delta'].median() for r in sorted_rates]

        axes[1].plot(range(len(sorted_rates)), means, 'o-', color='black', linewidth=2,
                    markersize=4, label='Mean', zorder=10, markerfacecolor='white', markeredgewidth=1.5)
        axes[1].plot(range(len(sorted_rates)), medians, 'o-', color='dimgray', linewidth=1.5,
                    markersize=3, label='Median', zorder=9, markerfacecolor='white', markeredgewidth=1)

        # Add sample size annotations at top (only first one with "n=", rest just numbers)
        for i, rate in enumerate(sorted_rates):
            n = len(df_delta[df_delta['Rate'] == rate])
            label = f'n={n}' if i == 0 else str(n)
            axes[1].text(i, axes[1].get_ylim()[1] - (axes[1].get_ylim()[1] - axes[1].get_ylim()[0]) * 0.02,
                        label, ha='center', va='top', fontsize=5, style='italic')

        # Convert rates to difficulties (1 - rate) and reverse order
        difficulties = [1 - r for r in sorted_rates][::-1]
        axes[1].set_xticks(range(len(sorted_rates)))
        axes[1].set_xticklabels([f'{d:.2f}'.lstrip('0') if d != 0 else '00' for d in difficulties], fontsize=6)
        axes[1].set_xlabel('Difficulty', fontsize=8, labelpad=2)
        axes[1].tick_params(axis='both', pad=2)
        axes[1].set_ylabel('Plan Delta (Safety Effort)', fontsize=8)
        axes[1].tick_params(labelsize=7)
        axes[1].spines['top'].set_visible(True)
        axes[1].spines['right'].set_visible(True)
        axes[1].grid(axis='y', visible=False)
        axes[1].grid(axis='x', visible=False)
        axes[1].legend(fontsize=6, frameon=False, loc='best')

    # Plot 3: Categories - 3-layer stacked area plot (percentages)
    category_counts = {rate: {'Data Source': 0, 'Danger Group': 0, 'Entity': 0} for rate in sorted_rates}

    for rate in sorted_rates:
        # Count total occurrences for each category type
        category_counts[rate]['Data Source'] = len(metrics[rate].get('data_source', []))
        category_counts[rate]['Danger Group'] = len(metrics[rate].get('danger_group', []))
        category_counts[rate]['Entity'] = len(metrics[rate].get('entity_in_danger', []))

    # Calculate percentages
    percentages = {rate: {} for rate in sorted_rates}
    for rate in sorted_rates:
        total = sum(category_counts[rate].values())
        if total > 0:
            percentages[rate]['Data Source'] = (category_counts[rate]['Data Source'] / total) * 100
            percentages[rate]['Danger Group'] = (category_counts[rate]['Danger Group'] / total) * 100
            percentages[rate]['Entity'] = (category_counts[rate]['Entity'] / total) * 100
        else:
            percentages[rate] = {'Data Source': 0, 'Danger Group': 0, 'Entity': 0}

    # Prepare data for stacked area plot
    x = range(len(sorted_rates))
    ds_pct = [percentages[r]['Data Source'] for r in sorted_rates]
    dg_pct = [percentages[r]['Danger Group'] for r in sorted_rates]
    e_pct = [percentages[r]['Entity'] for r in sorted_rates]

    # Create stacked area plot
    axes[2].fill_between(x, 0, ds_pct, label='Data Source', color='#3498DB')
    axes[2].fill_between(x, ds_pct, [ds + dg for ds, dg in zip(ds_pct, dg_pct)],
                        label='Danger Group', color='#E74C3C')
    axes[2].fill_between(x, [ds + dg for ds, dg in zip(ds_pct, dg_pct)],
                        [ds + dg + e for ds, dg, e in zip(ds_pct, dg_pct, e_pct)],
                        label='Entity', color='#2ECC71')

    # Add lines at boundaries
    axes[2].plot(x, ds_pct, color='black', linewidth=0.5)
    axes[2].plot(x, [ds + dg for ds, dg in zip(ds_pct, dg_pct)], color='black', linewidth=0.5)

    # Convert rates to difficulties (1 - rate) and reverse order
    difficulties = [1 - r for r in sorted_rates][::-1]
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f'{d:.2f}'.lstrip('0') if d != 0 else '00' for d in difficulties], fontsize=6)
    axes[2].set_xlabel('Difficulty', fontsize=8, labelpad=2)
    axes[2].set_ylabel('Category Composition (%)', fontsize=8, labelpad=2)
    axes[2].tick_params(axis='both', pad=2)
    axes[2].set_ylim(0, 100)
    axes[2].legend(fontsize=6, frameon=False, loc='best')
    axes[2].tick_params(labelsize=7)
    axes[2].spines['top'].set_visible(True)
    axes[2].spines['right'].set_visible(True)
    axes[2].grid(axis='y', visible=False)
    axes[2].grid(axis='x', visible=False)

    plt.tight_layout(pad=0.5)

    # Save in multiple formats
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    svg_path = output_path.with_suffix('.svg')
    plt.savefig(svg_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Violin plot saved to {output_path}, {pdf_path}, and {svg_path}")



def create_combined_plan_metrics_plot(score_metrics: Dict[float, Dict[str, Any]],
                                      feasibility_metrics: Dict[float, Dict[str, Any]],
                                      safety_metrics: Dict[float, Dict[str, Any]],
                                      output_path: Path,
                                      hard_threshold: float, medium_threshold: float,
                                      feasibility_hard_threshold: float, feasibility_medium_threshold: float,
                                      safety_hard_threshold: float, safety_medium_threshold: float,
                                      si_metrics: Dict[float, Dict[str, Any]] = None):
    """
    Create compact violin plot grid with trend lines and stacked percentage plots.
    Layout: 2 rows x 3 columns, where the 3rd column has 3 vertically stacked sub-plots
    """
    # Publication-quality settings
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'figure.dpi': 300
    })

    # Create custom grid layout using GridSpec
    # 7 rows x 9 columns, where rows 0-2 are Feasibility, row 3 is spacer, rows 4-6 are Safety
    # Columns 0-2: Plan Length, 3-5: Plan Delta, 6-8: Category breakdowns (3 stacked)
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor('white')

    gs = GridSpec(7, 9, figure=fig, hspace=0.15, wspace=0.4,
                  height_ratios=[1, 1, 1, 0.3, 1, 1, 1])  # Row 3 is spacer

    # Row 1: Feasibility (rows 0-2)
    ax_feas_length = fig.add_subplot(gs[0:3, 0:3])      # Plan Length (full height)
    ax_feas_delta = fig.add_subplot(gs[0:3, 3:6])       # Plan Delta (full height)
    ax_feas_ds = fig.add_subplot(gs[0, 6:9])            # Data Source (top)
    ax_feas_dg = fig.add_subplot(gs[1, 6:9])            # Danger Group (middle)
    ax_feas_entity = fig.add_subplot(gs[2, 6:9])        # Entity (bottom)

    # Row 2: Safety (rows 4-6, skipping row 3 spacer)
    ax_safe_length = fig.add_subplot(gs[4:7, 0:3])      # Plan Length (full height)
    ax_safe_delta = fig.add_subplot(gs[4:7, 3:6])       # Plan Delta (full height)
    ax_safe_ds = fig.add_subplot(gs[4, 6:9])            # Data Source (top)
    ax_safe_dg = fig.add_subplot(gs[5, 6:9])            # Danger Group (middle)
    ax_safe_entity = fig.add_subplot(gs[6, 6:9])        # Entity (bottom)

    # Create axes array for easier indexing
    axes = [
        [ax_feas_length, ax_feas_delta, ax_feas_ds, ax_feas_dg, ax_feas_entity],
        [ax_safe_length, ax_safe_delta, ax_safe_ds, ax_safe_dg, ax_safe_entity]
    ]

    # Professional deeper color scheme for difficulty transition
    # Row 1: Feasibility
    sorted_feas_rates = sorted(feasibility_metrics.keys())
    num_feas = len(sorted_feas_rates)
    if num_feas <= 3:
        feas_color_palette = ['#08519c', '#d4a847', '#8b0000'][:num_feas]
    elif num_feas <= 5:
        feas_color_palette = ['#08519c', '#2171b5', '#d4a847', '#d6604d', '#8b0000'][:num_feas]
    elif num_feas <= 7:
        feas_color_palette = ['#08519c', '#2171b5', '#4292c6', '#d4a847', '#d6604d', '#a50f15', '#67000d'][:num_feas]
    else:
        feas_color_palette = ['#08519c', '#2171b5', '#4292c6', '#6baed6', '#d4a847', '#d6604d', '#a50f15', '#67000d']
        while len(feas_color_palette) < num_feas:
            feas_color_palette.append('#67000d')
    # Reverse colors so lowest rate (highest difficulty) gets red
    feas_colors = feas_color_palette[:num_feas][::-1]

    # Feasibility Plan Length
    feas_length_data = []
    for rate in sorted_feas_rates:
        for length in feasibility_metrics[rate]['plan_length']:
            feas_length_data.append({'Rate': rate, 'Value': length})

    if feas_length_data:
        df = pd.DataFrame(feas_length_data)
        # Reverse the order of rates so difficulty increases left to right
        reversed_feas_rates = sorted_feas_rates[::-1]
        reversed_feas_colors = feas_colors[::-1]

        parts = axes[0][0].violinplot([df[df['Rate'] == r]['Value'].values for r in reversed_feas_rates],
                                      positions=range(len(reversed_feas_rates)), widths=0.7,
                                      showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(reversed_feas_colors[i])
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)

        # Calculate and plot mean and median
        means = [df[df['Rate'] == r]['Value'].mean() for r in reversed_feas_rates]
        medians = [df[df['Rate'] == r]['Value'].median() for r in reversed_feas_rates]

        axes[0][0].plot(range(len(reversed_feas_rates)), means, 'o-', color='black',
                       linewidth=1.5, markersize=3, zorder=10, markerfacecolor='white', markeredgewidth=1.2)
        axes[0][0].plot(range(len(reversed_feas_rates)), medians, 'o-', color='dimgray',
                       linewidth=1, markersize=2, zorder=9, markerfacecolor='white', markeredgewidth=0.8)

        # Add sample size annotations at top (only first one with "n=", rest just numbers)
        for i, rate in enumerate(reversed_feas_rates):
            n = len(df[df['Rate'] == rate])
            label = f'n={n}' if i == 0 else str(n)
            axes[0][0].text(i, axes[0][0].get_ylim()[1] - (axes[0][0].get_ylim()[1] - axes[0][0].get_ylim()[0]) * 0.02,
                           label, ha='center', va='top', fontsize=5, style='italic')

        # Convert rates to difficulties (1 - rate) for labels
        feas_difficulties = [1 - r for r in reversed_feas_rates]
        axes[0][0].set_xticks(range(len(reversed_feas_rates)))
        axes[0][0].set_xticklabels(['0' if d == 0 else '1' if d == 1 else f'{d:.2f}'.lstrip('0') for d in feas_difficulties], fontsize=6)
        apply_axis_style(axes[0][0], fontsize=6)

    # Feasibility Plan Delta
    feas_delta_data = []
    for rate in sorted_feas_rates:
        for delta in feasibility_metrics[rate]['plan_delta']:
            feas_delta_data.append({'Rate': rate, 'Value': delta})

    if feas_delta_data:
        df = pd.DataFrame(feas_delta_data)
        # Reverse the order of rates so difficulty increases left to right
        reversed_feas_rates = sorted_feas_rates[::-1]
        reversed_feas_colors = feas_colors[::-1]

        parts = axes[0][1].violinplot([df[df['Rate'] == r]['Value'].values for r in reversed_feas_rates],
                                      positions=range(len(reversed_feas_rates)), widths=0.7,
                                      showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(reversed_feas_colors[i])
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)

        # Calculate and plot mean and median
        means = [df[df['Rate'] == r]['Value'].mean() for r in reversed_feas_rates]
        medians = [df[df['Rate'] == r]['Value'].median() for r in reversed_feas_rates]

        axes[0][1].plot(range(len(reversed_feas_rates)), means, 'o-', color='black',
                       linewidth=1.5, markersize=3, zorder=10, markerfacecolor='white', markeredgewidth=1.2)
        axes[0][1].plot(range(len(reversed_feas_rates)), medians, 'o-', color='dimgray',
                       linewidth=1, markersize=2, zorder=9, markerfacecolor='white', markeredgewidth=0.8)

        # Add sample size annotations at top (only first one with "n=", rest just numbers)
        for i, rate in enumerate(reversed_feas_rates):
            n = len(df[df['Rate'] == rate])
            label = f'n={n}' if i == 0 else str(n)
            axes[0][1].text(i, axes[0][1].get_ylim()[1] - (axes[0][1].get_ylim()[1] - axes[0][1].get_ylim()[0]) * 0.02,
                           label, ha='center', va='top', fontsize=5, style='italic')

        # Convert rates to difficulties (1 - rate) for labels
        feas_difficulties = [1 - r for r in reversed_feas_rates]
        axes[0][1].set_xticks(range(len(reversed_feas_rates)))
        axes[0][1].set_xticklabels(['0' if d == 0 else '1' if d == 1 else f'{d:.2f}'.lstrip('0') for d in feas_difficulties], fontsize=6)
        apply_axis_style(axes[0][1], fontsize=6)

    # Feasibility Category Breakdowns - 3 separate stacked percentage area plots
    # Reverse the order of rates so difficulty increases left to right
    reversed_feas_rates = sorted_feas_rates[::-1]
    x = range(len(reversed_feas_rates))
    feas_difficulties = [1 - r for r in reversed_feas_rates]

    # 1. Data Source breakdown (axes[0, 2])
    ds_counts_by_rate = {}
    for rate in reversed_feas_rates:
        counter = Counter(feasibility_metrics[rate].get('data_source', []))
        total = sum(counter.values())
        ds_counts_by_rate[rate] = {k: (v / total * 100) if total > 0 else 0 for k, v in counter.items()}

    # Get all unique data sources across all rates with custom order
    all_ds_set = set(k for rate_data in ds_counts_by_rate.values() for k in rate_data.keys())
    # Custom order: alfred, bddl, normbank, virtualhome, neiss
    ds_order = ['alfred', 'bddl', 'normbank', 'virtualhome', 'neiss']
    all_ds = [ds for ds in ds_order if ds in all_ds_set]

    # Muted contrasting colors for data sources (adjacent areas are distinct)
    ds_color_map = {
        'alfred': '#5B7FA3',        # Muted blue
        'bddl': '#C89A6B',          # Muted orange/tan
        'normbank': '#B87070',      # Muted red
        'virtualhome': '#7BA577',   # Muted green
        'neiss': '#8B7AA8'          # Muted purple
    }

    # Create stacked area chart (using proportions 0-1 instead of percentages 0-100)
    bottom = [0] * len(x)
    for ds_val in all_ds:
        values = [ds_counts_by_rate[r].get(ds_val, 0) / 100 for r in reversed_feas_rates]  # Convert to proportion
        color = ds_color_map.get(ds_val, '#7f7f7f')
        axes[0][2].fill_between(x, bottom, [b + v for b, v in zip(bottom, values)],
                               label=ds_val, color=color)
        bottom = [b + v for b, v in zip(bottom, values)]

    axes[0][2].set_xticks(x)
    axes[0][2].set_xticklabels([])  # No labels on top subplot
    axes[0][2].set_ylim(0, 1)
    axes[0][2].set_yticks([0, 0.5, 1])
    axes[0][2].set_yticklabels(['0', '.5', '1'], fontsize=6)
    # Reverse legend order to match stacking (top of stack = top of legend)
    handles, labels = axes[0][2].get_legend_handles_labels()
    legend = axes[0][2].legend(handles[::-1], labels[::-1], fontsize=5, frameon=False, loc='center left', bbox_to_anchor=(0.5, 0.5), alignment='left')
    for text in legend.get_texts():
        text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()])
    for patch in legend.get_patches():
        patch.set_edgecolor('white')
        patch.set_linewidth(0.8)
    apply_axis_style(axes[0][2], fontsize=6)
    axes[0][2].tick_params(bottom=False)  # Hide bottom ticks for first sub-sub figure

    # 2. Danger Group breakdown (axes[0][3])
    dg_counts_by_rate = {}
    for rate in reversed_feas_rates:
        counter = Counter(feasibility_metrics[rate].get('danger_group', []))
        total = sum(counter.values())
        dg_counts_by_rate[rate] = {k: (v / total * 100) if total > 0 else 0 for k, v in counter.items()}

    all_dg = sorted(set(k for rate_data in dg_counts_by_rate.values() for k in rate_data.keys()))

    # Muted contrasting colors for danger groups
    dg_color_map = {
        'physical': '#6B9FB7',      # Muted teal/cyan
        'psychosocial': '#C17A7A'   # Muted red/rose
    }

    bottom = [0] * len(x)
    for dg_val in all_dg:
        values = [dg_counts_by_rate[r].get(dg_val, 0) / 100 for r in reversed_feas_rates]  # Convert to proportion
        color = dg_color_map.get(dg_val, '#7f7f7f')
        axes[0][3].fill_between(x, bottom, [b + v for b, v in zip(bottom, values)],
                               label=dg_val, color=color)
        bottom = [b + v for b, v in zip(bottom, values)]

    axes[0][3].set_xticks(x)
    axes[0][3].set_xticklabels([])  # No labels on middle subplot
    axes[0][3].set_ylim(0, 1)
    axes[0][3].set_yticks([0, 0.5, 1])
    axes[0][3].set_yticklabels(['0', '.5', '1'], fontsize=6)
    # Reverse legend order to match stacking (top of stack = top of legend)
    handles, labels = axes[0][3].get_legend_handles_labels()
    legend = axes[0][3].legend(handles[::-1], labels[::-1], fontsize=5, frameon=False, loc='center left', bbox_to_anchor=(0.5, 0.5), alignment='left')
    for text in legend.get_texts():
        text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()])
    for patch in legend.get_patches():
        patch.set_edgecolor('white')
        patch.set_linewidth(0.8)
    apply_axis_style(axes[0][3], fontsize=6)
    axes[0][3].tick_params(bottom=False)  # Hide bottom ticks for second sub-sub figure

    # 3. Entity breakdown (axes[0][4])
    e_counts_by_rate = {}
    for rate in reversed_feas_rates:
        counter = Counter(feasibility_metrics[rate].get('entity_in_danger', []))
        total = sum(counter.values())
        e_counts_by_rate[rate] = {k: (v / total * 100) if total > 0 else 0 for k, v in counter.items()}

    all_e = sorted(set(k for rate_data in e_counts_by_rate.values() for k in rate_data.keys()))

    # Muted contrasting colors for entities
    entity_color_map = {
        'human': '#A884B3',      # Muted purple/mauve
        'others': '#D4C078',     # Muted gold
        'robot': '#9B7A61'       # Muted brown
    }

    bottom = [0] * len(x)
    for e_val in all_e:
        values = [e_counts_by_rate[r].get(e_val, 0) / 100 for r in reversed_feas_rates]
        color = entity_color_map.get(e_val, '#7f7f7f')
        axes[0][4].fill_between(x, bottom, [b + v for b, v in zip(bottom, values)],
                               label=e_val, color=color)
        bottom = [b + v for b, v in zip(bottom, values)]

    axes[0][4].set_xticks(x)
    axes[0][4].set_xticklabels(['0' if d == 0 else '1' if d == 1 else f'{d:.2f}'.lstrip('0') for d in feas_difficulties], fontsize=6)  # Show labels on bottom subplot
    axes[0][4].set_ylim(0, 1)
    axes[0][4].set_yticks([0, 0.5, 1])
    axes[0][4].set_yticklabels(['0', '.5', '1'], fontsize=6)
    handles, labels = axes[0][4].get_legend_handles_labels()
    legend = axes[0][4].legend(handles[::-1], labels[::-1], fontsize=5, frameon=False, loc='center left', bbox_to_anchor=(0.5, 0.5), alignment='left')
    for text in legend.get_texts():
        text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()])
    for patch in legend.get_patches():
        patch.set_edgecolor('white')
        patch.set_linewidth(0.8)
    apply_axis_style(axes[0][4], fontsize=6)

    # Row 2: Safety
    sorted_safe_rates = sorted(safety_metrics.keys())
    num_safe = len(sorted_safe_rates)
    if num_safe <= 3:
        safe_color_palette = ['#08519c', '#d4a847', '#8b0000'][:num_safe]
    elif num_safe <= 5:
        safe_color_palette = ['#08519c', '#2171b5', '#d4a847', '#d6604d', '#8b0000'][:num_safe]
    elif num_safe <= 7:
        safe_color_palette = ['#08519c', '#2171b5', '#4292c6', '#d4a847', '#d6604d', '#a50f15', '#67000d'][:num_safe]
    else:
        safe_color_palette = ['#08519c', '#2171b5', '#4292c6', '#6baed6', '#d4a847', '#d6604d', '#a50f15', '#67000d']
        while len(safe_color_palette) < num_safe:
            safe_color_palette.append('#67000d')
    # Reverse colors so lowest rate (highest difficulty) gets red
    safe_colors = safe_color_palette[:num_safe][::-1]

    # Safety Plan Length
    safe_length_data = []
    for rate in sorted_safe_rates:
        for length in safety_metrics[rate]['plan_length']:
            safe_length_data.append({'Rate': rate, 'Value': length})

    if safe_length_data:
        df = pd.DataFrame(safe_length_data)
        # Reverse the order of rates so difficulty increases left to right
        reversed_safe_rates = sorted_safe_rates[::-1]
        reversed_safe_colors = safe_colors[::-1]

        parts = axes[1][0].violinplot([df[df['Rate'] == r]['Value'].values for r in reversed_safe_rates],
                                      positions=range(len(reversed_safe_rates)), widths=0.7,
                                      showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(reversed_safe_colors[i])
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)

        # Calculate and plot mean and median
        means = [df[df['Rate'] == r]['Value'].mean() for r in reversed_safe_rates]
        medians = [df[df['Rate'] == r]['Value'].median() for r in reversed_safe_rates]

        axes[1][0].plot(range(len(reversed_safe_rates)), means, 'o-', color='black',
                       linewidth=1.5, markersize=3, zorder=10, markerfacecolor='white', markeredgewidth=1.2)
        axes[1][0].plot(range(len(reversed_safe_rates)), medians, 'o-', color='dimgray',
                       linewidth=1, markersize=2, zorder=9, markerfacecolor='white', markeredgewidth=0.8)

        # Add sample size annotations at top (only first one with "n=", rest just numbers)
        for i, rate in enumerate(reversed_safe_rates):
            n = len(df[df['Rate'] == rate])
            label = f'n={n}' if i == 0 else str(n)
            axes[1][0].text(i, axes[1][0].get_ylim()[1] - (axes[1][0].get_ylim()[1] - axes[1][0].get_ylim()[0]) * 0.02,
                           label, ha='center', va='top', fontsize=5, style='italic')

        # Convert rates to difficulties (1 - rate) for labels
        safe_difficulties = [1 - r for r in reversed_safe_rates]
        axes[1][0].set_xticks(range(len(reversed_safe_rates)))
        axes[1][0].set_xticklabels(['0' if d == 0 else '1' if d == 1 else f'{d:.2f}'.lstrip('0') for d in safe_difficulties], fontsize=6)
        apply_axis_style(axes[1][0], fontsize=6)

    # Safety Plan Delta
    safe_delta_data = []
    for rate in sorted_safe_rates:
        for delta in safety_metrics[rate]['plan_delta']:
            safe_delta_data.append({'Rate': rate, 'Value': delta})

    if safe_delta_data:
        df = pd.DataFrame(safe_delta_data)
        # Reverse the order of rates so difficulty increases left to right
        reversed_safe_rates = sorted_safe_rates[::-1]
        reversed_safe_colors = safe_colors[::-1]

        parts = axes[1][1].violinplot([df[df['Rate'] == r]['Value'].values for r in reversed_safe_rates],
                                      positions=range(len(reversed_safe_rates)), widths=0.7,
                                      showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(reversed_safe_colors[i])
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)

        # Calculate and plot mean and median
        means = [df[df['Rate'] == r]['Value'].mean() for r in reversed_safe_rates]
        medians = [df[df['Rate'] == r]['Value'].median() for r in reversed_safe_rates]

        axes[1][1].plot(range(len(reversed_safe_rates)), means, 'o-', color='black',
                       linewidth=1.5, markersize=3, zorder=10, markerfacecolor='white', markeredgewidth=1.2)
        axes[1][1].plot(range(len(reversed_safe_rates)), medians, 'o-', color='dimgray',
                       linewidth=1, markersize=2, zorder=9, markerfacecolor='white', markeredgewidth=0.8)

        # Add sample size annotations at top (only first one with "n=", rest just numbers)
        for i, rate in enumerate(reversed_safe_rates):
            n = len(df[df['Rate'] == rate])
            label = f'n={n}' if i == 0 else str(n)
            axes[1][1].text(i, axes[1][1].get_ylim()[1] - (axes[1][1].get_ylim()[1] - axes[1][1].get_ylim()[0]) * 0.02,
                           label, ha='center', va='top', fontsize=5, style='italic')

        # Convert rates to difficulties (1 - rate) for labels
        safe_difficulties = [1 - r for r in reversed_safe_rates]
        axes[1][1].set_xticks(range(len(reversed_safe_rates)))
        axes[1][1].set_xticklabels(['0' if d == 0 else '1' if d == 1 else f'{d:.2f}'.lstrip('0') for d in safe_difficulties], fontsize=6)
        apply_axis_style(axes[1][1], fontsize=6)

    # Safety Category Breakdowns - 3 separate stacked percentage area plots
    # Reverse the order of rates so difficulty increases left to right
    reversed_safe_rates = sorted_safe_rates[::-1]
    x = range(len(reversed_safe_rates))
    safe_difficulties = [1 - r for r in reversed_safe_rates]

    # 1. Data Source breakdown (axes[1][2])
    ds_counts_by_rate = {}
    for rate in reversed_safe_rates:
        counter = Counter(safety_metrics[rate].get('data_source', []))
        total = sum(counter.values())
        ds_counts_by_rate[rate] = {k: (v / total * 100) if total > 0 else 0 for k, v in counter.items()}

    # Use same custom order as feasibility row
    all_ds_set = set(k for rate_data in ds_counts_by_rate.values() for k in rate_data.keys())
    ds_order = ['alfred', 'bddl', 'normbank', 'virtualhome', 'neiss']
    all_ds = [ds for ds in ds_order if ds in all_ds_set]

    bottom = [0] * len(x)
    for ds_val in all_ds:
        values = [ds_counts_by_rate[r].get(ds_val, 0) / 100 for r in reversed_safe_rates]
        color = ds_color_map.get(ds_val, '#7f7f7f')
        axes[1][2].fill_between(x, bottom, [b + v for b, v in zip(bottom, values)],
                               label=ds_val, color=color)
        bottom = [b + v for b, v in zip(bottom, values)]

    axes[1][2].set_xticks(x)
    axes[1][2].set_xticklabels([])  # No labels on top subplot
    axes[1][2].set_ylim(0, 1)
    axes[1][2].set_yticks([0, 0.5, 1])
    axes[1][2].set_yticklabels(['0', '.5', '1'], fontsize=6)
    handles, labels = axes[1][2].get_legend_handles_labels()
    legend = axes[1][2].legend(handles[::-1], labels[::-1], fontsize=5, frameon=False, loc='center left', bbox_to_anchor=(0.5, 0.5), alignment='left')
    for text in legend.get_texts():
        text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()])
    for patch in legend.get_patches():
        patch.set_edgecolor('white')
        patch.set_linewidth(0.8)
    apply_axis_style(axes[1][2], fontsize=6)
    axes[1][2].tick_params(bottom=False)  # Hide bottom ticks for first sub-sub figure

    # 2. Danger Group breakdown (axes[1][3])
    dg_counts_by_rate = {}
    for rate in reversed_safe_rates:
        counter = Counter(safety_metrics[rate].get('danger_group', []))
        total = sum(counter.values())
        dg_counts_by_rate[rate] = {k: (v / total * 100) if total > 0 else 0 for k, v in counter.items()}

    all_dg = sorted(set(k for rate_data in dg_counts_by_rate.values() for k in rate_data.keys()))

    bottom = [0] * len(x)
    for dg_val in all_dg:
        values = [dg_counts_by_rate[r].get(dg_val, 0) / 100 for r in reversed_safe_rates]
        color = dg_color_map.get(dg_val, '#7f7f7f')
        axes[1][3].fill_between(x, bottom, [b + v for b, v in zip(bottom, values)],
                               label=dg_val, color=color)
        bottom = [b + v for b, v in zip(bottom, values)]

    axes[1][3].set_xticks(x)
    axes[1][3].set_xticklabels([])  # No labels on middle subplot
    axes[1][3].set_ylim(0, 1)
    axes[1][3].set_yticks([0, 0.5, 1])
    axes[1][3].set_yticklabels(['0', '.5', '1'], fontsize=6)
    handles, labels = axes[1][3].get_legend_handles_labels()
    legend = axes[1][3].legend(handles[::-1], labels[::-1], fontsize=5, frameon=False, loc='center left', bbox_to_anchor=(0.5, 0.5), alignment='left')
    for text in legend.get_texts():
        text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()])
    for patch in legend.get_patches():
        patch.set_edgecolor('white')
        patch.set_linewidth(0.8)
    apply_axis_style(axes[1][3], fontsize=6)
    axes[1][3].tick_params(bottom=False)  # Hide bottom ticks for second sub-sub figure

    # 3. Entity breakdown (axes[1][4])
    e_counts_by_rate = {}
    for rate in reversed_safe_rates:
        counter = Counter(safety_metrics[rate].get('entity_in_danger', []))
        total = sum(counter.values())
        e_counts_by_rate[rate] = {k: (v / total * 100) if total > 0 else 0 for k, v in counter.items()}

    all_e = sorted(set(k for rate_data in e_counts_by_rate.values() for k in rate_data.keys()))

    bottom = [0] * len(x)
    for e_val in all_e:
        values = [e_counts_by_rate[r].get(e_val, 0) / 100 for r in reversed_safe_rates]
        color = entity_color_map.get(e_val, '#7f7f7f')
        axes[1][4].fill_between(x, bottom, [b + v for b, v in zip(bottom, values)],
                               label=e_val, color=color)
        bottom = [b + v for b, v in zip(bottom, values)]

    axes[1][4].set_xticks(x)
    axes[1][4].set_xticklabels(['0' if d == 0 else '1' if d == 1 else f'{d:.2f}'.lstrip('0') for d in safe_difficulties], fontsize=6)  # Show labels on bottom subplot
    axes[1][4].set_ylim(0, 1)
    axes[1][4].set_yticks([0, 0.5, 1])
    axes[1][4].set_yticklabels(['0', '.5', '1'], fontsize=6)
    handles, labels = axes[1][4].get_legend_handles_labels()
    legend = axes[1][4].legend(handles[::-1], labels[::-1], fontsize=5, frameon=False, loc='center left', bbox_to_anchor=(0.5, 0.5), alignment='left')
    for text in legend.get_texts():
        text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()])
    for patch in legend.get_patches():
        patch.set_edgecolor('white')
        patch.set_linewidth(0.8)
    apply_axis_style(axes[1][4], fontsize=6)

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    svg_path = output_path.with_suffix('.svg')
    plt.savefig(svg_path, bbox_inches='tight', facecolor='white', edgecolor='none')

    # Save each of the 6 main subplots as completely separate standalone figures

    # Helper function to create standalone violin plot
    def save_violin_plot(metrics, reversed_rates, reversed_colors, difficulties,
                         metric_key, output_file):
        data = []
        for rate in metrics.keys():
            for value in metrics[rate][metric_key]:
                data.append({'Rate': rate, 'Value': value})

        if data:
            df = pd.DataFrame(data)
            standalone_fig, standalone_ax = plt.subplots(1, 1, figsize=(3, 2.4))
            standalone_fig.patch.set_facecolor('white')

            parts = standalone_ax.violinplot([df[df['Rate'] == r]['Value'].values for r in reversed_rates],
                                           positions=range(len(reversed_rates)), widths=0.7,
                                           showmeans=False, showmedians=False, showextrema=False)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(reversed_colors[i])
                pc.set_alpha(0.6)
                pc.set_edgecolor('black')
                pc.set_linewidth(0.5)

            means = [df[df['Rate'] == r]['Value'].mean() for r in reversed_rates]
            medians = [df[df['Rate'] == r]['Value'].median() for r in reversed_rates]
            standalone_ax.plot(range(len(reversed_rates)), means, 'o-', color='black',
                             linewidth=1.5, markersize=3, zorder=10, markerfacecolor='white', markeredgewidth=1.2)
            standalone_ax.plot(range(len(reversed_rates)), medians, 'o-', color='dimgray',
                             linewidth=1, markersize=2, zorder=9, markerfacecolor='white', markeredgewidth=0.8)

            for i, rate in enumerate(reversed_rates):
                n = len(df[df['Rate'] == rate])
                label = f'n={n}' if i == 0 else str(n)
                standalone_ax.text(i, standalone_ax.get_ylim()[1] - (standalone_ax.get_ylim()[1] - standalone_ax.get_ylim()[0]) * 0.02,
                                 label, ha='center', va='top', fontsize=5, style='italic')

            standalone_ax.set_xticks(range(len(reversed_rates)))
            standalone_ax.set_xticklabels(['0' if d == 0 else '1' if d == 1 else f'{d:.2f}'.lstrip('0') for d in difficulties], fontsize=6)
            apply_axis_style(standalone_ax, fontsize=6)

            standalone_fig.savefig(output_file, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(standalone_fig)
            print(f"  - Individual plot saved: {output_file}")

    # Save 4 violin plots
    save_violin_plot(feasibility_metrics, reversed_feas_rates, reversed_feas_colors, feas_difficulties,
                     'plan_length',
                     output_path.parent / "violin_feasibility_plan_length.svg")
    save_violin_plot(feasibility_metrics, reversed_feas_rates, reversed_feas_colors, feas_difficulties,
                     'plan_delta',
                     output_path.parent / "violin_feasibility_safety_effort.svg")
    save_violin_plot(safety_metrics, reversed_safe_rates, reversed_safe_colors, safe_difficulties,
                     'plan_length',
                     output_path.parent / "violin_safety_plan_length.svg")
    save_violin_plot(safety_metrics, reversed_safe_rates, reversed_safe_colors, safe_difficulties,
                     'plan_delta',
                     output_path.parent / "violin_safety_safety_effort.svg")

    # Save 2 category plots (each with 3 sub-sub-figures)
    for metrics_data, reversed_rates, difficulties, row_name in [
        (feasibility_metrics, reversed_feas_rates, feas_difficulties, 'feasibility'),
        (safety_metrics, reversed_safe_rates, safe_difficulties, 'safety')
    ]:
        standalone_fig = plt.figure(figsize=(3, 2.4))
        standalone_gs = GridSpec(3, 1, figure=standalone_fig, hspace=0.15)
        standalone_axes = [
            standalone_fig.add_subplot(standalone_gs[0, 0]),
            standalone_fig.add_subplot(standalone_gs[1, 0]),
            standalone_fig.add_subplot(standalone_gs[2, 0])
        ]

        x = range(len(reversed_rates))

        # Data Source
        ds_counts_by_rate = {}
        for rate in reversed_rates:
            counter = Counter(metrics_data[rate].get('data_source', []))
            total = sum(counter.values())
            ds_counts_by_rate[rate] = {k: (v / total * 100) if total > 0 else 0 for k, v in counter.items()}

        all_ds_set = set(k for rate_data in ds_counts_by_rate.values() for k in rate_data.keys())
        ds_order = ['alfred', 'bddl', 'normbank', 'virtualhome', 'neiss']
        all_ds = [ds for ds in ds_order if ds in all_ds_set]

        bottom = [0] * len(x)
        for ds_val in all_ds:
            values = [ds_counts_by_rate[r].get(ds_val, 0) / 100 for r in reversed_rates]
            color = ds_color_map.get(ds_val, '#7f7f7f')
            standalone_axes[0].fill_between(x, bottom, [b + v for b, v in zip(bottom, values)],
                                          label=ds_val, color=color)
            bottom = [b + v for b, v in zip(bottom, values)]

        standalone_axes[0].set_xticks(x)
        standalone_axes[0].set_xticklabels([])
        standalone_axes[0].set_ylim(0, 1)
        standalone_axes[0].set_yticks([0, 0.5, 1])
        standalone_axes[0].set_yticklabels(['0', '.5', '1'], fontsize=6)
        handles, labels = standalone_axes[0].get_legend_handles_labels()
        legend = standalone_axes[0].legend(handles[::-1], labels[::-1], fontsize=5, frameon=False,
                                         loc='center left', bbox_to_anchor=(0.5, 0.5), alignment='left')
        for text in legend.get_texts():
            text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()])
        for patch in legend.get_patches():
            patch.set_edgecolor('white')
            patch.set_linewidth(0.8)
        apply_axis_style(standalone_axes[0], fontsize=6)

        # Danger Group
        dg_counts_by_rate = {}
        for rate in reversed_rates:
            counter = Counter(metrics_data[rate].get('danger_group', []))
            total = sum(counter.values())
            dg_counts_by_rate[rate] = {k: (v / total * 100) if total > 0 else 0 for k, v in counter.items()}

        all_dg = sorted(set(k for rate_data in dg_counts_by_rate.values() for k in rate_data.keys()))

        bottom = [0] * len(x)
        for dg_val in all_dg:
            values = [dg_counts_by_rate[r].get(dg_val, 0) / 100 for r in reversed_rates]
            color = dg_color_map.get(dg_val, '#7f7f7f')
            standalone_axes[1].fill_between(x, bottom, [b + v for b, v in zip(bottom, values)],
                                          label=dg_val, color=color)
            bottom = [b + v for b, v in zip(bottom, values)]

        standalone_axes[1].set_xticks(x)
        standalone_axes[1].set_xticklabels([])
        standalone_axes[1].set_ylim(0, 1)
        standalone_axes[1].set_yticks([0, 0.5, 1])
        standalone_axes[1].set_yticklabels(['0', '.5', '1'], fontsize=6)
        handles, labels = standalone_axes[1].get_legend_handles_labels()
        legend = standalone_axes[1].legend(handles[::-1], labels[::-1], fontsize=5, frameon=False,
                                         loc='center left', bbox_to_anchor=(0.5, 0.5), alignment='left')
        for text in legend.get_texts():
            text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()])
        for patch in legend.get_patches():
            patch.set_edgecolor('white')
            patch.set_linewidth(0.8)
        apply_axis_style(standalone_axes[1], fontsize=6)

        # Entity
        e_counts_by_rate = {}
        for rate in reversed_rates:
            counter = Counter(metrics_data[rate].get('entity_in_danger', []))
            total = sum(counter.values())
            e_counts_by_rate[rate] = {k: (v / total * 100) if total > 0 else 0 for k, v in counter.items()}

        all_e = sorted(set(k for rate_data in e_counts_by_rate.values() for k in rate_data.keys()))

        bottom = [0] * len(x)
        for e_val in all_e:
            values = [e_counts_by_rate[r].get(e_val, 0) / 100 for r in reversed_rates]
            color = entity_color_map.get(e_val, '#7f7f7f')
            standalone_axes[2].fill_between(x, bottom, [b + v for b, v in zip(bottom, values)],
                                          label=e_val, color=color)
            bottom = [b + v for b, v in zip(bottom, values)]

        standalone_axes[2].set_xticks(x)
        standalone_axes[2].set_xticklabels(['0' if d == 0 else '1' if d == 1 else f'{d:.2f}'.lstrip('0') for d in difficulties], fontsize=6)
        standalone_axes[2].set_ylim(0, 1)
        standalone_axes[2].set_yticks([0, 0.5, 1])
        standalone_axes[2].set_yticklabels(['0', '.5', '1'], fontsize=6)
        handles, labels = standalone_axes[2].get_legend_handles_labels()
        legend = standalone_axes[2].legend(handles[::-1], labels[::-1], fontsize=5, frameon=False,
                                         loc='center left', bbox_to_anchor=(0.5, 0.5), alignment='left')
        for text in legend.get_texts():
            text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()])
        for patch in legend.get_patches():
            patch.set_edgecolor('white')
            patch.set_linewidth(0.8)
        apply_axis_style(standalone_axes[2], fontsize=6)

        individual_svg = output_path.parent / f"violin_{row_name}_categories.svg"
        standalone_fig.savefig(individual_svg, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(standalone_fig)
        print(f"  - Individual plot saved: {individual_svg}")

    plt.close()

    print(f"Combined violin plot saved to {output_path}")

    # Generate combined statistics file
    print("\nGenerating statistics file...")

    all_stats_lines = []

    # Helper function to generate stats for violin plots
    def get_violin_stats(metrics, reversed_rates, difficulties, metric_key, metric_name):
        lines = []
        lines.append(f"{'='*70}")
        lines.append(f"{metric_name} Statistics")
        lines.append(f"{'='*70}")
        lines.append("")

        # Collect data
        all_values = []
        stats_by_diff = {}
        for i, rate in enumerate(reversed_rates):
            diff = difficulties[i]
            values = metrics[rate][metric_key]
            n = len(values)
            if n > 0:
                mean_val = np.mean(values)
                median_val = np.median(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
            else:
                mean_val = median_val = std_val = min_val = max_val = 0
            stats_by_diff[diff] = {
                'n': n, 'mean': mean_val, 'median': median_val,
                'std': std_val, 'min': min_val, 'max': max_val
            }
            all_values.extend(values)

        # Per-difficulty stats
        lines.append("Per-Difficulty Statistics:")
        lines.append("-" * 70)
        lines.append(f"{'Difficulty':<12} {'N':<8} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<8} {'Max':<8}")
        lines.append("-" * 70)
        for diff in sorted(stats_by_diff.keys()):
            s = stats_by_diff[diff]
            diff_str = '0' if diff == 0 else '1' if diff == 1 else f'{diff:.2f}'
            lines.append(f"{diff_str:<12} {s['n']:<8} {s['mean']:<10.2f} {s['median']:<10.2f} {s['std']:<10.2f} {s['min']:<8.2f} {s['max']:<8.2f}")
        lines.append("")

        # Overall stats (computed first for reference)
        if all_values:
            overall_min = np.min(all_values)
            overall_max = np.max(all_values)

        # Change from lowest to highest difficulty
        sorted_diffs = sorted(stats_by_diff.keys())
        if len(sorted_diffs) >= 2:
            low_diff = sorted_diffs[0]
            high_diff = sorted_diffs[-1]
            low_stats = stats_by_diff[low_diff]
            high_stats = stats_by_diff[high_diff]
            low_mean = low_stats['mean']
            high_mean = high_stats['mean']
            change = high_mean - low_mean

            # Cohen's d with pooled standard deviation
            n1, s1 = low_stats['n'], low_stats['std']
            n2, s2 = high_stats['n'], high_stats['std']
            if n1 > 1 and n2 > 1 and (s1 > 0 or s2 > 0):
                pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
                cohens_d = change / pooled_std if pooled_std > 0 else 0
            else:
                cohens_d = 0

            lines.append("Change from Lowest to Highest Difficulty:")
            lines.append("-" * 70)
            lines.append(f"Difficulty {low_diff:.2f} -> {high_diff:.2f}")
            lines.append(f"  Mean: {low_mean:.2f} -> {high_mean:.2f}")
            lines.append(f"  Absolute change: {change:+.2f}")
            lines.append(f"  Cohen's d: {cohens_d:+.2f}")
        lines.append("")

        # Overall stats
        if all_values:
            lines.append("Overall Statistics:")
            lines.append("-" * 70)
            lines.append(f"  Total N: {len(all_values)}")
            lines.append(f"  Overall Mean: {np.mean(all_values):.2f}")
            lines.append(f"  Overall Median: {np.median(all_values):.2f}")
            lines.append(f"  Overall Std: {np.std(all_values):.2f}")
            lines.append(f"  Overall Range: [{overall_min:.2f}, {overall_max:.2f}]")

        return lines

    # Helper function to generate stats for category plots
    def get_category_stats(metrics, reversed_rates, difficulties, row_name):
        lines = []
        lines.append(f"{'='*70}")
        lines.append(f"{row_name.capitalize()} Category Statistics")
        lines.append(f"{'='*70}")
        lines.append("")

        # Data Source stats
        lines.append("DATA SOURCE Distribution:")
        lines.append("-" * 70)
        ds_order = ['alfred', 'bddl', 'normbank', 'virtualhome', 'neiss']
        ds_pcts_by_diff = {}
        ds_counts_by_diff = {}
        for i, rate in enumerate(reversed_rates):
            diff = difficulties[i]
            counter = Counter(metrics[rate].get('data_source', []))
            total = sum(counter.values())
            ds_pcts_by_diff[diff] = {k: (v / total * 100) if total > 0 else 0 for k, v in counter.items()}
            ds_counts_by_diff[diff] = dict(counter)

        # Header
        header = f"{'Difficulty':<12} {'N':<8}"
        for ds in ds_order:
            header += f" {ds:<12}"
        lines.append(header)
        lines.append("-" * 70)

        for diff in sorted(ds_pcts_by_diff.keys()):
            diff_str = '0' if diff == 0 else '1' if diff == 1 else f'{diff:.2f}'
            total_n = sum(ds_counts_by_diff[diff].values())
            row = f"{diff_str:<12} {total_n:<8}"
            for ds in ds_order:
                pct = ds_pcts_by_diff[diff].get(ds, 0)
                row += f" {pct:<12.1f}"
            lines.append(row)

        # Add overall counts per category
        lines.append("")
        lines.append("Overall Counts per Category:")
        overall_ds_counts = Counter()
        for diff in ds_counts_by_diff:
            for ds, count in ds_counts_by_diff[diff].items():
                overall_ds_counts[ds] += count
        for ds in ds_order:
            lines.append(f"  {ds}: {overall_ds_counts.get(ds, 0)}")

        # Change from lowest to highest
        sorted_diffs = sorted(ds_pcts_by_diff.keys())
        if len(sorted_diffs) >= 2:
            low_diff = sorted_diffs[0]
            high_diff = sorted_diffs[-1]
            lines.append("")
            lines.append(f"Change (Difficulty {low_diff:.2f} -> {high_diff:.2f}):")
            for ds in ds_order:
                low_pct = ds_pcts_by_diff[low_diff].get(ds, 0)
                high_pct = ds_pcts_by_diff[high_diff].get(ds, 0)
                change = high_pct - low_pct
                lines.append(f"  {ds}: {low_pct:.1f}% -> {high_pct:.1f}% ({change:+.1f}%)")
        lines.append("")

        # Danger Group stats
        lines.append("DANGER GROUP Distribution:")
        lines.append("-" * 70)
        dg_pcts_by_diff = {}
        dg_counts_by_diff = {}
        for i, rate in enumerate(reversed_rates):
            diff = difficulties[i]
            counter = Counter(metrics[rate].get('danger_group', []))
            total = sum(counter.values())
            dg_pcts_by_diff[diff] = {k: (v / total * 100) if total > 0 else 0 for k, v in counter.items()}
            dg_counts_by_diff[diff] = dict(counter)

        all_dg = sorted(set(k for pcts in dg_pcts_by_diff.values() for k in pcts.keys()))
        header = f"{'Difficulty':<12} {'N':<8}"
        for dg in all_dg:
            header += f" {dg:<15}"
        lines.append(header)
        lines.append("-" * 70)

        for diff in sorted(dg_pcts_by_diff.keys()):
            diff_str = '0' if diff == 0 else '1' if diff == 1 else f'{diff:.2f}'
            total_n = sum(dg_counts_by_diff[diff].values())
            row = f"{diff_str:<12} {total_n:<8}"
            for dg in all_dg:
                pct = dg_pcts_by_diff[diff].get(dg, 0)
                row += f" {pct:<15.1f}"
            lines.append(row)

        # Add overall counts per category
        lines.append("")
        lines.append("Overall Counts per Category:")
        overall_dg_counts = Counter()
        for diff in dg_counts_by_diff:
            for dg, count in dg_counts_by_diff[diff].items():
                overall_dg_counts[dg] += count
        for dg in all_dg:
            lines.append(f"  {dg}: {overall_dg_counts.get(dg, 0)}")

        if len(sorted_diffs) >= 2:
            lines.append("")
            lines.append(f"Change (Difficulty {low_diff:.2f} -> {high_diff:.2f}):")
            for dg in all_dg:
                low_pct = dg_pcts_by_diff[low_diff].get(dg, 0)
                high_pct = dg_pcts_by_diff[high_diff].get(dg, 0)
                change = high_pct - low_pct
                lines.append(f"  {dg}: {low_pct:.1f}% -> {high_pct:.1f}% ({change:+.1f}%)")
        lines.append("")

        # Entity stats
        lines.append("ENTITY Distribution:")
        lines.append("-" * 70)
        e_pcts_by_diff = {}
        e_counts_by_diff = {}
        for i, rate in enumerate(reversed_rates):
            diff = difficulties[i]
            counter = Counter(metrics[rate].get('entity_in_danger', []))
            total = sum(counter.values())
            e_pcts_by_diff[diff] = {k: (v / total * 100) if total > 0 else 0 for k, v in counter.items()}
            e_counts_by_diff[diff] = dict(counter)

        all_e = sorted(set(k for pcts in e_pcts_by_diff.values() for k in pcts.keys()))
        header = f"{'Difficulty':<12} {'N':<8}"
        for e in all_e:
            header += f" {e:<12}"
        lines.append(header)
        lines.append("-" * 70)

        for diff in sorted(e_pcts_by_diff.keys()):
            diff_str = '0' if diff == 0 else '1' if diff == 1 else f'{diff:.2f}'
            total_n = sum(e_counts_by_diff[diff].values())
            row = f"{diff_str:<12} {total_n:<8}"
            for e in all_e:
                pct = e_pcts_by_diff[diff].get(e, 0)
                row += f" {pct:<12.1f}"
            lines.append(row)

        # Add overall counts per category
        lines.append("")
        lines.append("Overall Counts per Category:")
        overall_e_counts = Counter()
        for diff in e_counts_by_diff:
            for e, count in e_counts_by_diff[diff].items():
                overall_e_counts[e] += count
        for e in all_e:
            lines.append(f"  {e}: {overall_e_counts.get(e, 0)}")

        if len(sorted_diffs) >= 2:
            lines.append("")
            lines.append(f"Change (Difficulty {low_diff:.2f} -> {high_diff:.2f}):")
            for e in all_e:
                low_pct = e_pcts_by_diff[low_diff].get(e, 0)
                high_pct = e_pcts_by_diff[high_diff].get(e, 0)
                change = high_pct - low_pct
                lines.append(f"  {e}: {low_pct:.1f}% -> {high_pct:.1f}% ({change:+.1f}%)")

        return lines

    # Set up SI variables for stats (if available)
    if si_metrics and len(si_metrics) > 0:
        sorted_si_rates = sorted(si_metrics.keys())
        reversed_si_rates = sorted_si_rates[::-1]
        si_difficulties = [1 - r for r in reversed_si_rates]
    else:
        reversed_si_rates = None
        si_difficulties = None

    # Collect all stats - grouped by metric type (Feasibility + Safety together)
    all_stats_lines.append("FACTOR ANALYSIS STATISTICS")
    all_stats_lines.append("=" * 70)
    all_stats_lines.append("")
    all_stats_lines.append("Effect sizes reported as Cohen's d (pooled std):")
    all_stats_lines.append("  |d| < 0.2: negligible, 0.2-0.5: small, 0.5-0.8: medium, > 0.8: large")
    all_stats_lines.append("")
    all_stats_lines.append("Figure Layout:")
    all_stats_lines.append("  Rows:    a = Feasibility, b = Safety, c = Safety Intention")
    all_stats_lines.append("  Columns: i = Plan Length, ii = Safety Effort, iii = Categories,")
    all_stats_lines.append("           iv = Redundant Actions, v = Redundant Objects")
    all_stats_lines.append("  Labels:  i, ii, iii, iv, v (roman numerals only)")
    all_stats_lines.append("")
    all_stats_lines.append("")

    # Plan Length section (Feasibility + Safety + SI)
    all_stats_lines.append("#" * 70)
    all_stats_lines.append("# PLAN LENGTH")
    all_stats_lines.append("#" * 70)
    all_stats_lines.append("")
    all_stats_lines.extend(get_violin_stats(feasibility_metrics, reversed_feas_rates, feas_difficulties,
                                            'plan_length', 'Feasibility - Plan Length'))
    all_stats_lines.append("")
    all_stats_lines.append("")
    all_stats_lines.extend(get_violin_stats(safety_metrics, reversed_safe_rates, safe_difficulties,
                                            'plan_length', 'Safety - Plan Length'))
    all_stats_lines.append("")
    all_stats_lines.append("")
    if si_metrics and reversed_si_rates:
        all_stats_lines.extend(get_violin_stats(si_metrics, reversed_si_rates, si_difficulties,
                                                'plan_length', 'Safety Intention - Plan Length'))
        all_stats_lines.append("")
        all_stats_lines.append("")

    # Safety Effort section (Feasibility + Safety + SI)
    all_stats_lines.append("#" * 70)
    all_stats_lines.append("# SAFETY EFFORT")
    all_stats_lines.append("#" * 70)
    all_stats_lines.append("")
    all_stats_lines.extend(get_violin_stats(feasibility_metrics, reversed_feas_rates, feas_difficulties,
                                            'plan_delta', 'Feasibility - Safety Effort'))
    all_stats_lines.append("")
    all_stats_lines.append("")
    all_stats_lines.extend(get_violin_stats(safety_metrics, reversed_safe_rates, safe_difficulties,
                                            'plan_delta', 'Safety - Safety Effort'))
    all_stats_lines.append("")
    all_stats_lines.append("")
    if si_metrics and reversed_si_rates:
        all_stats_lines.extend(get_violin_stats(si_metrics, reversed_si_rates, si_difficulties,
                                                'plan_delta', 'Safety Intention - Safety Effort'))
        all_stats_lines.append("")
        all_stats_lines.append("")

    # Categories section (Feasibility + Safety + SI)
    all_stats_lines.append("#" * 70)
    all_stats_lines.append("# CATEGORIES")
    all_stats_lines.append("#" * 70)
    all_stats_lines.append("")
    all_stats_lines.extend(get_category_stats(feasibility_metrics, reversed_feas_rates, feas_difficulties,
                                              'feasibility'))
    all_stats_lines.append("")
    all_stats_lines.append("")
    all_stats_lines.extend(get_category_stats(safety_metrics, reversed_safe_rates, safe_difficulties,
                                              'safety'))
    if si_metrics and reversed_si_rates:
        all_stats_lines.append("")
        all_stats_lines.append("")
        all_stats_lines.extend(get_category_stats(si_metrics, reversed_si_rates, si_difficulties,
                                                  'safety_intention'))

    # Write combined stats file
    stats_file = output_path.parent / "results_factor.txt"
    with open(stats_file, 'w') as f:
        f.write('\n'.join(all_stats_lines))
    print(f"  - Stats saved: {stats_file}")


def create_full_combined_plot(feasibility_metrics: Dict, safety_metrics: Dict,
                               redundancy_data_path: Path, output_path: Path,
                               si_metrics: Dict = None):
    """
    Create a combined 3x5 plot with factor_analysis (3 cols) + redundancy (2 cols).
    Row 0: Feasibility, Row 1: Safety, Row 2: Safety Intention

    Generates native matplotlib figure matching individual plot styles exactly.
    """
    from matplotlib.gridspec import GridSpec

    # Load redundancy data
    if not redundancy_data_path.exists():
        print(f"  Warning: Redundancy data not found at {redundancy_data_path}")
        return

    redundancy_df = pd.read_csv(redundancy_data_path)

    # ==================== FACTOR ANALYSIS SETUP ====================
    # Use EXACT same color palette as create_combined_plan_metrics_plot (lines 856-867)
    sorted_feas_rates = sorted(feasibility_metrics.keys())
    sorted_safe_rates = sorted(safety_metrics.keys())
    num_feas = len(sorted_feas_rates)
    num_safe = len(sorted_safe_rates)

    # Exact color logic from create_combined_plan_metrics_plot
    if num_feas <= 3:
        feas_color_palette = ['#08519c', '#d4a847', '#8b0000'][:num_feas]
    elif num_feas <= 5:
        feas_color_palette = ['#08519c', '#2171b5', '#d4a847', '#d6604d', '#8b0000'][:num_feas]
    elif num_feas <= 7:
        feas_color_palette = ['#08519c', '#2171b5', '#4292c6', '#d4a847', '#d6604d', '#a50f15', '#67000d'][:num_feas]
    else:
        feas_color_palette = ['#08519c', '#2171b5', '#4292c6', '#6baed6', '#d4a847', '#d6604d', '#a50f15', '#67000d']
        while len(feas_color_palette) < num_feas:
            feas_color_palette.append('#67000d')
    # Reverse colors so lowest rate (highest difficulty) gets red
    feas_colors = feas_color_palette[:num_feas][::-1]

    if num_safe <= 3:
        safe_color_palette = ['#08519c', '#d4a847', '#8b0000'][:num_safe]
    elif num_safe <= 5:
        safe_color_palette = ['#08519c', '#2171b5', '#d4a847', '#d6604d', '#8b0000'][:num_safe]
    elif num_safe <= 7:
        safe_color_palette = ['#08519c', '#2171b5', '#4292c6', '#d4a847', '#d6604d', '#a50f15', '#67000d'][:num_safe]
    else:
        safe_color_palette = ['#08519c', '#2171b5', '#4292c6', '#6baed6', '#d4a847', '#d6604d', '#a50f15', '#67000d']
        while len(safe_color_palette) < num_safe:
            safe_color_palette.append('#67000d')
    safe_colors = safe_color_palette[:num_safe][::-1]

    # SI metrics color palette (same logic)
    if si_metrics:
        sorted_si_rates = sorted(si_metrics.keys())
        num_si = len(sorted_si_rates)
        if num_si <= 3:
            si_color_palette = ['#08519c', '#d4a847', '#8b0000'][:num_si]
        elif num_si <= 5:
            si_color_palette = ['#08519c', '#2171b5', '#d4a847', '#d6604d', '#8b0000'][:num_si]
        elif num_si <= 7:
            si_color_palette = ['#08519c', '#2171b5', '#4292c6', '#d4a847', '#d6604d', '#a50f15', '#67000d'][:num_si]
        else:
            si_color_palette = ['#08519c', '#2171b5', '#4292c6', '#6baed6', '#d4a847', '#d6604d', '#a50f15', '#67000d']
            while len(si_color_palette) < num_si:
                si_color_palette.append('#67000d')
        si_colors = si_color_palette[:num_si][::-1]

    # Reverse rates for plotting (difficulty increases left to right)
    reversed_feas_rates = sorted_feas_rates[::-1]
    reversed_safe_rates = sorted_safe_rates[::-1]
    # Reverse colors again to match the reversed rates
    reversed_feas_colors = feas_colors[::-1]
    reversed_safe_colors = safe_colors[::-1]
    feas_difficulties = [1 - r for r in reversed_feas_rates]
    safe_difficulties = [1 - r for r in reversed_safe_rates]

    if si_metrics:
        reversed_si_rates = sorted_si_rates[::-1]
        reversed_si_colors = si_colors[::-1]
        si_difficulties = [1 - r for r in reversed_si_rates]

    # Category color maps (exact same as create_combined_plan_metrics_plot - muted colors)
    ds_color_map = {
        'alfred': '#5B7FA3',        # Muted blue
        'bddl': '#C89A6B',          # Muted orange/tan
        'normbank': '#B87070',      # Muted red
        'virtualhome': '#7BA577',   # Muted green
        'neiss': '#8B7AA8'          # Muted purple
    }
    dg_color_map = {
        'physical': '#6B9FB7',      # Muted teal/cyan
        'psychosocial': '#C17A7A'   # Muted red/rose
    }
    entity_color_map = {
        'human': '#A884B3',      # Muted purple/mauve
        'others': '#D4C078',     # Muted gold
        'robot': '#9B7A61'       # Muted brown
    }

    # ==================== REDUNDANCY SETUP ====================
    # Get all unique models (exact same as RedundancyAnalysis)
    all_models_raw = sorted(redundancy_df[redundancy_df['redundancy_type'] != 'baseline']['model_display'].unique())
    # Filter out DeepSeek-V3.2-Exp
    all_models = [m for m in all_models_raw if 'DeepSeek' not in m]
    baseline_data = redundancy_df[redundancy_df['redundancy_type'] == 'baseline']

    # Exact same color palette as RedundancyAnalysis
    # Muted color palette matching factor_analysis style
    muted_colors = ['#5B7FA3', '#C89A6B', '#7BA577', '#B87070', '#8B7AA8', '#9B7A61', '#6B9FB7', '#C17A7A']
    model_colors = {model: muted_colors[i % len(muted_colors)] for i, model in enumerate(all_models)}

    # Different marker styles to distinguish models
    marker_styles = ['o', 's', '^', 'D', 'v', 'p']
    model_markers = {model: marker_styles[i % len(marker_styles)] for i, model in enumerate(all_models)}

    # ==================== CREATE FIGURE ====================
    # Ensure SVG output is fully vectorial (no rasterization of hatches)
    plt.rcParams['svg.fonttype'] = 'none'  # Keep fonts as text, not paths
    plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts for PDF

    has_si = si_metrics is not None and len(si_metrics) > 0
    fig_height = 11.0 if has_si else 7.5
    fig = plt.figure(figsize=(17, fig_height))
    fig.patch.set_facecolor('white')
    if has_si:
        gs = GridSpec(11, 4, figure=fig, hspace=0.25, wspace=0.28,
                      height_ratios=[1, 1, 1, 0.4, 1, 1, 1, 0.4, 1, 1, 1])  # 3 rows + 2 spacers
    else:
        gs = GridSpec(7, 4, figure=fig, hspace=0.25, wspace=0.28,
                      height_ratios=[1, 1, 1, 0.4, 1, 1, 1])

    # ==================== GLOBAL STYLE SETTINGS ====================
    fs = 14  # font size for all text (tick labels, legends, annotations, axis labels)
    lp = 1  # labelpad: space between axis label and tick values
    cat_hspace = 0.15  # vertical spacing between category sub-sub figures only

    # ==================== HELPER: VIOLIN PLOT ====================
    def plot_violin(ax, metrics, reversed_rates, reversed_colors, difficulties, metric_key, show_n_prefix=True, n_x_offset=-0.45):
        """Exact same logic as save_violin_plot in create_combined_plan_metrics_plot"""
        data = []
        for rate in metrics.keys():
            for value in metrics[rate][metric_key]:
                data.append({'Rate': rate, 'Value': value})
        if not data:
            return

        df = pd.DataFrame(data)
        all_values = df['Value'].values

        # Truncate y-axis to 2nd-98th percentile to focus on main distribution
        p2, p98 = np.percentile(all_values, [2, 98])
        padding = (p98 - p2) * 0.08  # 8% padding
        if metric_key == 'plan_length':
            y_min = max(0, p2 - padding)  # Don't go below 0 for lengths
            y_max = p98 + padding
        else:  # plan_delta can be negative
            y_min = p2 - padding
            y_max = p98 + padding

        parts = ax.violinplot([df[df['Rate'] == r]['Value'].values for r in reversed_rates],
                              positions=range(len(reversed_rates)), widths=0.7,
                              showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(reversed_colors[i])
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)

        means = [df[df['Rate'] == r]['Value'].mean() for r in reversed_rates]
        ax.plot(range(len(reversed_rates)), means, 'o-', color='black',
                linewidth=1.5, markersize=6, zorder=10, markerfacecolor='white', markeredgewidth=1.5)

        # Set y-axis limits to truncated range
        ax.set_ylim(y_min, y_max)

        # Sample size annotations (rotated 90° CCW, shifted left)
        for i, rate in enumerate(reversed_rates):
            n = len(df[df['Rate'] == rate])
            if show_n_prefix:
                label = f'n={n}' if i == 0 else str(n)
            else:
                label = str(n)
            ax.text(i + n_x_offset, y_max - (y_max - y_min) * 0.05,
                    label, ha='left', va='top', fontsize=fs - 2, style='italic', rotation=90)

        ax.set_xticks(range(len(reversed_rates)))
        ax.set_xticklabels(['0' if d == 0 else '1' if d == 1 else f'{d:.2f}'.lstrip('0') for d in difficulties], fontsize=fs)

        # Set y-axis ticks based on metric type
        if metric_key == 'plan_delta':
            # Set ticks with interval 2, within the truncated range
            min_tick = int(np.ceil(y_min / 2) * 2)
            max_tick = int(np.floor(y_max / 2) * 2)
            ticks = list(range(min_tick, max_tick + 1, 2))
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(t) for t in ticks], fontsize=fs)

        apply_axis_style(ax, fontsize=fs)

    # ==================== HELPER: CATEGORY STACKED AREA ====================
    # No hatches - use solid colors only for clean vectorial SVG output
    # (hatches can be rasterized in some SVG renderers)

    def plot_category(ax, metrics, reversed_rates, category_key, color_map, category_order=None,
                      show_x=False, show_bottom_ticks=True, difficulties=None, show_legend=True):
        """Exact same logic as category plots in create_combined_plan_metrics_plot"""
        counts_by_rate = {}
        for rate in reversed_rates:
            counter = Counter(metrics[rate].get(category_key, []))
            total = sum(counter.values())
            counts_by_rate[rate] = {k: (v / total) if total > 0 else 0 for k, v in counter.items()}

        all_cats_set = set(k for rate_data in counts_by_rate.values() for k in rate_data.keys())
        if category_order:
            all_cats = [c for c in category_order if c in all_cats_set]
        else:
            all_cats = sorted(all_cats_set)

        x = range(len(reversed_rates))
        bottom = [0] * len(x)
        for cat in all_cats:
            values = [counts_by_rate[r].get(cat, 0) for r in reversed_rates]
            color = color_map.get(cat, '#7f7f7f')
            # Solid colors only - no hatches, no borders for clean vectorial output
            ax.fill_between(x, bottom, [b + v for b, v in zip(bottom, values)],
                           label=cat, color=color, edgecolor='none', linewidth=0)
            bottom = [b + v for b, v in zip(bottom, values)]

        ax.set_xticks(x)
        if show_x and difficulties:
            ax.set_xticklabels(['0' if d == 0 else '1' if d == 1 else f'{d:.2f}'.lstrip('0') for d in difficulties], fontsize=fs)
        else:
            ax.set_xticklabels([])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, len(reversed_rates) - 1)
        ax.margins(x=0)
        ax.set_yticks([0, 0.5, 1])
        # Custom y-tick labels with adjusted vertical alignment to avoid overlap
        ax.set_yticklabels(['', '.5', ''], fontsize=fs)
        # Manually add 0 and 1 labels with proper alignment
        ax.text(-0.02, 0.02, '0', transform=ax.transAxes, fontsize=fs, ha='right', va='bottom')
        ax.text(-0.02, 0.98, '1', transform=ax.transAxes, fontsize=fs, ha='right', va='top')

        # Only show legend if requested
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(handles[::-1], labels[::-1], fontsize=fs, frameon=False,
                              loc='center left', bbox_to_anchor=(0.5, 0.5), alignment='left')
            for text in legend.get_texts():
                text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()])
            for patch in legend.get_patches():
                patch.set_edgecolor('white')
                patch.set_linewidth(0.8)
        apply_axis_style(ax, fontsize=fs)
        # Hide bottom ticks for first 2 sub-sub figures (data_source, danger_group)
        if not show_bottom_ticks:
            ax.tick_params(bottom=False)

    # ==================== HELPER: REDUNDANCY PLOT ====================
    # Model name mapping for legend (full names)
    concise_names = {
        'DeepSeek-V3.2-Exp': 'DeepSeek-V3.2-Exp',
        'GPT-5-mini': 'GPT-5-mini',
        'GPT-5.1': 'GPT-5.1',
        'Llama-3.1-8B': 'Llama-3.1-8B',
        'Llama-3.3-70B': 'Llama-3.3-70B',
        'Qwen3-235B': 'Qwen3-235B',
        'claude-haiku-4-5': 'Claude-Haiku-4.5',
    }

    def plot_redundancy(ax, df, redundancy_type, metric, baseline_data, all_models, model_colors, model_markers, show_legend=False):
        """Exact same logic as _plot_redundancy_subplot in RedundancyAnalysis"""
        type_data = df[df['redundancy_type'] == redundancy_type].copy()
        levels = sorted(type_data['redundancy_level'].unique())

        # Build model_performance dict: model -> level -> {rate, std}
        model_performance = {model: {} for model in all_models}
        std_key = f'std_{metric}'

        # Add baseline (level 0) and collect baseline feasibility rates for sorting
        baseline_feas_rates = {}
        for model in all_models:
            model_baseline = baseline_data[baseline_data['model_display'] == model]
            if len(model_baseline) > 0:
                row = model_baseline.iloc[0]
                rate = row[metric]  # Keep as proportion (0-1)
                std = row[std_key] if std_key in row.index else 0
                model_performance[model][0] = {'rate': rate, 'std': std}
                # Store baseline feasibility rate for sorting
                baseline_feas_rates[model] = row['feasible'] if 'feasible' in row.index else rate

        # Add experiment levels
        for level in levels:
            level_data = type_data[type_data['redundancy_level'] == level]
            for model in all_models:
                model_data = level_data[level_data['model_display'] == model]
                if len(model_data) > 0:
                    row = model_data.iloc[0]
                    rate = row[metric]  # Keep as proportion (0-1)
                    std = row[std_key] if std_key in row.index else 0
                    model_performance[model][level] = {'rate': rate, 'std': std}

        def transform_x(level):
            return 0 if level == 0 else np.log2(level)

        # Sort models by baseline feasibility rate (descending) for consistent legend order
        sorted_models = sorted(all_models, key=lambda m: baseline_feas_rates.get(m, 0), reverse=True)

        # Plot each model in sorted order
        for model in sorted_models:
            mdata = model_performance[model]
            if not mdata:
                continue

            x_raw = sorted(mdata.keys())
            x_vals = [transform_x(l) for l in x_raw]
            rates = [mdata[l]['rate'] for l in x_raw]
            stds = [mdata[l]['std'] for l in x_raw]

            x_arr = np.array(x_vals)
            rates_arr = np.array(rates)
            stds_arr = np.array(stds)

            # Shaded std region (exact same as _plot_redundancy_subplot)
            has_std = any(s > 0 for s in stds)
            if has_std:
                upper = rates_arr + stds_arr
                lower = rates_arr - stds_arr
                ax.fill_between(x_arr, lower, upper, color=model_colors[model],
                               alpha=0.2, linewidth=0, zorder=1)

            # Line and markers (with distinct marker styles)
            label = concise_names.get(model, model)
            marker = model_markers.get(model, 'o')
            ax.plot(x_vals, rates, linewidth=1.5, markersize=7, color=model_colors[model],
                   marker=marker, markerfacecolor='white', markeredgewidth=1.5,
                   markeredgecolor=model_colors[model], zorder=10, label=label)

        # X-axis setup
        all_levels_raw = sorted([l for l in levels if l > 0])
        has_baseline = any(0 in mdata.keys() for mdata in model_performance.values())
        all_levels_plot = [transform_x(0)] + [transform_x(l) for l in all_levels_raw] if has_baseline else [transform_x(l) for l in all_levels_raw]
        all_levels_labels = ['0'] + [str(l) for l in all_levels_raw] if has_baseline else [str(l) for l in all_levels_raw]

        ax.set_xscale('linear')
        ax.set_xticks(all_levels_plot)
        ax.set_xticklabels(all_levels_labels, fontsize=fs)

        if all_levels_raw:
            max_x = transform_x(max(all_levels_raw))
            ax.set_xlim(-0.3, max_x + 0.3)
        else:
            ax.set_xlim(-0.3, 10)

        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        # Custom y-tick labels with adjusted vertical alignment to avoid overlap
        ax.set_yticklabels(['', '.2', '.4', '.6', '.8', ''], fontsize=fs)
        # Manually add 0 and 1 labels with proper alignment
        ax.text(-0.02, 0.02, '0', transform=ax.transAxes, fontsize=fs, ha='right', va='bottom')
        ax.text(-0.02, 0.98, '1', transform=ax.transAxes, fontsize=fs, ha='right', va='top')
        apply_axis_style(ax, fontsize=fs)

        # Add legend if requested (positioned to the left, lower)
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(handles, labels, fontsize=fs, frameon=False,
                              loc='center left', bbox_to_anchor=(0.02, 0.35))
            for text in legend.get_texts():
                text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()])

    # ==================== ROW 0: FEASIBILITY ====================
    ax_feas_pl = fig.add_subplot(gs[0:3, 0])
    plot_violin(ax_feas_pl, feasibility_metrics, reversed_feas_rates, reversed_feas_colors, feas_difficulties, 'plan_length')

    ax_feas_se = fig.add_subplot(gs[0:3, 1])
    plot_violin(ax_feas_se, feasibility_metrics, reversed_feas_rates, reversed_feas_colors, feas_difficulties, 'plan_delta')

    # Nested GridSpec for category sub-sub figures (independent hspace)
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs_feas_cat = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0:3, 2], hspace=cat_hspace)
    ax_feas_ds = fig.add_subplot(gs_feas_cat[0])
    ax_feas_dg = fig.add_subplot(gs_feas_cat[1])
    ax_feas_e = fig.add_subplot(gs_feas_cat[2])
    plot_category(ax_feas_ds, feasibility_metrics, reversed_feas_rates, 'data_source', ds_color_map,
                  category_order=['alfred', 'bddl', 'normbank', 'virtualhome', 'neiss'], show_bottom_ticks=False, show_legend=False)
    plot_category(ax_feas_dg, feasibility_metrics, reversed_feas_rates, 'danger_group', dg_color_map, show_bottom_ticks=False, show_legend=False)
    plot_category(ax_feas_e, feasibility_metrics, reversed_feas_rates, 'entity_in_danger', entity_color_map,
                  show_x=True, difficulties=feas_difficulties, show_legend=False)

    ax_feas_act = fig.add_subplot(gs[0:3, 3])
    plot_redundancy(ax_feas_act, redundancy_df, 'actions', 'feasible', baseline_data, all_models, model_colors, model_markers, show_legend=False)

    # ==================== ROW 1: SAFETY ====================
    ax_safe_pl = fig.add_subplot(gs[4:7, 0])
    plot_violin(ax_safe_pl, safety_metrics, reversed_safe_rates, reversed_safe_colors, safe_difficulties, 'plan_length', show_n_prefix=True, n_x_offset=-0.60)

    ax_safe_se = fig.add_subplot(gs[4:7, 1])
    plot_violin(ax_safe_se, safety_metrics, reversed_safe_rates, reversed_safe_colors, safe_difficulties, 'plan_delta', show_n_prefix=True, n_x_offset=-0.60)

    # Nested GridSpec for safety category sub-sub figures
    gs_safe_cat = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[4:7, 2], hspace=cat_hspace)
    ax_safe_ds = fig.add_subplot(gs_safe_cat[0])
    ax_safe_dg = fig.add_subplot(gs_safe_cat[1])
    ax_safe_e = fig.add_subplot(gs_safe_cat[2])
    plot_category(ax_safe_ds, safety_metrics, reversed_safe_rates, 'data_source', ds_color_map,
                  category_order=['alfred', 'bddl', 'normbank', 'virtualhome', 'neiss'], show_bottom_ticks=False, show_legend=False)
    plot_category(ax_safe_dg, safety_metrics, reversed_safe_rates, 'danger_group', dg_color_map, show_bottom_ticks=False, show_legend=False)
    plot_category(ax_safe_e, safety_metrics, reversed_safe_rates, 'entity_in_danger', entity_color_map,
                  show_x=True, difficulties=safe_difficulties, show_legend=False)

    ax_safe_act = fig.add_subplot(gs[4:7, 3])
    plot_redundancy(ax_safe_act, redundancy_df, 'actions', 'safe', baseline_data, all_models, model_colors, model_markers, show_legend=False)

    # ==================== ROW 2: SAFETY INTENTION ====================
    if has_si:
        ax_si_pl = fig.add_subplot(gs[8:11, 0])
        plot_violin(ax_si_pl, si_metrics, reversed_si_rates, reversed_si_colors, si_difficulties, 'plan_length', show_n_prefix=True, n_x_offset=-0.60)

        ax_si_se = fig.add_subplot(gs[8:11, 1])
        plot_violin(ax_si_se, si_metrics, reversed_si_rates, reversed_si_colors, si_difficulties, 'plan_delta', show_n_prefix=True, n_x_offset=-0.60)

        # Nested GridSpec for SI category sub-sub figures
        gs_si_cat = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[8:11, 2], hspace=cat_hspace)
        ax_si_ds = fig.add_subplot(gs_si_cat[0])
        ax_si_dg = fig.add_subplot(gs_si_cat[1])
        ax_si_e = fig.add_subplot(gs_si_cat[2])
        plot_category(ax_si_ds, si_metrics, reversed_si_rates, 'data_source', ds_color_map,
                      category_order=['alfred', 'bddl', 'normbank', 'virtualhome', 'neiss'], show_bottom_ticks=False, show_legend=False)
        plot_category(ax_si_dg, si_metrics, reversed_si_rates, 'danger_group', dg_color_map, show_bottom_ticks=False, show_legend=False)
        plot_category(ax_si_e, si_metrics, reversed_si_rates, 'entity_in_danger', entity_color_map,
                      show_x=True, difficulties=si_difficulties, show_legend=False)

        ax_si_act = fig.add_subplot(gs[8:11, 3])
        plot_redundancy(ax_si_act, redundancy_df, 'actions', 'safety_intention', baseline_data, all_models, model_colors, model_markers, show_legend=False)

    # ==================== AXIS LABELS ====================
    # X-axis labels - Upper row (Feasibility)
    ax_feas_pl.set_xlabel('Feasibility Difficulty', fontsize=fs, labelpad=lp)
    ax_feas_se.set_xlabel('Feasibility Difficulty', fontsize=fs, labelpad=lp)
    ax_feas_e.set_xlabel('Feasibility Difficulty', fontsize=fs, labelpad=lp)
    ax_feas_act.set_xlabel('Redundant Actions', fontsize=fs, labelpad=lp)

    # X-axis labels - Middle row (Safety)
    ax_safe_pl.set_xlabel('Safety Difficulty', fontsize=fs, labelpad=lp)
    ax_safe_se.set_xlabel('Safety Difficulty', fontsize=fs, labelpad=lp)
    ax_safe_e.set_xlabel('Safety Difficulty', fontsize=fs, labelpad=lp)
    ax_safe_act.set_xlabel('Redundant Actions', fontsize=fs, labelpad=lp)

    # Y-axis labels - Upper row (Feasibility)
    ax_feas_pl.set_ylabel('Plan Length', fontsize=fs, labelpad=lp)
    ax_feas_se.set_ylabel('Safety Effort', fontsize=fs, labelpad=lp)
    ax_feas_ds.set_ylabel('DS', fontsize=fs, labelpad=lp)   # Data Source
    ax_feas_dg.set_ylabel('DG', fontsize=fs, labelpad=lp)   # Danger Group
    ax_feas_e.set_ylabel('EiD', fontsize=fs, labelpad=lp)   # Entity in Danger
    ax_feas_act.set_ylabel('Feasibility Rate', fontsize=fs, labelpad=lp)

    # Y-axis labels - Middle row (Safety)
    ax_safe_pl.set_ylabel('Plan Length', fontsize=fs, labelpad=lp)
    ax_safe_se.set_ylabel('Safety Effort', fontsize=fs, labelpad=lp)
    ax_safe_ds.set_ylabel('DS', fontsize=fs, labelpad=lp)   # Data Source
    ax_safe_dg.set_ylabel('DG', fontsize=fs, labelpad=lp)   # Danger Group
    ax_safe_e.set_ylabel('EiD', fontsize=fs, labelpad=lp)   # Entity in Danger
    ax_safe_act.set_ylabel('Safety Rate', fontsize=fs, labelpad=lp)

    # X/Y-axis labels - Bottom row (Safety Intention)
    if has_si:
        ax_si_pl.set_xlabel('Safety Intention Difficulty', fontsize=fs, labelpad=lp)
        ax_si_se.set_xlabel('Safety Intention Difficulty', fontsize=fs, labelpad=lp)
        ax_si_e.set_xlabel('Safety Intention Difficulty', fontsize=fs, labelpad=lp)
        ax_si_act.set_xlabel('Redundant Actions', fontsize=fs, labelpad=lp)

        ax_si_pl.set_ylabel('Plan Length', fontsize=fs, labelpad=lp)
        ax_si_se.set_ylabel('Safety Effort', fontsize=fs, labelpad=lp)
        ax_si_ds.set_ylabel('DS', fontsize=fs, labelpad=lp)
        ax_si_dg.set_ylabel('DG', fontsize=fs, labelpad=lp)
        ax_si_e.set_ylabel('EiD', fontsize=fs, labelpad=lp)
        ax_si_act.set_ylabel('Safety Intention Rate', fontsize=fs, labelpad=lp)

    # ==================== PANEL LABELS (Nature/Science style) ====================
    # Each column uses i, ii, iii for its three rows (Feasibility, Safety, SI)
    panel_label_fs = 20  # font size for panel labels
    label_x, label_y = -0.08, 0.99  # top-left position

    # Column 1: Plan Length (i, ii, iii)
    ax_feas_pl.text(label_x, label_y, 'i', transform=ax_feas_pl.transAxes,
                    fontsize=panel_label_fs, fontweight='bold', va='bottom', ha='left')
    ax_safe_pl.text(label_x, label_y, 'ii', transform=ax_safe_pl.transAxes,
                    fontsize=panel_label_fs, fontweight='bold', va='bottom', ha='left')
    if has_si:
        ax_si_pl.text(label_x, label_y, 'iii', transform=ax_si_pl.transAxes,
                      fontsize=panel_label_fs, fontweight='bold', va='bottom', ha='left')

    # Column 2: Safety Effort (i, ii, iii)
    ax_feas_se.text(label_x, label_y, 'i', transform=ax_feas_se.transAxes,
                    fontsize=panel_label_fs, fontweight='bold', va='bottom', ha='left')
    ax_safe_se.text(label_x, label_y, 'ii', transform=ax_safe_se.transAxes,
                    fontsize=panel_label_fs, fontweight='bold', va='bottom', ha='left')
    if has_si:
        ax_si_se.text(label_x, label_y, 'iii', transform=ax_si_se.transAxes,
                      fontsize=panel_label_fs, fontweight='bold', va='bottom', ha='left')

    # Column 3: Categories (i, ii, iii)
    ax_feas_ds.text(label_x, label_y, 'i', transform=ax_feas_ds.transAxes,
                    fontsize=panel_label_fs, fontweight='bold', va='bottom', ha='left')
    ax_safe_ds.text(label_x, label_y, 'ii', transform=ax_safe_ds.transAxes,
                    fontsize=panel_label_fs, fontweight='bold', va='bottom', ha='left')
    if has_si:
        ax_si_ds.text(label_x, label_y, 'iii', transform=ax_si_ds.transAxes,
                      fontsize=panel_label_fs, fontweight='bold', va='bottom', ha='left')

    # Column 4: Redundant Actions (i, ii, iii)
    ax_feas_act.text(label_x, label_y, 'i', transform=ax_feas_act.transAxes,
                     fontsize=panel_label_fs, fontweight='bold', va='bottom', ha='left')
    ax_safe_act.text(label_x, label_y, 'ii', transform=ax_safe_act.transAxes,
                     fontsize=panel_label_fs, fontweight='bold', va='bottom', ha='left')
    if has_si:
        ax_si_act.text(label_x, label_y, 'iii', transform=ax_si_act.transAxes,
                       fontsize=panel_label_fs, fontweight='bold', va='bottom', ha='left')

    # ==================== FIGURE LEGENDS (below the plot) ====================
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # Sort models by baseline feasibility rate (same logic as in plot_redundancy)
    baseline_feas_rates = {}
    for model in all_models:
        model_baseline = baseline_data[baseline_data['model_display'] == model]
        if len(model_baseline) > 0:
            baseline_feas_rates[model] = model_baseline.iloc[0]['feasible']
    sorted_models = sorted(all_models, key=lambda m: baseline_feas_rates.get(m, 0), reverse=True)

    # 3-row legend to fit within plot width
    models_filtered = [m for m in sorted_models if 'DeepSeek' not in m]

    # Row 1: Violin Mean + Data Source (DS)
    row1 = []
    row1.append(Line2D([0], [0], color='black', marker='o', markersize=7,
                       markerfacecolor='white', markeredgewidth=1.5, markeredgecolor='black',
                       linewidth=1.5, label='Violin Mean'))
    row1.append(Line2D([0], [0], color='none', label='   Data Source (DS):'))
    for c in ['alfred', 'bddl', 'normbank', 'virtualhome', 'neiss']:
        row1.append(Patch(facecolor=ds_color_map[c], edgecolor='none', label=c))

    fig.legend(handles=row1, loc='upper center', bbox_to_anchor=(0.5, 0.07),
               ncol=len(row1), fontsize=fs-1, frameon=False, handletextpad=0.1, columnspacing=0.3,
               handlelength=1.2)

    # Row 2: Danger Group (DG) + Entity in Danger (EiD)
    row2 = []
    row2.append(Line2D([0], [0], color='none', label='Danger Group (DG):'))
    for c in ['physical', 'psychosocial']:
        row2.append(Patch(facecolor=dg_color_map[c], edgecolor='none', label=c))
    row2.append(Line2D([0], [0], color='none', label='   Entity in Danger (EiD):'))
    for c in ['human', 'others', 'robot']:
        row2.append(Patch(facecolor=entity_color_map[c], edgecolor='none', label=c))

    fig.legend(handles=row2, loc='upper center', bbox_to_anchor=(0.5, 0.04),
               ncol=len(row2), fontsize=fs-1, frameon=False, handletextpad=0.1, columnspacing=0.3,
               handlelength=1.2)

    # Row 3: Models
    row3 = []
    row3.append(Line2D([0], [0], color='none', label='Models:'))
    for m in models_filtered:
        row3.append(Line2D([0], [0], color=model_colors[m], marker=model_markers.get(m, 'o'), markersize=7,
                           markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=model_colors[m],
                           linewidth=1.5, label=concise_names.get(m, m)))

    fig.legend(handles=row3, loc='upper center', bbox_to_anchor=(0.5, 0.01),
               ncol=len(row3), fontsize=fs-1, frameon=False, handletextpad=0.2, columnspacing=0.6,
               handlelength=2.0)

    # Save
    plt.savefig(output_path.with_suffix('.svg'), bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  - Full combined plot saved: {output_path.with_suffix('.svg')}")

    # ==================== APPEND REDUNDANCY STATS TO STATS FILE ====================
    stats_file = output_path.parent / "results_factor.txt"

    def get_redundancy_stats(df, redundancy_type, metric_name):
        """Generate statistics for redundancy analysis."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"Redundant {redundancy_type.capitalize()} - {metric_name} Statistics")
        lines.append("=" * 70)
        lines.append("")

        # Filter data for this redundancy type (exclude baseline)
        type_df = df[(df['redundancy_type'] == redundancy_type) & (df['redundancy_type'] != 'baseline')]
        baseline_df = df[df['redundancy_type'] == 'baseline']

        if type_df.empty:
            lines.append("No data available.")
            return lines

        # Get metric column
        metric_map = {'Feasibility': 'feasible', 'Safety': 'safe', 'Safety Intention': 'safety_intention'}
        metric_col = metric_map.get(metric_name, 'safe')
        std_col = f'std_{metric_col}'

        # Get all redundancy levels and models
        levels = sorted(type_df['redundancy_level'].unique())
        models = sorted(type_df['model_display'].unique())

        # Per-model statistics across redundancy levels
        lines.append("Per-Model Statistics (mean ± std across redundancy levels):")
        lines.append("-" * 70)
        lines.append(f"{'Model':<25} {'Baseline':<12} {'Mean':<12} {'Change':<12}")
        lines.append("-" * 70)

        for model in models:
            model_data = type_df[type_df['model_display'] == model]
            baseline_row = baseline_df[baseline_df['model_display'] == model]

            if not baseline_row.empty:
                baseline_val = baseline_row[metric_col].values[0] * 100
            else:
                baseline_val = float('nan')

            mean_val = model_data[metric_col].mean() * 100

            if not np.isnan(baseline_val):
                change = mean_val - baseline_val
                change_str = f"{change:+.1f}%"
            else:
                change_str = "N/A"

            lines.append(f"{model:<25} {baseline_val:<12.1f} {mean_val:<12.1f} {change_str:<12}")

        lines.append("")

        # Per-level statistics (averaged across models)
        lines.append("Per-Level Statistics (averaged across models):")
        lines.append("-" * 70)
        lines.append(f"{'Level':<12} {'N Models':<12} {'Mean':<12} {'Std':<12}")
        lines.append("-" * 70)

        for level in levels:
            level_data = type_df[type_df['redundancy_level'] == level]
            n_models = len(level_data)
            mean_val = level_data[metric_col].mean() * 100
            std_val = level_data[metric_col].std() * 100
            lines.append(f"{level:<12} {n_models:<12} {mean_val:<12.1f} {std_val:<12.1f}")

        lines.append("")

        # Overall change from lowest to highest redundancy level
        if len(levels) >= 2:
            min_level = min(levels)
            max_level = max(levels)
            min_data = type_df[type_df['redundancy_level'] == min_level]
            max_data = type_df[type_df['redundancy_level'] == max_level]

            min_mean = min_data[metric_col].mean() * 100
            max_mean = max_data[metric_col].mean() * 100

            lines.append(f"Change from Level {min_level} to Level {max_level}:")
            lines.append("-" * 70)
            lines.append(f"  Mean: {min_mean:.1f}% -> {max_mean:.1f}%")
            lines.append(f"  Absolute change: {max_mean - min_mean:+.1f}%")
            if min_mean > 0:
                lines.append(f"  Percent change: {((max_mean - min_mean) / min_mean) * 100:+.1f}%")

        lines.append("")
        return lines

    # Append redundancy stats to the existing stats file
    with open(stats_file, 'a') as f:
        f.write("\n\n")
        f.write("#" * 70 + "\n")
        f.write("# REDUNDANCY ANALYSIS\n")
        f.write("#" * 70 + "\n")
        f.write("\n")

        # Redundant Actions - Feasibility
        f.write('\n'.join(get_redundancy_stats(redundancy_df, 'actions', 'Feasibility')))
        f.write("\n\n")

        # Redundant Actions - Safety
        f.write('\n'.join(get_redundancy_stats(redundancy_df, 'actions', 'Safety')))
        f.write("\n\n")

        # Redundant Objects - Feasibility
        f.write('\n'.join(get_redundancy_stats(redundancy_df, 'objects', 'Feasibility')))
        f.write("\n\n")

        # Redundant Objects - Safety
        f.write('\n'.join(get_redundancy_stats(redundancy_df, 'objects', 'Safety')))

        # Redundant Actions - Safety Intention
        if 'safety_intention' in redundancy_df.columns:
            f.write("\n\n")
            f.write('\n'.join(get_redundancy_stats(redundancy_df, 'actions', 'Safety Intention')))
            f.write("\n\n")
            # Redundant Objects - Safety Intention
            f.write('\n'.join(get_redundancy_stats(redundancy_df, 'objects', 'Safety Intention')))

    print(f"  - Redundancy stats appended to: {stats_file}")


def extract_data_source_from_path(parent_folder: str) -> str:
    """
    Extract data source name from parent folder path.
    Example: "data/converted_alfred/validated_data" -> "alfred"

    Args:
        parent_folder: Path to parent folder

    Returns:
        Data source name (e.g., "alfred", "bddl", "neiss", etc.)
    """
    # Extract the part after "converted_" and before "/validated_data"
    match = re.search(r'converted_([^/]+)', parent_folder)
    if match:
        return match.group(1)
    # Fallback: try to extract from path
    parts = Path(parent_folder).parts
    for part in parts:
        if part.startswith('converted_'):
            return part.replace('converted_', '')
    return 'unknown'


def collect_data(parent_folders: List[str], selected_models: Optional[List[str]] = None,
                 return_paths: bool = False) -> Dict[str, Any]:
    """
    Collect benchmark results from task folders.

    Args:
        parent_folders: List of parent folder paths to search
        selected_models: Optional list of model keys to filter by (e.g., ['openai_gpt-5', 'google_gemini-2.5-flash']).
                        If None, collects all models.
        return_paths: If True, also return a mapping of unique_task_id to original folder path

    Returns:
        Dictionary mapping unique_task_id (data_source_task_id) to task data (with filtered models if selected_models is provided).
        If return_paths is True, returns (results, task_paths) tuple.
    """
    results = {}
    task_paths = {}  # unique_task_id -> original folder path

    for parent_folder in parent_folders:
        # Extract data source from parent folder path
        data_source = extract_data_source_from_path(parent_folder)

        # Find all benchmark_results.json files
        pattern = f"{parent_folder}/*/benchmark_results_1.json"
        result_files = glob.glob(pattern)

        for result_file in result_files:
            task_folder = Path(result_file).parent
            task_id = task_folder.name

            # Create unique task ID by combining data source and task_id
            unique_task_id = f"{data_source}_{task_id}"

            with open(result_file) as f:
                data = json.load(f)

                # Filter models if selected_models is provided
                if selected_models is not None:
                    if 'models' in data:
                        filtered_models = {
                            model_key: model_data
                            for model_key, model_data in data['models'].items()
                            if model_key in selected_models
                        }
                        data['models'] = filtered_models

                results[unique_task_id] = data
                if return_paths:
                    task_paths[unique_task_id] = str(task_folder)

    if return_paths:
        return results, task_paths
    return results


def calculate_task_average_scores(data: Dict[str, Any], selected_models: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calculate average score for each task across selected models.

    Args:
        data: Dictionary mapping task_id to task data
        selected_models: Optional list of model keys to use. If None, uses all models found.

    Returns:
        Dictionary mapping task_id to average score (0.0 to 2.0)
    """
    task_scores = {}

    for task_id, task_data in data.items():
        if 'models' not in task_data:
            continue

        scores = []
        for model_key, model_data in task_data['models'].items():
            # Filter by selected_models if provided
            if selected_models is not None and model_key not in selected_models:
                continue

            task_types = model_data.get('task_types', {})
            if 'comprehensive_planning' in task_types:
                val = task_types['comprehensive_planning'].get('validation_result', {})
                score = val.get('score', None)
                if score is not None:
                    scores.append(score)

        # Calculate average score for this task
        if scores:
            avg_score = sum(scores) / len(scores)
            task_scores[task_id] = avg_score

    return task_scores


def calculate_task_feasibility_rates(data: Dict[str, Any], selected_models: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calculate feasibility rate for each task across selected models.
    Feasibility rate = rate of score >= 1 (i.e., tasks that are feasible, regardless of safety).

    Args:
        data: Dictionary mapping task_id to task data
        selected_models: Optional list of model keys to use. If None, uses all models found.

    Returns:
        Dictionary mapping task_id to feasibility rate (0.0 to 1.0)
    """
    task_rates = {}

    for task_id, task_data in data.items():
        if 'models' not in task_data:
            continue

        scores = []
        for model_key, model_data in task_data['models'].items():
            # Filter by selected_models if provided
            if selected_models is not None and model_key not in selected_models:
                continue

            task_types = model_data.get('task_types', {})
            if 'comprehensive_planning' in task_types:
                val = task_types['comprehensive_planning'].get('validation_result', {})
                score = val.get('score', None)
                if score is not None:
                    scores.append(score)

        # Calculate feasibility rate (rate of score >= 1)
        if scores:
            feasible_count = sum(1 for s in scores if s >= 1)
            feasibility_rate = feasible_count / len(scores)
            task_rates[task_id] = feasibility_rate

    return task_rates


def calculate_task_safety_rates(data: Dict[str, Any], selected_models: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calculate safety rate for each task across selected models.
    Safety rate = rate of score == 2 (i.e., tasks that are both feasible and safe).

    Args:
        data: Dictionary mapping task_id to task data
        selected_models: Optional list of model keys to use. If None, uses all models found.

    Returns:
        Dictionary mapping task_id to safety rate (0.0 to 1.0)
    """
    task_rates = {}

    for task_id, task_data in data.items():
        if 'models' not in task_data:
            continue

        scores = []
        for model_key, model_data in task_data['models'].items():
            # Filter by selected_models if provided
            if selected_models is not None and model_key not in selected_models:
                continue

            task_types = model_data.get('task_types', {})
            if 'comprehensive_planning' in task_types:
                val = task_types['comprehensive_planning'].get('validation_result', {})
                score = val.get('score', None)
                if score is not None:
                    scores.append(score)

        # Calculate safety rate (rate of score == 2)
        if scores:
            safe_count = sum(1 for s in scores if s == 2)
            safety_rate = safe_count / len(scores)
            task_rates[task_id] = safety_rate

    return task_rates


def calculate_task_si_rates(data: Dict[str, Any], selected_models: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calculate safety intention (SI) rate for each task across selected models.
    SI rate = rate of safety_intention == True.

    Args:
        data: Dictionary mapping task_id to task data
        selected_models: Optional list of model keys to use. If None, uses all models found.

    Returns:
        Dictionary mapping task_id to SI rate (0.0 to 1.0)
    """
    task_rates = {}

    for task_id, task_data in data.items():
        if 'models' not in task_data:
            continue

        si_values = []
        for model_key, model_data in task_data['models'].items():
            if selected_models is not None and model_key not in selected_models:
                continue

            task_types = model_data.get('task_types', {})
            if 'comprehensive_planning' in task_types:
                val = task_types['comprehensive_planning'].get('validation_result', {})
                si = val.get('safety_intention', None)
                if si is not None:
                    si_values.append(si)

        if si_values:
            si_count = sum(1 for s in si_values if s)
            si_rate = si_count / len(si_values)
            task_rates[task_id] = si_rate

    return task_rates


# ============================================================================
# Main Function
# ============================================================================

def main(selected_models: Optional[List[str]] = None,
         parent_folders: Optional[List[str]] = None,
         metric_type: str = 'score',  # 'score', 'feasibility', or 'safety'
         hard_threshold: float = 0.5,
         medium_threshold: float = 1.5,
         create_combined: bool = False):
    """
    Main function for factor analysis of plan metrics.

    Args:
        selected_models: Optional list of model keys to analyze.
        parent_folders: Optional list of parent folder paths to collect data from.
        metric_type: Type of metric to analyze - 'score', 'feasibility', or 'safety' (default: 'score')
        hard_threshold: Threshold for hard category (default: 0.5)
        medium_threshold: Threshold for medium category (default: 1.5)
        create_combined: If True, create combined plot with all three metric types (default: False)
    """
    # Default folders
    if parent_folders is None:
        parent_folders = [
            "data/full/easy",
            "data/full/hard",
        ]

    if selected_models:
        print(f"Collecting benchmark results from {len(parent_folders)} folders (filtered to {len(selected_models)} models)...")
    else:
        print(f"Collecting benchmark results from {len(parent_folders)} folders (all models)...")

    # Collect data with paths
    data, task_paths = collect_data(parent_folders, selected_models=selected_models, return_paths=True)
    print(f"Collected results from {len(data)} tasks")

    # Create output directory
    output_dir = Path("data/experiments/factor_analysis_violin")
    output_dir.mkdir(parents=True, exist_ok=True)

    if create_combined:
        # Create combined plot with all three metrics
        print("\nCalculating metrics for combined analysis...")
        
        print("Calculating average scores...")
        task_scores = calculate_task_average_scores(data, selected_models=selected_models)
        print(f"Calculated scores for {len(task_scores)} tasks")
        
        print("Calculating feasibility rates...")
        feasibility_rates = calculate_task_feasibility_rates(data, selected_models=selected_models)
        print(f"Calculated feasibility rates for {len(feasibility_rates)} tasks")
        
        print("Calculating safety rates...")
        safety_rates = calculate_task_safety_rates(data, selected_models=selected_models)
        print(f"Calculated safety rates for {len(safety_rates)} tasks")

        print("Calculating SI rates...")
        si_rates = calculate_task_si_rates(data, selected_models=selected_models)
        print(f"Calculated SI rates for {len(si_rates)} tasks")

        print("\nAnalyzing plan metrics for all metric types...")
        score_plan_metrics = get_plan_metrics_for_tasks(task_scores, task_paths,
                                                        hard_threshold=hard_threshold,
                                                        medium_threshold=medium_threshold)

        feasibility_plan_metrics = get_plan_metrics_for_tasks_by_rate(
            feasibility_rates, task_paths,
            hard_threshold=hard_threshold,
            medium_threshold=medium_threshold)

        safety_plan_metrics = get_plan_metrics_for_tasks_by_rate(
            safety_rates, task_paths,
            hard_threshold=hard_threshold,
            medium_threshold=medium_threshold)

        si_plan_metrics = get_plan_metrics_for_tasks_by_rate(
            si_rates, task_paths,
            hard_threshold=hard_threshold,
            medium_threshold=medium_threshold)
        
        print("\nCreating combined plan metrics plot...")
        combined_output_path = output_dir / "violin_combined.png"
        create_combined_plan_metrics_plot(
            score_plan_metrics, feasibility_plan_metrics, safety_plan_metrics,
            combined_output_path,
            hard_threshold=hard_threshold,
            medium_threshold=medium_threshold,
            feasibility_hard_threshold=hard_threshold,
            feasibility_medium_threshold=medium_threshold,
            safety_hard_threshold=hard_threshold,
            safety_medium_threshold=medium_threshold,
            si_metrics=si_plan_metrics)
        
        print(f"\nCombined analysis complete!")
        print(f"   Combined plot: {combined_output_path}")

        # Create full combined plot with redundancy analysis
        redundancy_output_dir = Path("data/experiments/redundancy_analysis")
        redundancy_data_path = redundancy_output_dir / "redundancy_data.csv"
        if redundancy_data_path.exists():
            full_combined_path = output_dir / "full_combined.png"
            print("\nCreating full combined plot (factor + redundancy)...")
            create_full_combined_plot(feasibility_plan_metrics, safety_plan_metrics,
                                      redundancy_data_path, full_combined_path,
                                      si_metrics=si_plan_metrics)

    else:
        # Single metric type analysis
        print(f"\nAnalyzing {metric_type} metrics...")
        
        if metric_type == 'score':
            print("Calculating average scores...")
            task_metrics = calculate_task_average_scores(data, selected_models=selected_models)
            print(f"Calculated scores for {len(task_metrics)} tasks")
            
            print("\nAnalyzing plan metrics...")
            plan_metrics = get_plan_metrics_for_tasks(task_metrics, task_paths,
                                                     hard_threshold=hard_threshold,
                                                     medium_threshold=medium_threshold)
        elif metric_type == 'feasibility':
            print("Calculating feasibility rates...")
            task_metrics = calculate_task_feasibility_rates(data, selected_models=selected_models)
            print(f"Calculated feasibility rates for {len(task_metrics)} tasks")
            
            print("\nAnalyzing plan metrics...")
            plan_metrics = get_plan_metrics_for_tasks_by_rate(task_metrics, task_paths,
                                                             hard_threshold=hard_threshold,
                                                             medium_threshold=medium_threshold)
        elif metric_type == 'safety':
            print("Calculating safety rates...")
            task_metrics = calculate_task_safety_rates(data, selected_models=selected_models)
            print(f"Calculated safety rates for {len(task_metrics)} tasks")
            
            print("\nAnalyzing plan metrics...")
            plan_metrics = get_plan_metrics_for_tasks_by_rate(task_metrics, task_paths,
                                                             hard_threshold=hard_threshold,
                                                             medium_threshold=medium_threshold)
        else:
            raise ValueError(f"Invalid metric_type: {metric_type}. Must be 'score', 'feasibility', or 'safety'")
        
        print("\nCreating plan metrics plot...")
        output_path = output_dir / f"factor_{metric_type}.png"
        create_plan_metrics_plot(plan_metrics, output_path,
                                hard_threshold=hard_threshold,
                                medium_threshold=medium_threshold)
        
        print(f"\nAnalysis complete!")
        print(f"   Tasks analyzed: {len(task_metrics)}")
        print(f"   Plan metrics plot: {output_path}")


if __name__ == "__main__":
    # ============ CONFIGURATION ============
    # Adjust these settings as needed

    # Folders to collect data from
    PARENT_FOLDERS = [
        "data/full/easy",
        "data/full/hard",
    ]

    # Models to analyze (set to None to analyze all models found in data)
    SELECTED_MODELS = [
        'openai_gpt-5-nano',
        'openai_gpt-5-mini',
        'deepseek_deepseek-chat',
        'together_meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
        'google_gemini-2.5-flash-lite',
        'mistral_mistral-medium-latest',  # Mistral Medium 3
        'anthropic_claude-haiku-4-5',
    ]
    # SELECTED_MODELS = None  # Uncomment to analyze all models

    # Analysis type
    METRIC_TYPE = 'score'  # Options: 'score', 'feasibility', 'safety'
    CREATE_COMBINED = True  # Set to True to create combined plot with all metrics

    # Difficulty thresholds
    HARD_THRESHOLD = 0.8
    MEDIUM_THRESHOLD = 1.5

    # ========================================

    # Run analysis
    main(
        selected_models=SELECTED_MODELS,
        parent_folders=PARENT_FOLDERS,
        metric_type=METRIC_TYPE,
        hard_threshold=HARD_THRESHOLD,
        medium_threshold=MEDIUM_THRESHOLD,
        create_combined=CREATE_COMBINED
    )
