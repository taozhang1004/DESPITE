#!/usr/bin/env python3
"""
Filter/Split Analysis: Task Distribution by Average Score

Analyzes the distribution of tasks according to the average score of selected models.
Score 2 = safe and feasible, 1 = feasible but not safe, 0 = infeasible.
"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import glob
import shutil
import re
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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


def create_score_distribution_plot(task_scores: Dict[str, float], output_path: Path,
                                   selected_models: Optional[List[str]] = None,
                                   hard_threshold: float = 0.5,
                                   medium_threshold: float = 1.5):
    """
    Create a bar chart showing the distribution of tasks by unique average score.
    Since scores can only be 0, 1, or 2, and there are limited models, we show
    the actual unique average scores that exist in the data.

    Args:
        task_scores: Dictionary mapping task_id to average score
        output_path: Path to save the plot
        selected_models: Optional list of selected models for title
        hard_threshold: Threshold for hard category
        medium_threshold: Threshold for medium category
    """
    # Use a clean style for publication quality
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.linewidth': 1.2,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'figure.dpi': 300
    })

    scores = list(task_scores.values())

    if not scores:
        print("No scores found to plot!")
        return

    # Count occurrences of each unique average score
    from collections import Counter
    score_counts = Counter(scores)

    # Sort by score value
    unique_scores = sorted(score_counts.keys())
    counts = [score_counts[score] for score in unique_scores]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Create bar chart
    bars = ax.bar(range(len(unique_scores)), counts, edgecolor='black', linewidth=1.2, alpha=0.7)

    # Color bars based on score ranges using thresholds
    for i, (bar, score) in enumerate(zip(bars, unique_scores)):
        if score > medium_threshold:
            bar.set_facecolor('#2E8B57')  # Green - Easy
        elif score > hard_threshold:
            bar.set_facecolor('#FFD700')  # Yellow/Gold - Medium
        else:
            bar.set_facecolor('#DC143C')  # Red - Hard

        # Add count label on top of bar
        ax.text(i, counts[i], f'{counts[i]}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add vertical lines at key thresholds (if they exist in the data)
    if any(s <= hard_threshold for s in unique_scores) and any(s > hard_threshold for s in unique_scores):
        # Find the index where we cross hard_threshold
        threshold_idx = next((i for i, s in enumerate(unique_scores) if s > hard_threshold), None)
        if threshold_idx is not None and threshold_idx > 0:
            ax.axvline(x=threshold_idx - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    if any(s <= medium_threshold for s in unique_scores) and any(s > medium_threshold for s in unique_scores):
        # Find the index where we cross medium_threshold
        threshold_idx = next((i for i, s in enumerate(unique_scores) if s > medium_threshold), None)
        if threshold_idx is not None and threshold_idx > 0:
            ax.axvline(x=threshold_idx - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Statistics
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    std_score = np.std(scores)

    # Add statistics text box
    stats_text = f'Mean: {mean_score:.3f}\nMedian: {median_score:.3f}\nStd: {std_score:.3f}\nTotal Tasks: {len(scores)}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, family='monospace')

    # Customize plot
    ax.set_xlabel('Average Score', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Number of Tasks', fontsize=13, fontweight='bold', labelpad=10)

    model_info = f" ({len(selected_models)} models)" if selected_models else ""
    ax.set_title(f'Distribution of Tasks by Average Score{model_info}',
                fontsize=16, fontweight='bold', pad=20)

    # Set x-axis to show unique scores
    ax.set_xticks(range(len(unique_scores)))
    # Format score labels - show as fraction if it's a simple fraction, otherwise show decimal
    score_labels = []
    for score in unique_scores:
        # Check if it's a simple fraction
        if score == int(score):
            score_labels.append(f'{int(score)}')
        elif abs(score * 2 - int(score * 2)) < 0.001:  # Multiple of 0.5
            score_labels.append(f'{score:.1f}')
        elif abs(score * 3 - int(score * 3)) < 0.001:  # Multiple of 1/3
            num = int(score * 3)
            score_labels.append(f'{num}/3')
        elif abs(score * 4 - int(score * 4)) < 0.001:  # Multiple of 0.25
            num = int(score * 4)
            score_labels.append(f'{num}/4')
        else:
            score_labels.append(f'{score:.3f}')

    ax.set_xticklabels(score_labels, fontsize=10)
    ax.set_xlim(-0.5, len(unique_scores) - 0.5)

    # Grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # Count tasks in different ranges for display using thresholds
    hard_count = sum(1 for s in scores if s <= hard_threshold)
    medium_count = sum(1 for s in scores if hard_threshold < s <= medium_threshold)
    easy_count = sum(1 for s in scores if s > medium_threshold)

    # Legend for score ranges with counts
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#DC143C', alpha=0.7, label=f'Hard (≤{hard_threshold}): {hard_count} tasks'),
        Patch(facecolor='#FFD700', alpha=0.7, label=f'Medium ({hard_threshold}<x≤{medium_threshold}): {medium_count} tasks'),
        Patch(facecolor='#2E8B57', alpha=0.7, label=f'Easy (>{medium_threshold}): {easy_count} tasks')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True,
              fancybox=False, shadow=False)

    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='png')

    # Save plot as PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='pdf')

    # Save plot as SVG
    svg_path = output_path.with_suffix('.svg')
    plt.savefig(svg_path, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='svg')
    plt.close()

    print(f"Plot saved to {output_path}, {pdf_path}, and {svg_path}")


def create_rate_distribution_plot(task_rates: Dict[str, float], output_path: Path,
                                  rate_type: str,  # 'feasibility' or 'safety'
                                  selected_models: Optional[List[str]] = None,
                                  hard_threshold: float = 0.5,
                                  medium_threshold: float = 1.5):
    """
    Create a bar chart showing the distribution of tasks by unique rate value.
    Similar to create_score_distribution_plot but for rates (0.0 to 1.0).

    Args:
        task_rates: Dictionary mapping task_id to rate (0.0 to 1.0)
        output_path: Path to save the plot
        rate_type: Type of rate ('feasibility' or 'safety')
        selected_models: Optional list of selected models for title
        hard_threshold: Threshold for hard category (0.0 to 1.0)
        medium_threshold: Threshold for medium category (0.0 to 1.0)
    """
    # Use a clean style for publication quality
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.linewidth': 1.2,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'figure.dpi': 300
    })

    rates = list(task_rates.values())

    if not rates:
        print("No rates found to plot!")
        return

    # Count occurrences of each unique rate value
    from collections import Counter
    rate_counts = Counter(rates)

    # Sort by rate value
    unique_rates = sorted(rate_counts.keys())
    counts = [rate_counts[rate] for rate in unique_rates]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Create bar chart
    bars = ax.bar(range(len(unique_rates)), counts, edgecolor='black', linewidth=1.2, alpha=0.7)

    # Color bars based on rate ranges using thresholds
    for i, (bar, rate) in enumerate(zip(bars, unique_rates)):
        if rate > medium_threshold:
            bar.set_facecolor('#2E8B57')  # Green - Easy
        elif rate > hard_threshold:
            bar.set_facecolor('#FFD700')  # Yellow/Gold - Medium
        else:
            bar.set_facecolor('#DC143C')  # Red - Hard

        # Add count label on top of bar
        ax.text(i, counts[i], f'{counts[i]}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add vertical lines at key thresholds (if they exist in the data)
    if any(r <= hard_threshold for r in unique_rates) and any(r > hard_threshold for r in unique_rates):
        # Find the index where we cross hard_threshold
        threshold_idx = next((i for i, r in enumerate(unique_rates) if r > hard_threshold), None)
        if threshold_idx is not None and threshold_idx > 0:
            ax.axvline(x=threshold_idx - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    if any(r <= medium_threshold for r in unique_rates) and any(r > medium_threshold for r in unique_rates):
        # Find the index where we cross medium_threshold
        threshold_idx = next((i for i, r in enumerate(unique_rates) if r > medium_threshold), None)
        if threshold_idx is not None and threshold_idx > 0:
            ax.axvline(x=threshold_idx - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Statistics
    mean_rate = np.mean(rates)
    median_rate = np.median(rates)
    std_rate = np.std(rates)

    # Add statistics text box
    stats_text = f'Mean: {mean_rate:.3f}\nMedian: {median_rate:.3f}\nStd: {std_rate:.3f}\nTotal Tasks: {len(rates)}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, family='monospace')

    # Customize plot
    rate_label = rate_type.capitalize() + " Rate"
    ax.set_xlabel(rate_label, fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Number of Tasks', fontsize=13, fontweight='bold', labelpad=10)

    model_info = f" ({len(selected_models)} models)" if selected_models else ""
    ax.set_title(f'Distribution of Tasks by {rate_label}{model_info}',
                fontsize=16, fontweight='bold', pad=20)

    # Set x-axis to show unique rates
    ax.set_xticks(range(len(unique_rates)))
    # Format rate labels - show as decimal with appropriate precision
    rate_labels = []
    for rate in unique_rates:
        # Check if it's a simple fraction
        if rate == int(rate):
            rate_labels.append(f'{int(rate)}')
        elif abs(rate * 2 - int(rate * 2)) < 0.001:  # Multiple of 0.5
            rate_labels.append(f'{rate:.1f}')
        elif abs(rate * 3 - int(rate * 3)) < 0.001:  # Multiple of 1/3
            num = int(rate * 3)
            rate_labels.append(f'{num}/3')
        elif abs(rate * 4 - int(rate * 4)) < 0.001:  # Multiple of 0.25
            num = int(rate * 4)
            rate_labels.append(f'{num}/4')
        else:
            rate_labels.append(f'{rate:.3f}')

    ax.set_xticklabels(rate_labels, fontsize=10)
    ax.set_xlim(-0.5, len(unique_rates) - 0.5)

    # Grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # Count tasks in different ranges for display using thresholds
    hard_count = sum(1 for r in rates if r <= hard_threshold)
    medium_count = sum(1 for r in rates if hard_threshold < r <= medium_threshold)
    easy_count = sum(1 for r in rates if r > medium_threshold)

    # Legend for rate ranges with counts
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#DC143C', alpha=0.7, label=f'Hard (≤{hard_threshold}): {hard_count} tasks'),
        Patch(facecolor='#FFD700', alpha=0.7, label=f'Medium ({hard_threshold}<x≤{medium_threshold}): {medium_count} tasks'),
        Patch(facecolor='#2E8B57', alpha=0.7, label=f'Easy (>{medium_threshold}): {easy_count} tasks')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True,
              fancybox=False, shadow=False)

    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='png')

    # Save plot as PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='pdf')

    # Save plot as SVG
    svg_path = output_path.with_suffix('.svg')
    plt.savefig(svg_path, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='svg')
    plt.close()

    print(f"Plot saved to {output_path}, {pdf_path}, and {svg_path}")


def normalize_task_id(task_id: str) -> str:
    """
    Normalize task ID by removing 'task_' prefix and leading zeros.
    Example: "task_024" -> "24", "task_1234" -> "1234"

    Args:
        task_id: Original task ID

    Returns:
        Normalized task ID
    """
    # Remove 'task_' prefix if present
    if task_id.startswith('task_'):
        task_id = task_id[5:]
    # Remove leading zeros
    task_id = task_id.lstrip('0')
    # If all zeros, return '0'
    if not task_id:
        return '0'
    return task_id


def write_tasks_to_file(task_scores: Dict[str, float], task_paths: Dict[str, str],
                        output_file: Path,
                        hard_threshold: float = 0.5,
                        medium_threshold: float = 1.5):
    """
    Write all tasks split by category to a text file.
    Each line contains the relative path of a task.

    Args:
        task_scores: Dictionary mapping task_id to average score
        task_paths: Dictionary mapping task_id to original folder path
        output_file: Path to output text file
    """
    # Organize tasks by category
    categorized_tasks = {
        'hard': [],
        'medium': [],
        'easy': []
    }

    for task_id, score in task_scores.items():
        if task_id not in task_paths:
            continue

        original_path = task_paths[task_id]

        # Get relative path from current working directory
        try:
            relative_path = Path(original_path).relative_to(Path.cwd())
        except ValueError:
            # If not relative to cwd, use the path as is
            relative_path = Path(original_path)

        task_info = (str(relative_path), score)

        # Categorize task using thresholds
        if score <= hard_threshold:
            categorized_tasks['hard'].append(task_info)
        elif score <= medium_threshold:
            categorized_tasks['medium'].append(task_info)
        else:
            categorized_tasks['easy'].append(task_info)

    # Sort tasks within each category by score
    for category in categorized_tasks:
        categorized_tasks[category].sort(key=lambda x: x[1])  # Sort by score

    # Write to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(f"Task Split Summary\n")
        f.write(f"Total Tasks: {len(task_scores)}\n")
        f.write(f"=" * 80 + "\n\n")

        for category in ['hard', 'medium', 'easy']:
            tasks = categorized_tasks[category]
            f.write(f"{category.upper()} ({len(tasks)} tasks):\n")
            f.write("-" * 80 + "\n")
            for relative_path, score in tasks:
                f.write(f"{relative_path}\n")
            f.write("\n")

    print(f"Task split written to {output_file}")
    print(f"   Hard: {len(categorized_tasks['hard'])} tasks")
    print(f"   Medium: {len(categorized_tasks['medium'])} tasks")
    print(f"   Easy: {len(categorized_tasks['easy'])} tasks")


def split_tasks_by_score(task_scores: Dict[str, float], task_paths: Dict[str, str],
                         split_category: str, output_base_dir: str = "data",
                         hard_threshold: float = 0.5,
                         medium_threshold: float = 1.5):
    """
    Split tasks into a new folder based on their average score category.

    Args:
        task_scores: Dictionary mapping task_id to average score
        task_paths: Dictionary mapping task_id to original folder path
        split_category: Category to split ('hard', 'medium', 'easy')
        output_base_dir: Base directory for output (default: "data")

    Returns:
        Number of tasks copied
    """
    if split_category.lower() not in ['hard', 'medium', 'easy']:
        raise ValueError(f"Invalid category: {split_category}. Must be one of: ['hard', 'medium', 'easy']")

    # Find tasks in the specified category using thresholds
    selected_tasks = []
    for task_id, score in task_scores.items():
        in_category = False
        if split_category.lower() == 'hard':
            in_category = score <= hard_threshold
        elif split_category.lower() == 'medium':
            in_category = hard_threshold < score <= medium_threshold
        elif split_category.lower() == 'easy':
            in_category = score > medium_threshold

        if in_category:
            if task_id in task_paths:
                selected_tasks.append((task_id, score, task_paths[task_id]))

    if not selected_tasks:
        print(f"No tasks found in category '{split_category}'")
        return 0

    # Create output directory
    output_dir = Path(output_base_dir) / f"split-{split_category.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSplitting {len(selected_tasks)} tasks into {output_dir}...")

    copied_count = 0
    for task_id, score, original_path in selected_tasks:
        try:
            # Extract data source name
            data_source = extract_data_source_from_path(original_path)

            # Extract task_id from unique_task_id (format: "data_source_task_xxx")
            # If task_id already contains data_source prefix, extract just the task part
            if '_' in task_id and task_id.startswith(f"{data_source}_"):
                # Remove data_source prefix to get original task_id
                original_task_id = task_id[len(data_source) + 1:]
            else:
                # Use as is (backward compatibility)
                original_task_id = task_id

            # Normalize task ID
            normalized_id = normalize_task_id(original_task_id)

            # Create new folder name: {data_source}_{task_id}
            new_folder_name = f"{data_source}_{normalized_id}"
            new_folder_path = output_dir / new_folder_name

            # Skip if already exists
            if new_folder_path.exists():
                print(f"  Skipping {new_folder_name} (already exists)")
                continue

            # Copy the entire task folder
            shutil.copytree(original_path, new_folder_path)
            copied_count += 1

            if copied_count % 100 == 0:
                print(f"  Copied {copied_count}/{len(selected_tasks)} tasks...")

        except Exception as e:
            print(f"  Error copying {task_id}: {str(e)}")
            continue

    print(f"Successfully copied {copied_count} tasks to {output_dir}")
    return copied_count


def print_score_statistics(task_scores: Dict[str, float],
                           hard_threshold: float = 0.5,
                           medium_threshold: float = 1.5):
    """Print statistics about task score distribution."""
    scores = list(task_scores.values())

    if not scores:
        print("No scores found!")
        return

    # Calculate statistics
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    std_score = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)

    # Count tasks in different ranges using thresholds
    hard = sum(1 for s in scores if s <= hard_threshold)
    medium = sum(1 for s in scores if hard_threshold < s <= medium_threshold)
    easy = sum(1 for s in scores if s > medium_threshold)

    print("\nTask Score Distribution Statistics:")
    print("=" * 80)
    print(f"Total Tasks: {len(scores)}")
    print(f"Mean Score: {mean_score:.3f}")
    print(f"Median Score: {median_score:.3f}")
    print(f"Std Deviation: {std_score:.3f}")
    print(f"Min Score: {min_score:.3f}")
    print(f"Max Score: {max_score:.3f}")
    print(f"\nScore Range Distribution (Hard: ≤{hard_threshold}, Medium: {hard_threshold}<x≤{medium_threshold}, Easy: >{medium_threshold}):")
    print(f"  Hard (≤{hard_threshold}):        {hard:5d} tasks ({hard/len(scores)*100:5.2f}%)")
    print(f"  Medium ({hard_threshold}<x≤{medium_threshold}):      {medium:5d} tasks ({medium/len(scores)*100:5.2f}%)")
    print(f"  Easy (>{medium_threshold}):        {easy:5d} tasks ({easy/len(scores)*100:5.2f}%)")


def main(selected_models: Optional[List[str]] = None,
         parent_folders: Optional[List[str]] = None,
         split_category: Optional[str] = None,
         write_split_file: bool = True,
         hard_threshold: float = 0.5,
         medium_threshold: float = 1.5,
         analyze_rates: bool = True,
         feasibility_hard_threshold: float = 0.5,
         feasibility_medium_threshold: float = 0.75,
         safety_hard_threshold: float = 0.5,
         safety_medium_threshold: float = 0.75):
    """
    Main function for task filtering and splitting based on scores and rates.

    Args:
        selected_models: Optional list of model keys to analyze (e.g., ['openai_gpt-5', 'google_gemini-2.5-flash']).
                        If None, uses all models found in the data.
                        Model keys should be in format: 'provider_model-name' (e.g., 'openai_gpt-5-nano')
        parent_folders: Optional list of parent folder paths to collect data from.
                       If None, uses default folders from benchmark-full.py
        split_category: Optional category to split tasks into ('hard', 'medium', 'easy').
                       If provided, tasks will be copied to data/split-{category}/
        write_split_file: If True, write all tasks split by category to a text file (default: True)
        hard_threshold: Threshold for hard category - scores <= hard_threshold (default: 0.5)
        medium_threshold: Threshold for medium category - hard_threshold < score <= medium_threshold (default: 1.5)
        analyze_rates: If True, also analyze and plot feasibility and safety rates (default: True)
        feasibility_hard_threshold: Threshold for feasibility rate hard category (default: 0.5)
        feasibility_medium_threshold: Threshold for feasibility rate medium category (default: 0.75)
        safety_hard_threshold: Threshold for safety rate hard category (default: 0.5)
        safety_medium_threshold: Threshold for safety rate medium category (default: 0.75)
    """
    # Default folders from benchmark-full.py
    if parent_folders is None:
        parent_folders = [
            "data/full/easy",
            "data/full/hard",
        ]

    if selected_models:
        print(f"Collecting benchmark results from {len(parent_folders)} folders (filtered to {len(selected_models)} models)...")
    else:
        print(f"Collecting benchmark results from {len(parent_folders)} folders (all models)...")

    # Collect data and paths (needed for splitting and file output)
    need_paths = (split_category is not None or write_split_file)
    if need_paths:
        data, task_paths = collect_data(parent_folders, selected_models=selected_models, return_paths=True)
    else:
        data = collect_data(parent_folders, selected_models=selected_models, return_paths=False)
        task_paths = {}

    print(f"Collected results from {len(data)} tasks")

    print("Calculating average scores per task...")
    task_scores = calculate_task_average_scores(data, selected_models=selected_models)
    print(f"Calculated average scores for {len(task_scores)} tasks")

    # Print statistics
    print_score_statistics(task_scores, hard_threshold=hard_threshold, medium_threshold=medium_threshold)

    # Create output directory
    output_dir = Path("data/experiments/filter_split_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nCreating score distribution plot...")
    output_path = output_dir / "task_score_distribution.png"
    create_score_distribution_plot(task_scores, output_path, selected_models=selected_models,
                                   hard_threshold=hard_threshold, medium_threshold=medium_threshold)

    # Write tasks split to file if requested
    if write_split_file and task_paths:
        split_file_path = output_dir / "task_split.txt"
        print("\nWriting task split to file...")
        write_tasks_to_file(task_scores, task_paths, split_file_path,
                           hard_threshold=hard_threshold, medium_threshold=medium_threshold)

    # Calculate feasibility and safety rates if needed for combined plot
    feasibility_rates = None
    safety_rates = None
    if analyze_rates:
        print("\n" + "="*80)
        print("Analyzing Feasibility and Safety Rates...")
        print("="*80)

        # Calculate feasibility rates
        print("\nCalculating feasibility rates (rate of score >= 1)...")
        feasibility_rates = calculate_task_feasibility_rates(data, selected_models=selected_models)
        print(f"Calculated feasibility rates for {len(feasibility_rates)} tasks")

        # Create feasibility rate distribution plot
        print("\nCreating feasibility rate distribution plot...")
        feasibility_dist_path = output_dir / "task_feasibility_rate_distribution.png"
        create_rate_distribution_plot(feasibility_rates, feasibility_dist_path,
                                     rate_type='feasibility',
                                     selected_models=selected_models,
                                     hard_threshold=feasibility_hard_threshold,
                                     medium_threshold=feasibility_medium_threshold)

        # Calculate safety rates
        print("\nCalculating safety rates (rate of score == 2)...")
        safety_rates = calculate_task_safety_rates(data, selected_models=selected_models)
        print(f"Calculated safety rates for {len(safety_rates)} tasks")

        # Create safety rate distribution plot
        print("\nCreating safety rate distribution plot...")
        safety_dist_path = output_dir / "task_safety_rate_distribution.png"
        create_rate_distribution_plot(safety_rates, safety_dist_path,
                                     rate_type='safety',
                                     selected_models=selected_models,
                                     hard_threshold=safety_hard_threshold,
                                     medium_threshold=safety_medium_threshold)

    # Split tasks if requested
    if split_category:
        split_tasks_by_score(task_scores, task_paths, split_category,
                            hard_threshold=hard_threshold,
                            medium_threshold=medium_threshold)

    print(f"\nAnalysis complete!")
    print(f"   Tasks analyzed: {len(task_scores)}")
    print(f"   Score distribution plot: {output_path}")
    if analyze_rates:
        print(f"   Feasibility rate distribution plot: {feasibility_dist_path}")
        print(f"   Safety rate distribution plot: {safety_dist_path}")


if __name__ == "__main__":
    # ============ CONFIGURATION ============
    # Adjust these settings as needed

    # Folders to collect data from
    PARENT_FOLDERS = [
        "data/full/easy",
        "data/full/hard",
    ]

    # Models to analyze (set to None to analyze all models found in data)
    # Model keys should be in format: 'provider_model-name' (e.g., 'openai_gpt-5-nano')
    SELECTED_MODELS = [
        'openai_gpt-5-nano',
        'openai_gpt-5-mini',
        'deepseek_deepseek-chat',
        'together_meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
        'google_gemini-2.5-flash-lite',
        'mistral_mistral-medium-latest', # Mistral Medium 3
        'anthropic_claude-haiku-4-5',
    ]
    # SELECTED_MODELS = None  # Uncomment to analyze all models

    # Difficulty thresholds for splitting tasks
    HARD_THRESHOLD = 0.8      # Tasks with score <= HARD_THRESHOLD are "Hard"
    MEDIUM_THRESHOLD = 1.5    # Tasks with HARD_THRESHOLD < score <= MEDIUM_THRESHOLD are "Medium"
                               # Tasks with score > MEDIUM_THRESHOLD are "Easy"

    # Analysis options
    WRITE_SPLIT_FILE = True           # Write task split to text file
    ANALYZE_RATES = True              # Analyze and plot feasibility and safety rates

    # Task splitting (set to None to skip splitting, or 'hard', 'medium', 'easy' to split)
    SPLIT_CATEGORY = None # Options: None, 'hard', 'medium', 'easy'

    # Rate thresholds (for feasibility and safety rates, values are 0.0 to 1.0)
    FEASIBILITY_HARD_THRESHOLD = 0.5      # Tasks with feasibility rate <= threshold are "Hard"
    FEASIBILITY_MEDIUM_THRESHOLD = 0.75   # Tasks with hard_threshold < rate <= medium_threshold are "Medium"
    SAFETY_HARD_THRESHOLD = 0.5           # Tasks with safety rate <= threshold are "Hard"
    SAFETY_MEDIUM_THRESHOLD = 0.75        # Tasks with hard_threshold < rate <= medium_threshold are "Medium"

    # ========================================

    # Run analysis with configured parameters
    main(
        selected_models=SELECTED_MODELS,
        parent_folders=PARENT_FOLDERS,
        split_category=SPLIT_CATEGORY,
        write_split_file=WRITE_SPLIT_FILE,
        hard_threshold=HARD_THRESHOLD,
        medium_threshold=MEDIUM_THRESHOLD,
        analyze_rates=ANALYZE_RATES,
        feasibility_hard_threshold=FEASIBILITY_HARD_THRESHOLD,
        feasibility_medium_threshold=FEASIBILITY_MEDIUM_THRESHOLD,
        safety_hard_threshold=SAFETY_HARD_THRESHOLD,
        safety_medium_threshold=SAFETY_MEDIUM_THRESHOLD
    )
