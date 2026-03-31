#!/usr/bin/env python3
"""
Distribution Analysis: Visualize the distribution of tasks by plan length and plan delta.

Creates a scatter plot showing how tasks are distributed across different combinations
of plan length (average of safe and unsafe feasible plans) and plan delta.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


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


def collect_task_distributions(parent_folders: List[str]) -> List[Dict]:
    """
    Collect plan length and plan delta for all tasks in the given parent folders.
    
    Args:
        parent_folders: List of parent folder paths to search
    
    Returns:
        List of dictionaries with task_id, avg_plan_length, and plan_delta
    """
    results = []
    
    for parent_folder in parent_folders:
        parent_path = Path(parent_folder)
        if not parent_path.exists():
            print(f"⚠️  Directory not found: {parent_folder}")
            continue
        
        print(f"📂 Collecting from {parent_folder}...")
        task_count = 0
        
        # Find all task directories (check for code.py to ensure it's a valid task)
        for task_dir in sorted(parent_path.iterdir()):
            if not task_dir.is_dir():
                continue
            
            # Only process if code.py exists (valid task)
            if not (task_dir / "code.py").exists():
                continue
            
            # Calculate plan metrics
            unsafe_len = calculate_plan_length(task_dir, 'unsafe')
            safe_len = calculate_plan_length(task_dir, 'safe')
            
            # Skip if we don't have both plans
            if unsafe_len == 0 and safe_len == 0:
                continue
            
            # Calculate average plan length (avg of safe and unsafe feasible plans)
            if unsafe_len > 0 and safe_len > 0:
                avg_plan_length = (unsafe_len + safe_len) / 2.0
            elif unsafe_len > 0:
                avg_plan_length = unsafe_len
            elif safe_len > 0:
                avg_plan_length = safe_len
            else:
                continue
            
            plan_delta = safe_len - unsafe_len
            
            results.append({
                'task_id': task_dir.name,
                'avg_plan_length': avg_plan_length,
                'plan_delta': plan_delta,
                'unsafe_len': unsafe_len,
                'safe_len': safe_len
            })
            task_count += 1
        
        print(f"   Found {task_count} tasks")
    
    return results


def create_heatmap_plot(task_data: List[Dict], output_path: Path):
    """
    Create a Nature/Science quality heatmap showing task distribution.

    Args:
        task_data: List of task dictionaries with avg_plan_length and plan_delta
        output_path: Path to save the plot
    """
    if not task_data:
        print("❌ No task data to plot")
        return

    # Count tasks at each (plan_length, plan_delta) combination
    distribution = defaultdict(int)
    for task in task_data:
        plan_length = round(task['avg_plan_length'])
        plan_delta = task['plan_delta']
        distribution[(plan_length, plan_delta)] += 1

    # Determine grid dimensions (include ALL data)
    x_values = [k[0] for k in distribution.keys()]
    y_values = [k[1] for k in distribution.keys()]

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    # Create 2D grid for heatmap
    grid = np.zeros((y_max - y_min + 1, x_max - x_min + 1))

    for (plan_length, plan_delta), count in distribution.items():
        x_idx = plan_length - x_min
        y_idx = plan_delta - y_min
        grid[y_idx, x_idx] = count

    # Use log scale for color (better visibility across wide range)
    # Mask zeros so they appear as white background
    # Use log10(count) directly - zeros are masked, so no +1 needed
    grid_log = np.ma.masked_where(grid == 0, np.log10(np.where(grid == 0, 1, grid)))
    grid_display = grid_log  # Already masked

    # === Nature/Science style setup ===
    plt.style.use('default')  # Reset to default first
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.linewidth': 0.8,
        'axes.grid': False,  # No grid
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
    })

    # Figure size - more compact
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(False)  # Ensure no grid

    # Create heatmap with muted blue to muted red colormap
    # Use pcolormesh for proper vector output in SVG
    from matplotlib.colors import LinearSegmentedColormap
    # Muted blue → muted purple → muted red
    colors = ['#c4d4e8', '#a8b8d4', '#9a9fc4', '#ab8fa8', '#b87f8c', '#c46f70', '#b85450']
    cmap = LinearSegmentedColormap.from_list('muted_blue_red', colors)
    cmap.set_bad(color='white')  # Masked (zero) values appear white

    # Create coordinate arrays for pcolormesh
    x_edges = np.arange(x_min - 0.5, x_max + 1.5, 1)
    y_edges = np.arange(y_min - 0.5, y_max + 1.5, 1)
    im = ax.pcolormesh(x_edges, y_edges, grid_display, cmap=cmap, vmin=0, shading='flat',
                       edgecolors='none', linewidth=0, antialiased=False)  # Remove white borders

    # Professional colorbar at the top (1 to 1000)
    # vmin=0 (log10(1)=0), vmax based on data
    im.set_clim(vmin=0, vmax=grid_log.max())
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.6, aspect=18,
                        pad=0.02, location='top')

    # Set colorbar ticks at evenly spaced log positions: 0, 1, 2, 3 = log10(1, 10, 100, 1000)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['1', '10', '100', '1000'])
    cbar.ax.tick_params(labelsize=7, width=0.3, length=2, pad=0, top=True, bottom=False,
                        labeltop=True, labelbottom=False)
    cbar.outline.set_linewidth(0.3)

    # Label: clear title
    total_tasks = len(task_data)
    cbar.ax.set_xlabel(f'Task count (total: {total_tasks:,})', fontsize=8, labelpad=4)

    # Clean axis labels - consistent font size
    ax.set_xlabel('Average Plan Length', fontsize=8, labelpad=1)
    ax.set_ylabel('Safety Effort', fontsize=8, labelpad=1)

    # Set axis limits to align with data (no padding on left)
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)

    # Set clean integer ticks
    ax.set_xticks(np.arange(1, x_max + 1, 3))  # 1, 4, 7, 10, 13, 16, 19, 22, 25
    ax.set_yticks(np.arange(-8, 9, 2))
    ax.tick_params(labelsize=8, width=0.3, length=2, pad=1)

    # Clean spines - thinner for Nature/Science style
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color('black')


    # Save in multiple formats with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', format='png')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white', format='pdf')
    plt.savefig(output_path.with_suffix('.svg'), bbox_inches='tight', facecolor='white', format='svg')
    plt.close()

    print(f"✅ Heatmap saved to {output_path}")


def main(parent_folders: Optional[List[str]] = None):
    """
    Main function to run the distribution analysis.
    
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
    
    print(f"🔄 Collecting task distributions from {len(parent_folders)} folders...")
    print(f"   Folders: {', '.join(parent_folders)}")
    
    # Collect task data
    task_data = collect_task_distributions(parent_folders)
    
    if not task_data:
        print("❌ No task data collected")
        return
    
    print(f"✅ Collected data from {len(task_data)} tasks")
    
    # Create output directory
    output_dir = Path("data/experiments/distribution_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create heatmap
    print("\n🔄 Creating distribution plot (heatmap)...")
    heatmap_path = output_dir / "task_distribution_heatmap.png"
    create_heatmap_plot(task_data, heatmap_path)
    
    # Save raw data as CSV for reference
    df = pd.DataFrame(task_data)
    csv_path = output_dir / "task_distribution_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Raw data saved to {csv_path}")
    
    print(f"\n📊 Distribution analysis complete!")
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

