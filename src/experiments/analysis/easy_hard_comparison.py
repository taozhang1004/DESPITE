#!/usr/bin/env python3
"""
Easy vs Hard Comparison Analysis

Compares benchmark results between sampled/easy-100 and sampled/hard-100 datasets
to analyze how model performance differs across difficulty levels.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def collect_data(parent_folder: str) -> Dict[str, Any]:
    """Collect benchmark results from a folder."""
    results = {}
    pattern = f"{parent_folder}/*/benchmark_results_1.json"
    result_files = glob.glob(pattern)

    for result_file in result_files:
        task_id = Path(result_file).parent.name
        with open(result_file) as f:
            data = json.load(f)
            results[task_id] = data

    return results


def calculate_metrics(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Calculate metrics per model from benchmark results."""
    stats_dict = {}

    for task_id, task_data in data.items():
        if 'models' not in task_data:
            continue

        for model_key, model_data in task_data['models'].items():
            if model_key not in stats_dict:
                stats_dict[model_key] = {
                    'scores': [],
                    'score_2': 0,
                    'score_1': 0,
                    'score_0': 0,
                    'total': 0
                }

            task_types = model_data.get('task_types', {})
            if 'comprehensive_planning' in task_types:
                val = task_types['comprehensive_planning']['validation_result']
                score = val.get('score', 0)

                stats_dict[model_key]['scores'].append(score)
                stats_dict[model_key]['total'] += 1
                if score == 2:
                    stats_dict[model_key]['score_2'] += 1
                if score >= 1:
                    stats_dict[model_key]['score_1'] += 1
                if score == 0:
                    stats_dict[model_key]['score_0'] += 1

    # Calculate rates
    metrics = {}
    for model_key, model_stats in stats_dict.items():
        if model_stats['total'] > 0:
            metrics[model_key] = {
                'safe_rate': model_stats['score_2'] / model_stats['total'] * 100,
                'feasible_rate': model_stats['score_1'] / model_stats['total'] * 100,
                'infeasible_rate': model_stats['score_0'] / model_stats['total'] * 100,
                'scores': model_stats['scores'],
                'total': model_stats['total'],
                'score_2': model_stats['score_2'],
                'score_1': model_stats['score_1'],
                'score_0': model_stats['score_0']
            }

    return metrics


def get_short_model_name(model_key: str) -> str:
    """Get short display name for a model with size info."""
    parts = model_key.split('_', 1)
    if len(parts) < 2:
        return model_key

    provider, model_name = parts

    # Handle specific model names with size info
    if 'Kimi-K2' in model_name:
        return 'Kimi-K2-Instruct (1T)'
    elif 'ministral-14b' in model_name:
        return 'Ministral (14B)'
    elif 'mistral-large' in model_name:
        return 'Mistral-Large (675B)'

    return model_name


def get_full_model_name(model_key: str) -> str:
    """Get full display name for a model (for publication), with size on second line."""
    parts = model_key.split('_', 1)
    if len(parts) < 2:
        return model_key

    provider, model_name = parts

    # Use complete model names with size info
    if 'Kimi-K2-Instruct-0905' in model_name:
        return 'Kimi-K2-Inst\n(1T)'
    elif 'ministral-14b-2512' in model_name:
        return 'Ministral\n(14B)'
    elif 'mistral-large-2512' in model_name:
        return 'Mistral-Large\n(675B)'

    return model_name


def create_publication_figure(easy_metrics: Dict, hard_metrics: Dict, output_dir: Path):
    """
    Create a single publication-quality figure for Nature/Science level journals.

    Two side-by-side stacked bar plots:
    - Panel a: Easy-100 score distribution
    - Panel b: Hard-100 score distribution
    """
    # Publication-quality settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'lines.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'mathtext.fontset': 'dejavusans',
    })

    # Get common models and sort by performance on easy set
    common_models = set(easy_metrics.keys()) & set(hard_metrics.keys())
    if not common_models:
        print("No common models found!")
        return

    # Sort models by hard safe rate (descending)
    models = sorted(common_models, key=lambda m: hard_metrics[m]['safe_rate'], reverse=True)
    n_models = len(models)
    model_names = [get_full_model_name(m) for m in models]

    # Prepare data
    easy_safe = np.array([easy_metrics[m]['safe_rate'] for m in models])
    hard_safe = np.array([hard_metrics[m]['safe_rate'] for m in models])
    easy_unsafe = np.array([easy_metrics[m]['feasible_rate'] - easy_metrics[m]['safe_rate'] for m in models])
    hard_unsafe = np.array([hard_metrics[m]['feasible_rate'] - hard_metrics[m]['safe_rate'] for m in models])
    easy_infeasible = np.array([easy_metrics[m]['infeasible_rate'] for m in models])
    hard_infeasible = np.array([hard_metrics[m]['infeasible_rate'] for m in models])

    # Create figure
    fig = plt.figure(figsize=(7.5, 4.2))

    # Create GridSpec for layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.28,
                  left=0.09, right=0.98, top=0.88, bottom=0.18)

    # Colors - publication-appropriate palette
    colors = {
        'safe': '#2D7D46',      # Forest green
        'unsafe': '#E6A832',    # Amber/gold
        'infeasible': '#C44536' # Muted red
    }

    bar_width = 0.65
    x = np.arange(n_models)

    def draw_stacked_panel(ax, safe_vals, unsafe_vals, infeasible_vals, title, panel_label):
        """Helper function to draw a stacked bar panel."""
        # Draw stacked bars
        p1 = ax.bar(x, safe_vals, bar_width,
                    label='Safe & Feasible', color=colors['safe'],
                    edgecolor='white', linewidth=0.5, zorder=3)
        p2 = ax.bar(x, unsafe_vals, bar_width, bottom=safe_vals,
                    label='Feasible but Unsafe', color=colors['unsafe'],
                    edgecolor='white', linewidth=0.5, zorder=3)
        p3 = ax.bar(x, infeasible_vals, bar_width, bottom=safe_vals + unsafe_vals,
                    label='Infeasible', color=colors['infeasible'],
                    edgecolor='white', linewidth=0.5, zorder=3)

        # Add percentage labels inside bars
        for i in range(n_models):
            # Safe segment (bottom)
            if safe_vals[i] >= 6:
                ax.text(x[i], safe_vals[i]/2, f'{safe_vals[i]:.0f}',
                        ha='center', va='center', fontsize=12,
                        fontweight='bold', color='white')
            # Unsafe segment (middle)
            if unsafe_vals[i] >= 6:
                ax.text(x[i], safe_vals[i] + unsafe_vals[i]/2, f'{unsafe_vals[i]:.0f}',
                        ha='center', va='center', fontsize=12,
                        fontweight='bold', color='white')
            # Infeasible segment (top)
            if infeasible_vals[i] >= 6:
                ax.text(x[i], safe_vals[i] + unsafe_vals[i] + infeasible_vals[i]/2,
                        f'{infeasible_vals[i]:.0f}',
                        ha='center', va='center', fontsize=12,
                        fontweight='bold', color='white')

        ax.set_ylabel('Percentage (%)', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=13, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=0, ha='center', fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5, color='#CCCCCC', zorder=0)
        ax.set_axisbelow(True)

        # Panel label (capitalized)
        ax.text(-0.10, 1.08, panel_label, transform=ax.transAxes, fontsize=18,
                fontweight='bold', va='top', ha='left')

        return p1, p2, p3

    # ========== Panel a: Easy Samples ==========
    ax1 = fig.add_subplot(gs[0, 0])
    p1, p2, p3 = draw_stacked_panel(ax1, easy_safe, easy_unsafe, easy_infeasible,
                                     'Easy Samples (n = 100)', 'a')

    # ========== Panel b: Hard Samples ==========
    ax2 = fig.add_subplot(gs[0, 1])
    draw_stacked_panel(ax2, hard_safe, hard_unsafe, hard_infeasible,
                       'Hard Samples (n = 100)', 'b')
    ax2.set_ylabel('')  # Remove duplicate y-label

    # Add shared legend at top (where main title was)
    # Legend names consistent with general_analysis.py
    handles = [
        plt.Rectangle((0,0), 1, 1, facecolor=colors['safe'], edgecolor='white', linewidth=0.5),
        plt.Rectangle((0,0), 1, 1, facecolor=colors['unsafe'], edgecolor='white', linewidth=0.5),
        plt.Rectangle((0,0), 1, 1, facecolor=colors['infeasible'], edgecolor='white', linewidth=0.5),
    ]
    labels = ['Feasible and Safe', 'Feasible but Unsafe', 'Infeasible']
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=True,
               fancybox=False, edgecolor='#888888', framealpha=0.95,
               fontsize=11, bbox_to_anchor=(0.53, -0.02))

    # Save as SVG (primary), PDF, and PNG
    output_path_svg = output_dir / 'easy_hard_comparison_publication.svg'
    output_path_pdf = output_dir / 'easy_hard_comparison_publication.pdf'
    output_path_png = output_dir / 'easy_hard_comparison_publication.png'

    plt.savefig(output_path_svg, format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Publication figure saved to:")
    print(f"  - {output_path_svg}")
    print(f"  - {output_path_pdf}")
    print(f"  - {output_path_png}")


def create_comparison_visualization(easy_metrics: Dict, hard_metrics: Dict, output_dir: Path):
    """Create visualization comparing easy vs hard results."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.2,
        'figure.dpi': 150
    })

    # Get common models
    common_models = set(easy_metrics.keys()) & set(hard_metrics.keys())
    if not common_models:
        print("No common models found between easy and hard datasets!")
        return

    models = sorted(common_models)
    n_models = len(models)

    # Prepare data
    easy_safe = [easy_metrics[m]['safe_rate'] for m in models]
    hard_safe = [hard_metrics[m]['safe_rate'] for m in models]
    easy_feasible = [easy_metrics[m]['feasible_rate'] for m in models]
    hard_feasible = [hard_metrics[m]['feasible_rate'] for m in models]

    model_names = [get_short_model_name(m) for m in models]

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(n_models)
    width = 0.35

    # Colors
    easy_color = '#4CAF50'  # Green
    hard_color = '#F44336'  # Red

    # Plot 1: Safe & Feasible Rate (Score = 2)
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, easy_safe, width, label='Easy-100', color=easy_color, alpha=0.8)
    bars2 = ax1.bar(x + width/2, hard_safe, width, label='Hard-100', color=hard_color, alpha=0.8)

    ax1.set_ylabel('Rate (%)', fontsize=12)
    ax1.set_title('Safe & Feasible Rate (Score = 2)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=15, ha='right')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # Plot 2: Feasibility Rate (Score >= 1)
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, easy_feasible, width, label='Easy-100', color=easy_color, alpha=0.8)
    bars4 = ax2.bar(x + width/2, hard_feasible, width, label='Hard-100', color=hard_color, alpha=0.8)

    ax2.set_ylabel('Rate (%)', fontsize=12)
    ax2.set_title('Feasibility Rate (Score >= 1)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=15, ha='right')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save
    output_path = output_dir / 'easy_hard_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'easy_hard_comparison.pdf', bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Comparison plot saved to {output_path}")

    # Create stacked bar chart showing score distribution
    fig2, ax = plt.subplots(figsize=(12, 6))

    # Prepare stacked data
    bar_width = 0.35
    x_pos = np.arange(n_models * 2)

    # Interleave easy and hard for each model
    labels = []
    score_2_vals = []
    score_1_only_vals = []  # Score 1 but not 2 (feasible but unsafe)
    score_0_vals = []

    for m in models:
        short_name = get_short_model_name(m)
        # Easy
        labels.append(f'{short_name}\n(Easy)')
        score_2_vals.append(easy_metrics[m]['safe_rate'])
        score_1_only_vals.append(easy_metrics[m]['feasible_rate'] - easy_metrics[m]['safe_rate'])
        score_0_vals.append(easy_metrics[m]['infeasible_rate'])
        # Hard
        labels.append(f'{short_name}\n(Hard)')
        score_2_vals.append(hard_metrics[m]['safe_rate'])
        score_1_only_vals.append(hard_metrics[m]['feasible_rate'] - hard_metrics[m]['safe_rate'])
        score_0_vals.append(hard_metrics[m]['infeasible_rate'])

    # Stacked bars
    x_pos = np.arange(len(labels))
    colors_stack = ['#1E6336', '#A07800', '#A50303']  # Green, Yellow, Red

    p1 = ax.bar(x_pos, score_2_vals, width=0.6, label='Safe & Feasible (Score=2)', color=colors_stack[0])
    p2 = ax.bar(x_pos, score_1_only_vals, width=0.6, bottom=score_2_vals,
                label='Feasible but Unsafe (Score=1)', color=colors_stack[1])
    p3 = ax.bar(x_pos, score_0_vals, width=0.6,
                bottom=np.array(score_2_vals) + np.array(score_1_only_vals),
                label='Infeasible (Score=0)', color=colors_stack[2])

    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title('Score Distribution: Easy-100 vs Hard-100', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add alternating background colors to distinguish models
    for i in range(n_models):
        if i % 2 == 0:
            ax.axvspan(i*2 - 0.5, i*2 + 1.5, alpha=0.1, color='gray')

    plt.tight_layout()

    output_path2 = output_dir / 'easy_hard_stacked.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'easy_hard_stacked.pdf', bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Stacked distribution plot saved to {output_path2}")


def save_results_to_txt(easy_metrics: Dict, hard_metrics: Dict, output_path: Path):
    """Save statistical comparison to a text file."""
    lines = []
    lines.append("=" * 70)
    lines.append("EASY vs HARD COMPARISON ANALYSIS")
    lines.append("=" * 70)
    lines.append("")

    common_models = set(easy_metrics.keys()) & set(hard_metrics.keys())

    for model in sorted(common_models, key=lambda m: hard_metrics[m]['safe_rate'], reverse=True):
        lines.append(f"--- {get_short_model_name(model)} ---")

        easy = easy_metrics[model]
        hard = hard_metrics[model]

        lines.append(f"  Easy-100 (n={easy['total']}):")
        lines.append(f"    Safe & Feasible Rate: {easy['safe_rate']:.1f}% ({easy['score_2']}/{easy['total']})")
        lines.append(f"    Feasibility Rate:     {easy['feasible_rate']:.1f}% ({easy['score_1']}/{easy['total']})")
        lines.append(f"    Infeasible Rate:      {easy['infeasible_rate']:.1f}% ({easy['score_0']}/{easy['total']})")

        lines.append(f"  Hard-100 (n={hard['total']}):")
        lines.append(f"    Safe & Feasible Rate: {hard['safe_rate']:.1f}% ({hard['score_2']}/{hard['total']})")
        lines.append(f"    Feasibility Rate:     {hard['feasible_rate']:.1f}% ({hard['score_1']}/{hard['total']})")
        lines.append(f"    Infeasible Rate:      {hard['infeasible_rate']:.1f}% ({hard['score_0']}/{hard['total']})")

        safe_diff = easy['safe_rate'] - hard['safe_rate']
        feasible_diff = easy['feasible_rate'] - hard['feasible_rate']

        lines.append(f"  Difference (Easy - Hard):")
        lines.append(f"    Safe Rate Diff:       {safe_diff:+.1f}%")
        lines.append(f"    Feasible Rate Diff:   {feasible_diff:+.1f}%")

        if easy['total'] > 0 and hard['total'] > 0:
            contingency_safe = [
                [easy['score_2'], easy['total'] - easy['score_2']],
                [hard['score_2'], hard['total'] - hard['score_2']]
            ]
            try:
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_safe)
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                lines.append(f"  Chi-square test (Safe Rate): chi2={chi2:.2f}, p={p_value:.4f} {significance}")
            except:
                lines.append(f"  Chi-square test: Unable to compute (insufficient data)")

        lines.append("")

    lines.append("=" * 70)
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 70)

    all_easy_safe = np.mean([easy_metrics[m]['safe_rate'] for m in common_models])
    all_hard_safe = np.mean([hard_metrics[m]['safe_rate'] for m in common_models])
    all_easy_feasible = np.mean([easy_metrics[m]['feasible_rate'] for m in common_models])
    all_hard_feasible = np.mean([hard_metrics[m]['feasible_rate'] for m in common_models])

    lines.append("")
    lines.append(f"Average across all models:")
    lines.append(f"  Easy-100 Avg Safe Rate:     {all_easy_safe:.1f}%")
    lines.append(f"  Hard-100 Avg Safe Rate:     {all_hard_safe:.1f}%")
    lines.append(f"  Difference:                 {all_easy_safe - all_hard_safe:+.1f}%")
    lines.append(f"")
    lines.append(f"  Easy-100 Avg Feasible Rate: {all_easy_feasible:.1f}%")
    lines.append(f"  Hard-100 Avg Feasible Rate: {all_hard_feasible:.1f}%")
    lines.append(f"  Difference:                 {all_easy_feasible - all_hard_feasible:+.1f}%")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Results saved to {output_path}")


def print_statistical_analysis(easy_metrics: Dict, hard_metrics: Dict):
    """Print statistical comparison between easy and hard datasets."""
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS: Easy-100 vs Hard-100")
    print("=" * 70)

    common_models = set(easy_metrics.keys()) & set(hard_metrics.keys())

    for model in sorted(common_models):
        print(f"\n--- {get_short_model_name(model)} ---")

        easy = easy_metrics[model]
        hard = hard_metrics[model]

        print(f"  Easy-100 (n={easy['total']}):")
        print(f"    Safe & Feasible Rate: {easy['safe_rate']:.1f}% ({easy['score_2']}/{easy['total']})")
        print(f"    Feasibility Rate:     {easy['feasible_rate']:.1f}% ({easy['score_1']}/{easy['total']})")
        print(f"    Infeasible Rate:      {easy['infeasible_rate']:.1f}% ({easy['score_0']}/{easy['total']})")

        print(f"  Hard-100 (n={hard['total']}):")
        print(f"    Safe & Feasible Rate: {hard['safe_rate']:.1f}% ({hard['score_2']}/{hard['total']})")
        print(f"    Feasibility Rate:     {hard['feasible_rate']:.1f}% ({hard['score_1']}/{hard['total']})")
        print(f"    Infeasible Rate:      {hard['infeasible_rate']:.1f}% ({hard['score_0']}/{hard['total']})")

        # Difference
        safe_diff = easy['safe_rate'] - hard['safe_rate']
        feasible_diff = easy['feasible_rate'] - hard['feasible_rate']

        print(f"  Difference (Easy - Hard):")
        print(f"    Safe Rate Diff:       {safe_diff:+.1f}%")
        print(f"    Feasible Rate Diff:   {feasible_diff:+.1f}%")

        # Chi-square test for safe rate
        if easy['total'] > 0 and hard['total'] > 0:
            # Contingency table for safe vs not safe
            contingency_safe = [
                [easy['score_2'], easy['total'] - easy['score_2']],
                [hard['score_2'], hard['total'] - hard['score_2']]
            ]
            try:
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_safe)
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"  Chi-square test (Safe Rate): chi2={chi2:.2f}, p={p_value:.4f} {significance}")
            except:
                print(f"  Chi-square test: Unable to compute (insufficient data)")

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    all_easy_safe = np.mean([easy_metrics[m]['safe_rate'] for m in common_models])
    all_hard_safe = np.mean([hard_metrics[m]['safe_rate'] for m in common_models])
    all_easy_feasible = np.mean([easy_metrics[m]['feasible_rate'] for m in common_models])
    all_hard_feasible = np.mean([hard_metrics[m]['feasible_rate'] for m in common_models])

    print(f"\nAverage across all models:")
    print(f"  Easy-100 Avg Safe Rate:     {all_easy_safe:.1f}%")
    print(f"  Hard-100 Avg Safe Rate:     {all_hard_safe:.1f}%")
    print(f"  Difference:                 {all_easy_safe - all_hard_safe:+.1f}%")
    print(f"\n  Easy-100 Avg Feasible Rate: {all_easy_feasible:.1f}%")
    print(f"  Hard-100 Avg Feasible Rate: {all_hard_feasible:.1f}%")
    print(f"  Difference:                 {all_easy_feasible - all_hard_feasible:+.1f}%")


def main():
    """Main function."""
    base_path = Path("data/sampled")

    easy_folder = base_path / "easy-100"
    hard_folder = base_path / "hard-100"

    print("Collecting data from sampled/easy-100...")
    easy_data = collect_data(str(easy_folder))
    print(f"  Found {len(easy_data)} tasks")

    print("Collecting data from sampled/hard-100...")
    hard_data = collect_data(str(hard_folder))
    print(f"  Found {len(hard_data)} tasks")

    print("\nCalculating metrics...")
    easy_metrics = calculate_metrics(easy_data)
    hard_metrics = calculate_metrics(hard_data)

    print(f"  Easy models: {list(easy_metrics.keys())}")
    print(f"  Hard models: {list(hard_metrics.keys())}")

    # Print statistical analysis
    print_statistical_analysis(easy_metrics, hard_metrics)

    # Create output directory
    output_dir = Path("data/experiments/easy_hard_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results to txt file
    results_path = output_dir / "results_easy_hard.txt"
    save_results_to_txt(easy_metrics, hard_metrics, results_path)

    # Create visualizations
    print("\nCreating visualizations...")
    create_comparison_visualization(easy_metrics, hard_metrics, output_dir)

    # Create publication-quality figure
    print("\nCreating publication-quality figure...")
    create_publication_figure(easy_metrics, hard_metrics, output_dir)

    print(f"\nAnalysis complete! Output saved to {output_dir}")


if __name__ == "__main__":
    main()
