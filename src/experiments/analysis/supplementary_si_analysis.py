#!/usr/bin/env python3
"""
Supplementary Analysis: Safety Intention (SI)

Creates scatter plots matching the exact style of scale_analysis.py with a 2x3 layout:
- Top row: Feasibility vs Size, Safety vs Size, Safety vs Feasibility (standard metrics)
- Bottom row: SI vs Safety, SI vs Model Size, SI vs Feasibility (SI metrics)

Safety Intention measures whether a plan intends to be safe, regardless of feasibility.
This allows vertical comparison of columns 2 and 3 between standard and SI metrics.
"""
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from adjustText import adjust_text
import matplotlib.patheffects as path_effects
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_model_info() -> Dict[str, Dict[str, Any]]:
    """
    Define model information including size, family, and release date.
    Matches scale_analysis.py exactly.
    """
    model_info = {}

    # DeepSeek models - V3.2-Exp released September 29, 2025
    model_info['deepseek_deepseek-chat'] = {
        'size': 685,
        'family': 'DeepSeek',
        'display_name': 'V3.2-Exp (685B)',
        'release_date': '2025-09-29'
    }
    model_info['deepseek_deepseek-reasoner'] = {
        'size': 685,
        'family': 'DeepSeek',
        'display_name': 'V3.2-Exp-T (685B)',
        'release_date': '2025-09-29'
    }

    # Qwen models
    model_info['together_Qwen/Qwen2.5-7B-Instruct-Turbo'] = {
        'size': 7,
        'family': 'Qwen',
        'display_name': '2.5-7B',
        'release_date': '2024-09-19'
    }
    model_info['together_Qwen/Qwen2.5-72B-Instruct-Turbo'] = {
        'size': 72,
        'family': 'Qwen',
        'display_name': '2.5-72B',
        'release_date': '2024-09-19'
    }
    model_info['together_Qwen/Qwen2.5-Coder-32B-Instruct'] = {
        'size': 32,
        'family': 'Qwen',
        'display_name': '2.5-Coder-32B',
        'release_date': '2024-11-12'
    }
    model_info['together_Qwen/QwQ-32B'] = {
        'size': 32,
        'family': 'Qwen',
        'display_name': 'QwQ-32B',
        'release_date': '2024-11-28'
    }
    model_info['together_Qwen/Qwen3-235B-A22B-Instruct-2507-tput'] = {
        'size': 235,
        'family': 'Qwen',
        'display_name': '3-235B',
        'release_date': '2025-04-28'
    }
    model_info['together_Qwen/Qwen3-235B-A22B-Thinking-2507'] = {
        'size': 235,
        'family': 'Qwen',
        'display_name': '3-235B-T',
        'release_date': '2025-04-28'
    }
    model_info['together_Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8'] = {
        'size': 480,
        'family': 'Qwen',
        'display_name': '3-Coder-480B',
        'release_date': '2025-07-22'
    }
    model_info['together_Qwen/Qwen3-Next-80B-A3B-Instruct'] = {
        'size': 80,
        'family': 'Qwen',
        'display_name': '3-Next-80B',
        'release_date': '2025-09-10'
    }
    model_info['together_Qwen/Qwen3-Next-80B-A3B-Thinking'] = {
        'size': 80,
        'family': 'Qwen',
        'display_name': '3-Next-80B-T',
        'release_date': '2025-09-10'
    }

    # Meta Llama models
    model_info['together_meta-llama/Meta-Llama-3-8B-Instruct-Lite'] = {
        'size': 8,
        'family': 'Llama',
        'display_name': '3-8B',
        'release_date': '2024-04-18'
    }
    model_info['together_meta-llama/Meta-Llama-3-70B-Instruct-Turbo'] = {
        'size': 70,
        'family': 'Llama',
        'display_name': '3-70B',
        'release_date': '2024-04-18'
    }
    model_info['together_meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'] = {
        'size': 8,
        'family': 'Llama',
        'display_name': '3.1-8B',
        'release_date': '2024-07-23'
    }
    model_info['together_meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'] = {
        'size': 70,
        'family': 'Llama',
        'display_name': '3.1-70B',
        'release_date': '2024-07-23'
    }
    model_info['together_meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo'] = {
        'size': 405,
        'family': 'Llama',
        'display_name': '3.1-405B',
        'release_date': '2024-07-23'
    }
    model_info['together_meta-llama/Llama-3.2-3B-Instruct-Turbo'] = {
        'size': 3,
        'family': 'Llama',
        'display_name': '3.2-3B',
        'release_date': '2024-09-25'
    }
    model_info['together_meta-llama/Llama-3.3-70B-Instruct-Turbo'] = {
        'size': 70,
        'family': 'Llama',
        'display_name': '3.3-70B',
        'release_date': '2024-12-06'
    }
    model_info['together_meta-llama/Llama-4-Scout-17B-16E-Instruct'] = {
        'size': 17,
        'family': 'Llama',
        'display_name': '4-Scout (17B)',
        'release_date': '2025-04-05'
    }
    model_info['together_meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'] = {
        'size': 17,
        'family': 'Llama',
        'display_name': '4-Maverick (17B)',
        'release_date': '2025-04-05'
    }

    # OpenAI OSS models - released August 5, 2025
    model_info['together_openai/gpt-oss-20b'] = {
        'size': 20,
        'family': 'OpenAI',
        'display_name': 'GPT-OSS-20B',
        'release_date': '2025-08-05'
    }
    model_info['together_openai/gpt-oss-120b'] = {
        'size': 120,
        'family': 'OpenAI',
        'display_name': 'GPT-OSS-120B',
        'release_date': '2025-08-05'
    }

    return model_info


def collect_data(parent_folders: List[str], selected_models: Optional[List[str]] = None) -> Dict[str, Any]:
    """Collect benchmark results from task folders."""
    results = {}

    for parent_folder in parent_folders:
        pattern = f"{parent_folder}/*/benchmark_results_1.json"
        result_files = glob.glob(pattern)

        for result_file in result_files:
            task_id = Path(result_file).parent.name

            with open(result_file) as f:
                data = json.load(f)

                if selected_models is not None:
                    if 'models' in data:
                        filtered_models = {
                            model_key: model_data
                            for model_key, model_data in data['models'].items()
                            if model_key in selected_models
                        }
                        data['models'] = filtered_models

                results[task_id] = data

    return results


def calculate_metrics(data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Calculate metrics including safety intention from benchmark results."""
    stats_dict = {}

    for task_id, task_data in data.items():
        if 'models' not in task_data:
            continue

        for model_key, model_data in task_data['models'].items():
            if model_key not in stats_dict:
                stats_dict[model_key] = {
                    'score_2': 0, 'score_1': 0, 'safety_intention': 0, 'total': 0
                }

            task_types = model_data.get('task_types', {})

            if 'comprehensive_planning' in task_types:
                stats_dict[model_key]['total'] += 1
                val = task_types['comprehensive_planning']['validation_result']
                score = val.get('score', 0)
                has_safety_intention = val.get('safety_intention', False)

                if score == 2:
                    stats_dict[model_key]['score_2'] += 1
                if score >= 1:
                    stats_dict[model_key]['score_1'] += 1
                if has_safety_intention:
                    stats_dict[model_key]['safety_intention'] += 1

    # Calculate final metrics
    final_metrics = {}

    for model_key, model_stats in stats_dict.items():
        if model_stats['total'] > 0:
            total = model_stats['total']
            final_metrics[model_key] = {
                'safe_rate': model_stats['score_2'] / total,
                'feasible_rate': model_stats['score_1'] / total,
                'si_rate': model_stats['safety_intention'] / total,
                'total': total
            }

    return final_metrics


def format_r2(r2):
    """Format R² value - show more precision for small values."""
    if r2 < 0.01:
        return f'R²={r2:.4f}'
    else:
        return f'R²={r2:.2f}'


def apply_axis_style(ax, fontsize=14):
    """Apply consistent Nature/Science style to an axis."""
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(0.5)

    ax.tick_params(axis='both', which='both', direction='out', length=3, width=0.5,
                   labelsize=fontsize, pad=3, colors='black',
                   bottom=True, left=True, top=False, right=False)

    ax.grid(axis='y', linestyle='--', alpha=0.4, color='gray', linewidth=0.3)
    ax.grid(axis='x', visible=False)
    ax.set_axisbelow(True)


def create_supplementary_plots(metrics: Dict[str, Dict[str, float]],
                                model_info: Dict[str, Dict[str, Any]],
                                output_path: Path):
    """Create publication-quality scatter plots (Nature/Science style) with SI metrics."""
    # Prepare data
    plot_data = []
    for model_key, metric_data in metrics.items():
        if model_key in model_info and metric_data.get('total', 0) > 0:
            plot_data.append({
                'model_key': model_key,
                'size': model_info[model_key]['size'],
                'family': model_info[model_key]['family'],
                'display_name': model_info[model_key]['display_name'],
                'release_date': model_info[model_key].get('release_date', '2099-01-01'),
                'safe_rate': metric_data['safe_rate'] * 100,
                'feasible_rate': metric_data['feasible_rate'] * 100,
                'si_rate': metric_data['si_rate'] * 100
            })

    if not plot_data:
        print("No data to plot!")
        return

    # Normalize safety rates
    safe_rates_raw = np.array([d['safe_rate'] for d in plot_data])
    max_safe_rate = safe_rates_raw.max()
    for d in plot_data:
        d['safe_rate_normalized'] = d['safe_rate'] / max_safe_rate if max_safe_rate > 0 else 0

    # Muted color palette matching scale_analysis style
    families = sorted(set(d['family'] for d in plot_data))
    family_colors = {
        'DeepSeek': '#4477AA',
        'Llama': '#228833',
        'OpenAI': '#CC6677',
        'Qwen': '#DDCC77'
    }
    family_markers = {
        'DeepSeek': 'o',
        'Llama': 'D',
        'OpenAI': 's',
        'Qwen': 'o'
    }

    # Assign numbers by GLOBAL release date (matching scale_analysis.py exactly)
    date_groups = defaultdict(list)
    for d in plot_data:
        date_groups[d['release_date']].append(d)

    sorted_dates = sorted(date_groups.keys())
    model_labels = {}
    legend_entries = []
    current_number = 1

    for date in sorted_dates:
        models_on_date = sorted(date_groups[date], key=lambda x: (x['family'], x['display_name']))
        if len(models_on_date) == 1:
            label = str(current_number)
            model_labels[models_on_date[0]['model_key']] = label
            legend_entries.append((label, models_on_date[0]['display_name'], models_on_date[0]['family']))
        else:
            for i, model in enumerate(models_on_date):
                letter = chr(ord('a') + i)
                label = f"{current_number}{letter}"
                model_labels[model['model_key']] = label
                legend_entries.append((label, model['display_name'], model['family']))
        current_number += 1

    for d in plot_data:
        d['label'] = model_labels[d['model_key']]

    # Darker colors for better contrast with white text
    family_colors_dark = {
        'DeepSeek': '#2c5aa0',
        'Llama': '#1a6622',
        'OpenAI': '#a84a5a',
        'Qwen': '#b8a030'
    }

    # Create figure with 2 rows of plots + legend space at bottom
    fs = 14
    fig = plt.figure(figsize=(18, 12.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.35], hspace=0.25, wspace=0.20)

    # Top row: standard metrics (matching scale_analysis.py)
    ax1 = fig.add_subplot(gs[0, 0])  # Feasibility vs Size
    ax2 = fig.add_subplot(gs[0, 1])  # Safety vs Size
    ax3 = fig.add_subplot(gs[0, 2])  # Safety vs Feasibility

    # Bottom row: SI metrics (for vertical comparison)
    ax4 = fig.add_subplot(gs[1, 0])  # SI vs Safety
    ax5 = fig.add_subplot(gs[1, 1])  # SI vs Size
    ax6 = fig.add_subplot(gs[1, 2])  # SI vs Feasibility

    ax_legend = fig.add_subplot(gs[2, :])
    fig.patch.set_facecolor('white')

    sizes = np.array([d['size'] for d in plot_data])
    safe_rates = np.array([d['safe_rate'] for d in plot_data])
    feasible_rates = np.array([d['feasible_rate'] for d in plot_data])
    si_rates = np.array([d['si_rate'] for d in plot_data])
    safe_rates_normalized_scaled = np.array([d['safe_rate_normalized'] for d in plot_data]) * max_safe_rate

    x_min, x_max = sizes.min(), sizes.max()

    # Manual overrides for label positions (may need tuning)
    manual_overrides = {
        0: {  # Feasibility vs Size
            '4': {'length': 20},
            '7': {'length': 20},
            '2a': {'angle': 180, 'length': 30},
            '2b': {'angle': 120, 'length': 30},
            '6': {'length': 20},
            '8b': {'length': 30},
            '12b': {'length': 5},
            '13a': {'angle': 0},
            '13b': {'angle': 0},
        },
        1: {  # Safety vs Size
            '7': {'length': 20},
            '1a': {'length': 20},
            '2a': {'angle': 120},
            '10': {'angle': -120, 'length': 20},
            '11a': {'angle': -90, 'length': 20},
        },
        2: {  # Safety vs Feasibility
            '2c': {'angle': 120, 'length': 10},
            '2a': {'angle': 0, 'length': 10},
            '3a': {'angle': 180, 'length': 30},
            '11b': {'angle': -120, 'length': 20},
            '9a': {'angle': -120, 'length': 20},
            '9b': {'angle': -90, 'length': 20},
            '12b': {'angle': 90},
            '1a': {'angle': 180, 'length': 30},
            '8b': {'angle': 180, 'length': 30},
            '2b': {'angle': -45, 'length': 10},
            '13b': {'angle': 180, 'length': 30},
            '7': {'angle': -90, 'length': 20},
        },
        3: {},  # SI vs Safety
        4: {},  # SI vs Size
        5: {},  # SI vs Feasibility
    }

    def plot_scatter(ax, x_func, y_func, subplot_idx=0):
        """Plot scatter with fixed-length label offsets (matching scale_analysis.py)."""
        for family in families:
            fam_data = [d for d in plot_data if d['family'] == family]
            if not fam_data:
                continue
            xs = [x_func(d) for d in fam_data]
            ys = [y_func(d) for d in fam_data]
            color = family_colors_dark.get(family, '#808080')
            marker = family_markers.get(family, 'o')

            if family == 'DeepSeek':
                ax.scatter(xs, ys, s=70, alpha=0.9,
                          facecolors='none',
                          edgecolors=color, linewidths=1.5,
                          marker=marker,
                          label=family, zorder=3)
            else:
                ax.scatter(xs, ys, s=70, alpha=0.9,
                          color=color,
                          marker=marker,
                          edgecolors='white', linewidths=0.4,
                          label=family, zorder=3)

        outline = [path_effects.Stroke(linewidth=1.5, foreground='white'),
                   path_effects.Normal()]

        texts = []
        point_coords = []
        data_refs = []
        for d in plot_data:
            x, y = x_func(d), y_func(d)
            color = family_colors_dark.get(d['family'], '#333333')
            txt = ax.text(x, y, d['label'], fontsize=fs,
                         color=color, fontfamily='sans-serif', zorder=5,
                         path_effects=outline)
            texts.append(txt)
            point_coords.append((x, y))
            data_refs.append(d)

        adjust_text(texts, ax=ax,
                   expand_points=(5.0, 5.0),
                   expand_text=(4.0, 4.0),
                   force_text=(5.0, 5.0),
                   force_points=(3.0, 3.0),
                   lim=3000)

        min_len = 12
        max_len = 25
        transform = ax.transData

        for txt, (orig_x, orig_y), d in zip(texts, point_coords, data_refs):
            txt_pos = txt.get_position()
            orig_disp = transform.transform((orig_x, orig_y))
            txt_disp = transform.transform(txt_pos)
            dx_disp = txt_disp[0] - orig_disp[0]
            dy_disp = txt_disp[1] - orig_disp[1]
            dist_disp = (dx_disp**2 + dy_disp**2)**0.5

            txt.remove()

            if dist_disp > 1:
                norm_dx = dx_disp / dist_disp
                norm_dy = dy_disp / dist_disp
                edge_len = max(min_len, min(max_len, dist_disp / 4))
            else:
                norm_dx, norm_dy = 0.707, 0.707
                edge_len = min_len

            label = d['label']
            if subplot_idx in manual_overrides and label in manual_overrides[subplot_idx]:
                override = manual_overrides[subplot_idx][label]
                if 'angle' in override:
                    angle_rad = np.radians(override['angle'])
                    norm_dx = np.cos(angle_rad)
                    norm_dy = np.sin(angle_rad)
                if 'length' in override:
                    edge_len = override['length']

            color = family_colors_dark.get(d['family'], '#333333')
            ax.annotate(d['label'], xy=(orig_x, orig_y),
                       xytext=(edge_len * norm_dx, edge_len * norm_dy),
                       textcoords='offset points', fontsize=fs,
                       color=color, fontfamily='sans-serif', zorder=5,
                       path_effects=outline,
                       arrowprops=dict(arrowstyle='-', color='gray', lw=0.5,
                                      shrinkA=0, shrinkB=1))

    pad = 5

    # ==================== TOP ROW ====================
    # Plot 1: Feasibility Rate vs Model Size
    plot_scatter(ax1, lambda d: d['size'], lambda d: d['feasible_rate'], subplot_idx=0)
    r2_1 = None
    if len(sizes) > 1:
        log_sizes = np.log10(sizes)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, feasible_rates)
        r2_1 = r_value**2
        x_line = np.logspace(np.log10(x_min), np.log10(x_max), 100)
        y_line = slope * np.log10(x_line) + intercept
        ax1.plot(x_line, y_line, color='black', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)

    ax1.set_xlabel('Model Size (B)', fontsize=fs, labelpad=3)
    ax1.set_ylabel('Feasibility Rate (%)', fontsize=fs, labelpad=3)
    ax1.set_xscale('log')
    ax1.set_xlim(x_min * 0.4, x_max * 2.5)
    ax1.set_ylim(-pad, 100 + pad)
    apply_axis_style(ax1, fontsize=fs)
    if r2_1 is not None:
        ax1.text(0.95, 0.05, format_r2(r2_1), transform=ax1.transAxes, fontsize=fs,
                ha='right', va='bottom', style='italic')

    # Plot 2: Safety Rate vs Model Size
    plot_scatter(ax2, lambda d: d['size'], lambda d: d['safe_rate'], subplot_idx=1)
    r2_2 = None
    if len(sizes) > 1:
        log_sizes = np.log10(sizes)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, safe_rates)
        r2_2 = r_value**2
        x_line = np.logspace(np.log10(x_min), np.log10(x_max), 100)
        y_line = slope * np.log10(x_line) + intercept
        ax2.plot(x_line, y_line, color='black', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)

    ax2.set_xlabel('Model Size (B)', fontsize=fs, labelpad=3)
    ax2.set_ylabel('Safety Rate (%)', fontsize=fs, labelpad=3)
    ax2.set_xscale('log')
    ax2.set_xlim(x_min * 0.4, x_max * 2.5)
    ax2.set_ylim(-pad, 100 + pad)
    apply_axis_style(ax2, fontsize=fs)
    if r2_2 is not None:
        ax2.text(0.95, 0.05, format_r2(r2_2), transform=ax2.transAxes, fontsize=fs,
                ha='right', va='bottom', style='italic')

    # Plot 3: Safety Rate vs Feasibility Rate
    plot_scatter(ax3, lambda d: d['feasible_rate'], lambda d: d['safe_rate_normalized'] * max_safe_rate, subplot_idx=2)
    r2_3 = None
    if len(feasible_rates) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(feasible_rates, safe_rates_normalized_scaled)
        r2_3 = r_value**2
        x_line = np.linspace(0, 100, 100)
        y_line = slope * x_line + intercept
        ax3.plot(x_line, y_line, color='black', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)

    ax3.set_xlabel('Feasibility Rate (%)', fontsize=fs, labelpad=3)
    ax3.set_ylabel('Safety Rate (%)', fontsize=fs, labelpad=3)
    y_pad = max_safe_rate * 0.05
    ax3.set_xlim(-pad, 100 + pad)
    ax3.set_ylim(-y_pad, max_safe_rate + y_pad)
    apply_axis_style(ax3, fontsize=fs)
    if r2_3 is not None:
        ax3.text(0.95, 0.05, format_r2(r2_3), transform=ax3.transAxes, fontsize=fs,
                ha='right', va='bottom', style='italic')

    # ==================== BOTTOM ROW (SI metrics) ====================
    # Plot 4: SI Rate vs Safety Rate
    plot_scatter(ax4, lambda d: d['safe_rate'], lambda d: d['si_rate'], subplot_idx=3)
    r2_4 = None
    if len(safe_rates) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(safe_rates, si_rates)
        r2_4 = r_value**2
        x_line = np.linspace(0, max_safe_rate, 100)
        y_line = slope * x_line + intercept
        ax4.plot(x_line, y_line, color='black', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)

    ax4.set_xlabel('Safety Rate (%)', fontsize=fs, labelpad=3)
    ax4.set_ylabel('Safety Intention Rate (%)', fontsize=fs, labelpad=3)
    ax4.set_xlim(-y_pad, max_safe_rate + y_pad)
    ax4.set_ylim(-pad, 100 + pad)
    apply_axis_style(ax4, fontsize=fs)
    if r2_4 is not None:
        ax4.text(0.95, 0.05, format_r2(r2_4), transform=ax4.transAxes, fontsize=fs,
                ha='right', va='bottom', style='italic')

    # Plot 5: SI Rate vs Model Size (compare with ax2 above)
    plot_scatter(ax5, lambda d: d['size'], lambda d: d['si_rate'], subplot_idx=4)
    r2_5 = None
    if len(sizes) > 1:
        log_sizes = np.log10(sizes)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, si_rates)
        r2_5 = r_value**2
        x_line = np.logspace(np.log10(x_min), np.log10(x_max), 100)
        y_line = slope * np.log10(x_line) + intercept
        ax5.plot(x_line, y_line, color='black', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)

    ax5.set_xlabel('Model Size (B)', fontsize=fs, labelpad=3)
    ax5.set_ylabel('Safety Intention Rate (%)', fontsize=fs, labelpad=3)
    ax5.set_xscale('log')
    ax5.set_xlim(x_min * 0.4, x_max * 2.5)
    ax5.set_ylim(-pad, 100 + pad)
    apply_axis_style(ax5, fontsize=fs)
    if r2_5 is not None:
        ax5.text(0.95, 0.05, format_r2(r2_5), transform=ax5.transAxes, fontsize=fs,
                ha='right', va='bottom', style='italic')

    # Plot 6: SI Rate vs Feasibility Rate (compare with ax3 above)
    plot_scatter(ax6, lambda d: d['feasible_rate'], lambda d: d['si_rate'], subplot_idx=5)
    r2_6 = None
    if len(feasible_rates) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(feasible_rates, si_rates)
        r2_6 = r_value**2
        x_line = np.linspace(0, 100, 100)
        y_line = slope * x_line + intercept
        ax6.plot(x_line, y_line, color='black', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)

    ax6.set_xlabel('Feasibility Rate (%)', fontsize=fs, labelpad=3)
    ax6.set_ylabel('Safety Intention Rate (%)', fontsize=fs, labelpad=3)
    ax6.set_xlim(-pad, 100 + pad)
    ax6.set_ylim(-pad, 100 + pad)
    apply_axis_style(ax6, fontsize=fs)
    if r2_6 is not None:
        ax6.text(0.95, 0.05, format_r2(r2_6), transform=ax6.transAxes, fontsize=fs,
                ha='right', va='bottom', style='italic')

    # ==================== SUBPLOT LABELS (Nature style) ====================
    for ax, label in zip([ax1, ax2, ax3, ax4, ax5, ax6], ['a', 'b', 'c', 'd', 'e', 'f']):
        ax.text(-0.08, 1.02, label, transform=ax.transAxes, fontsize=24,
                fontweight='bold', va='bottom', ha='left', clip_on=False)

    # ==================== LEGEND PANEL ====================
    ax_legend.axis('off')

    legend_by_family = {}
    for label, name, family in legend_entries:
        if family not in legend_by_family:
            legend_by_family[family] = []
        legend_by_family[family].append((label, name))

    marker_char = {'o': '●', 'D': '◆', 's': '■'}
    marker_char_hollow = {'DeepSeek': '○'}
    line_height = 0.25

    def draw_family_single_row(family, y_start, x_start=0.01):
        if family not in legend_by_family:
            return y_start
        color = family_colors_dark.get(family, '#333333')
        marker = family_markers[family]
        symbol = marker_char_hollow.get(family, marker_char.get(marker, '●'))

        entries = legend_by_family[family]
        y_pos = y_start

        ax_legend.text(x_start, y_pos, f"{symbol} {family}:",
                      transform=ax_legend.transAxes,
                      fontsize=fs, fontweight='bold', ha='left', va='top',
                      color=color, fontfamily='sans-serif')

        char_width = 0.0065
        entry_gap = 0.015

        x_pos = x_start + 0.07
        for label, name in entries:
            combined = f"{label}: {name}"
            ax_legend.text(x_pos, y_pos, f"{label}:",
                          transform=ax_legend.transAxes, fontsize=fs,
                          ha='left', va='top', color=color,
                          fontfamily='sans-serif')

            index_str = f"{label}: "
            index_width = len(index_str) * char_width
            ax_legend.text(x_pos + index_width, y_pos, f"{name}",
                          transform=ax_legend.transAxes, fontsize=fs,
                          ha='left', va='top', color='black',
                          fontfamily='sans-serif')

            x_pos += len(combined) * char_width + entry_gap

        return y_pos - line_height

    # Row 1: Llama
    y_pos = draw_family_single_row('Llama', 0.95)

    # Row 2: Qwen
    y_pos = draw_family_single_row('Qwen', y_pos)

    # Row 3: DeepSeek + OpenAI
    color_ds = family_colors_dark.get('DeepSeek', '#333333')
    color_oa = family_colors_dark.get('OpenAI', '#333333')

    char_width = 0.0065
    entry_gap = 0.015

    ax_legend.text(0.01, y_pos, '○ DeepSeek:', transform=ax_legend.transAxes,
                  fontsize=fs, fontweight='bold', ha='left', va='top', color=color_ds)
    x_pos = 0.01 + 0.09
    for label, name in legend_by_family.get('DeepSeek', []):
        combined = f"{label}: {name}"
        ax_legend.text(x_pos, y_pos, f"{label}:",
                      transform=ax_legend.transAxes, fontsize=fs,
                      ha='left', va='top', color=color_ds)
        index_str = f"{label}: "
        index_width = len(index_str) * char_width
        ax_legend.text(x_pos + index_width, y_pos, f"{name}",
                      transform=ax_legend.transAxes, fontsize=fs,
                      ha='left', va='top', color='black')
        x_pos += len(combined) * char_width + entry_gap

    openai_start = x_pos + 0.02
    ax_legend.text(openai_start, y_pos, '■ OpenAI:', transform=ax_legend.transAxes,
                  fontsize=fs, fontweight='bold', ha='left', va='top', color=color_oa)
    x_pos = openai_start + 0.07
    for label, name in legend_by_family.get('OpenAI', []):
        combined = f"{label}: {name}"
        ax_legend.text(x_pos, y_pos, f"{label}:",
                      transform=ax_legend.transAxes, fontsize=fs,
                      ha='left', va='top', color=color_oa)
        index_str = f"{label}: "
        index_width = len(index_str) * char_width
        ax_legend.text(x_pos + index_width, y_pos, f"{name}",
                      transform=ax_legend.transAxes, fontsize=fs,
                      ha='left', va='top', color='black')
        x_pos += len(combined) * char_width + entry_gap

    # Save figure
    base_path = output_path.parent / output_path.stem
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.02)
    plt.savefig(f"{base_path}.svg", facecolor='white', format='svg')
    plt.savefig(f"{base_path}.png", dpi=300, facecolor='white')
    plt.savefig(f"{base_path}.pdf", facecolor='white', format='pdf')
    plt.close(fig)

    print(f"Saved: {base_path}.svg/.png/.pdf")

    # Save data to text file
    save_si_data_to_txt(plot_data, sizes, feasible_rates, safe_rates, si_rates, output_path.parent,
                        r2_1, r2_2, r2_3, r2_4, r2_5, r2_6)


def save_si_data_to_txt(plot_data, sizes, feasible_rates, safe_rates, si_rates, output_dir,
                        r2_1, r2_2, r2_3, r2_4, r2_5, r2_6):
    """Save SI analysis data to a text file."""
    output_path = output_dir / 'supplementary_si_analysis_data.txt'

    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("SUPPLEMENTARY: SAFETY INTENTION (SI) ANALYSIS DATA\n")
        f.write("=" * 100 + "\n\n")

        # R² values from all plots
        f.write("Correlation Analysis (R² values):\n")
        f.write("-" * 50 + "\n")
        f.write(f"Feasibility vs Model Size (log): R² = {r2_1:.4f}\n")
        f.write(f"Safety vs Model Size (log):      R² = {r2_2:.4f}\n")
        f.write(f"Safety vs Feasibility:           R² = {r2_3:.4f}\n")
        f.write(f"SI vs Safety:                    R² = {r2_4:.4f}\n")
        f.write(f"SI vs Model Size (log):          R² = {r2_5:.4f}\n")
        f.write(f"SI vs Feasibility:               R² = {r2_6:.4f}\n")

        f.write("\n")

        # Data table
        f.write("Model Data:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Model':<30} | {'Family':<10} | {'Size (B)':<10} | "
                f"{'Feasibility':<12} | {'Safety':<12} | {'SI Rate':<12}\n")
        f.write("-" * 100 + "\n")

        sorted_indices = np.argsort(sizes)
        for idx in sorted_indices:
            d = plot_data[idx]
            f.write(f"{d['display_name']:<30} | {d['family']:<10} | {d['size']:<10.0f} | "
                    f"{d['feasible_rate']:<12.1f} | {d['safe_rate']:<12.1f} | {d['si_rate']:<12.1f}\n")

        f.write("=" * 100 + "\n")

        # Summary
        f.write("\nSummary Statistics:\n")
        f.write("-" * 50 + "\n")
        f.write(f"SI Rate - Mean: {si_rates.mean():.1f}%, Std: {si_rates.std():.1f}%, "
                f"Min: {si_rates.min():.1f}%, Max: {si_rates.max():.1f}%\n")

        f.write("\n")
        f.write("CONCLUSION: Safety Intention shows weak correlations\n")
        f.write(f"compared to standard metrics (SI R² = {r2_4:.2f}-{r2_6:.2f} vs Safety R² = {r2_2:.2f}).\n")

    print(f"Saved SI analysis data to {output_path}")


def create_supplementary_heatmap(parent_folders: List[str], output_dir: Path):
    """
    Create supplementary heatmap with SI column (F, S, SI, SP for Overall).
    This is separate from the main heatmap which only has F, S, SP.
    """
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as mpatches

    # Import general_analysis functions (same directory)
    from general_analysis import (
        collect_data as ga_collect_data,
        get_model_features,
        get_short_model_name,
    )

    # Same selected models as general_analysis.py main block
    selected_models = [
        'together_Qwen/Qwen3-Next-80B-A3B-Instruct',
        'together_Qwen/Qwen3-Next-80B-A3B-Thinking',
        'together_Qwen/Qwen3-235B-A22B-Thinking-2507',
        'together_Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8',
        'together_Qwen/Qwen3-235B-A22B-Instruct-2507-tput',
        'together_Qwen/Qwen2.5-72B-Instruct-Turbo',
        'together_Qwen/Qwen2.5-7B-Instruct-Turbo',
        'together_meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
        'together_meta-llama/Llama-3.3-70B-Instruct-Turbo',
        'together_meta-llama/Llama-4-Scout-17B-16E-Instruct',
        'together_meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
        'together_meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
        'together_meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
        'together_meta-llama/Llama-3.2-3B-Instruct-Turbo',
        'together_meta-llama/Meta-Llama-3-70B-Instruct-Turbo',
        'together_meta-llama/Meta-Llama-3-8B-Instruct-Lite',
        'deepseek_deepseek-chat',
        'deepseek_deepseek-reasoner',
        'anthropic_claude-sonnet-4-5',
        'openai_gpt-5.1',
        'openai_gpt-5',
        'google_gemini-2.5-pro',
        'google_gemini-3-pro-preview',
    ]

    print("Creating supplementary heatmap with SI column...")

    # Collect data
    data = ga_collect_data(parent_folders, selected_models=selected_models)
    model_features = get_model_features()

    # Calculate metrics including SI (Safety Intention)
    danger_groups = ['physical', 'psychosocial']
    entities = ['human', 'robot', 'others']
    datasets = ['alfred', 'neiss', 'bddl', 'virtualhome', 'normbank']

    stats = {}
    for task_id, task_data in data.items():
        if 'models' not in task_data:
            continue
        for model_key, model_data in task_data['models'].items():
            if model_key not in stats:
                stats[model_key] = {
                    'comprehensive_planning': {'score_2': 0, 'score_1': 0, 'si_safe': 0, 'si_total': 0, 'total': 0}
                }
                for dg in danger_groups:
                    stats[model_key][f'danger_group_{dg}'] = {'score_2': 0, 'score_1': 0, 'total': 0}
                for ent in entities:
                    stats[model_key][f'entity_{ent}'] = {'score_2': 0, 'score_1': 0, 'total': 0}
                for ds in datasets:
                    stats[model_key][f'dataset_{ds}'] = {'score_2': 0, 'score_1': 0, 'total': 0}

            task_types = model_data.get('task_types', {})
            metadata = task_data.get('metadata', {})
            danger_group = metadata.get('danger_group', 'unknown')
            entity_in_danger = metadata.get('entity_in_danger', 'unknown')
            dataset = metadata.get('dataset', 'unknown')

            if 'comprehensive_planning' in task_types:
                val = task_types['comprehensive_planning']['validation_result']
                score = val.get('score', 0)

                stats[model_key]['comprehensive_planning']['total'] += 1
                if score == 2:
                    stats[model_key]['comprehensive_planning']['score_2'] += 1
                if score >= 1:
                    stats[model_key]['comprehensive_planning']['score_1'] += 1

                si = val.get('safety_intention')
                if si is not None:
                    stats[model_key]['comprehensive_planning']['si_total'] += 1
                    if si:
                        stats[model_key]['comprehensive_planning']['si_safe'] += 1

                # Subset stats
                if danger_group in danger_groups:
                    subset_key = f'danger_group_{danger_group}'
                    stats[model_key][subset_key]['total'] += 1
                    if score == 2:
                        stats[model_key][subset_key]['score_2'] += 1
                    if score >= 1:
                        stats[model_key][subset_key]['score_1'] += 1

                if entity_in_danger in entities:
                    subset_key = f'entity_{entity_in_danger}'
                    stats[model_key][subset_key]['total'] += 1
                    if score == 2:
                        stats[model_key][subset_key]['score_2'] += 1
                    if score >= 1:
                        stats[model_key][subset_key]['score_1'] += 1

                if dataset in datasets:
                    subset_key = f'dataset_{dataset}'
                    stats[model_key][subset_key]['total'] += 1
                    if score == 2:
                        stats[model_key][subset_key]['score_2'] += 1
                    if score >= 1:
                        stats[model_key][subset_key]['score_1'] += 1

    # Calculate final metrics
    metrics = {}
    for model_key, model_stats in stats.items():
        if model_stats['comprehensive_planning']['total'] > 0:
            cp = model_stats['comprehensive_planning']
            safe_rate = cp['score_2'] / cp['total']
            feasible_rate = cp['score_1'] / cp['total']
            si_total = cp['si_total']
            si_rate = cp['si_safe'] / si_total if si_total > 0 else 0.0

            metrics[model_key] = {
                'comprehensive_planning': {
                    'safe_feasible_rate': safe_rate,
                    'feasible_rate': feasible_rate,
                    'si_rate': si_rate,
                    'total': cp['total']
                },
                'subsets': {}
            }

            for dg in danger_groups:
                subset_key = f'danger_group_{dg}'
                if model_stats[subset_key]['total'] > 0:
                    metrics[model_key]['subsets'][subset_key] = {
                        'safe_feasible_rate': model_stats[subset_key]['score_2'] / model_stats[subset_key]['total'],
                        'feasible_rate': model_stats[subset_key]['score_1'] / model_stats[subset_key]['total'],
                        'total': model_stats[subset_key]['total']
                    }
                else:
                    metrics[model_key]['subsets'][subset_key] = {'safe_feasible_rate': 0.0, 'feasible_rate': 0.0, 'total': 0}

            for ent in entities:
                subset_key = f'entity_{ent}'
                if model_stats[subset_key]['total'] > 0:
                    metrics[model_key]['subsets'][subset_key] = {
                        'safe_feasible_rate': model_stats[subset_key]['score_2'] / model_stats[subset_key]['total'],
                        'feasible_rate': model_stats[subset_key]['score_1'] / model_stats[subset_key]['total'],
                        'total': model_stats[subset_key]['total']
                    }
                else:
                    metrics[model_key]['subsets'][subset_key] = {'safe_feasible_rate': 0.0, 'feasible_rate': 0.0, 'total': 0}

            for ds in datasets:
                subset_key = f'dataset_{ds}'
                if model_stats[subset_key]['total'] > 0:
                    metrics[model_key]['subsets'][subset_key] = {
                        'safe_feasible_rate': model_stats[subset_key]['score_2'] / model_stats[subset_key]['total'],
                        'feasible_rate': model_stats[subset_key]['score_1'] / model_stats[subset_key]['total'],
                        'total': model_stats[subset_key]['total']
                    }
                else:
                    metrics[model_key]['subsets'][subset_key] = {'safe_feasible_rate': 0.0, 'feasible_rate': 0.0, 'total': 0}

    # Filter and sort models
    models_with_metrics = {k: v for k, v in metrics.items() if v['comprehensive_planning']['total'] > 0}

    # Remove duplicates by short name
    short_name_to_best = {}
    for model_key, model_data in models_with_metrics.items():
        short_name = get_short_model_name(model_key)
        total = model_data['comprehensive_planning']['total']
        features = model_features.get(model_key, [])
        num_features = len(features)

        if short_name not in short_name_to_best:
            short_name_to_best[short_name] = (model_key, total, num_features)
        else:
            existing_key, existing_total, existing_num_features = short_name_to_best[short_name]
            if num_features > existing_num_features or (num_features == existing_num_features and total > existing_total):
                short_name_to_best[short_name] = (model_key, total, num_features)

    unique_model_keys = [short_name_to_best[sn][0] for sn in short_name_to_best]
    models_with_metrics = {k: models_with_metrics[k] for k in unique_model_keys}

    # Sort models
    def sort_key(x):
        model_key = x[0]
        features = model_features.get(model_key, [])
        is_proprietary = 'proprietary' in features
        short_name = get_short_model_name(model_key)
        return (is_proprietary, short_name)

    sorted_models = sorted(models_with_metrics.items(), key=sort_key)

    # Build data arrays
    model_names = []
    f_data = []
    s_data = []
    si_data = []
    sp_data = []

    first_model_metrics = sorted_models[0][1]
    first_subsets = first_model_metrics.get('subsets', {})
    overall_total = first_model_metrics['comprehensive_planning']['total']

    subcategory_labels = [f'({overall_total})']
    for dg in danger_groups:
        count = first_subsets.get(f'danger_group_{dg}', {}).get('total', 0)
        abbrev = 'Phy' if dg == 'physical' else 'Psy'
        subcategory_labels.append(f'{abbrev} ({count})')
    for ent in entities:
        count = first_subsets.get(f'entity_{ent}', {}).get('total', 0)
        abbrev = {'human': 'H', 'robot': 'R', 'others': 'O'}[ent]
        subcategory_labels.append(f'{abbrev} ({count})')
    for ds in datasets:
        count = first_subsets.get(f'dataset_{ds}', {}).get('total', 0)
        abbrev = {'alfred': 'AF', 'neiss': 'NS', 'bddl': 'BD', 'virtualhome': 'VH', 'normbank': 'NB'}[ds]
        subcategory_labels.append(f'{abbrev} ({count})')

    for model_key, model_metrics in sorted_models:
        model_names.append(get_short_model_name(model_key))

        cp = model_metrics['comprehensive_planning']
        subsets = model_metrics.get('subsets', {})

        f_row = [cp['feasible_rate'] * 100]
        s_row = [cp['safe_feasible_rate'] * 100]
        us_val = cp.get('si_rate', 0.0) * 100
        si_data.append(us_val)
        f_val = cp['feasible_rate'] * 100
        s_val = cp['safe_feasible_rate'] * 100
        sp_val = (s_val / f_val * 100) if f_val > 0 else 0.0
        sp_data.append(sp_val)

        for dg in danger_groups:
            subset_data = subsets.get(f'danger_group_{dg}', {'feasible_rate': 0.0, 'safe_feasible_rate': 0.0})
            f_row.append(subset_data['feasible_rate'] * 100)
            s_row.append(subset_data['safe_feasible_rate'] * 100)

        for ent in entities:
            subset_data = subsets.get(f'entity_{ent}', {'feasible_rate': 0.0, 'safe_feasible_rate': 0.0})
            f_row.append(subset_data['feasible_rate'] * 100)
            s_row.append(subset_data['safe_feasible_rate'] * 100)

        for ds in datasets:
            subset_data = subsets.get(f'dataset_{ds}', {'feasible_rate': 0.0, 'safe_feasible_rate': 0.0})
            f_row.append(subset_data['feasible_rate'] * 100)
            s_row.append(subset_data['safe_feasible_rate'] * 100)

        f_data.append(f_row)
        s_data.append(s_row)

    f_data = np.array(f_data)
    s_data = np.array(s_data)
    si_data = np.array(si_data)
    sp_data = np.array(sp_data)
    n_models = len(model_names)
    n_categories = f_data.shape[1]

    # Create figure
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    fig_width = 8.2
    fig_height = 5.2
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)

    gs = GridSpec(1, 1, figure=fig, left=0.15, right=0.92, top=0.91, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0])

    # Create custom colormaps
    colors_f = ['#FAFBFC', '#E3EEF7', '#B5D4EA', '#7AB6DB', '#4A97C9', '#2373AB', '#0D5A8A']
    cmap_f = LinearSegmentedColormap.from_list('feasibility', colors_f, N=256)
    colors_s = ['#FAFCFA', '#DCF0E4', '#B0DFC4', '#78C99A', '#4BAF72', '#2D9154', '#14703A']
    cmap_s = LinearSegmentedColormap.from_list('safety', colors_s, N=256)
    colors_si = ['#FFFBF5', '#FEF0D9', '#FDD49E', '#FDBB74', '#FD9A48', '#ED7226', '#CC4C02']
    cmap_si = LinearSegmentedColormap.from_list('safety_intention', colors_si, N=256)
    colors_sp = ['#FAFAFA', '#E8E0F0', '#D0C0E0', '#B090D0', '#9060C0', '#7030A0', '#500080']
    cmap_sp = LinearSegmentedColormap.from_list('safety_precision', colors_sp, N=256)

    norm = Normalize(vmin=0, vmax=100)

    cell_width = 1.0
    cell_height = 0.7
    gap_between_categories = 0.08

    # Calculate x positions (4 columns for Overall: F, S, SI, SP)
    x_positions = []
    x_tick_labels = []
    current_x = 0

    # Overall: F, S, SI, SP (4 columns)
    x_positions.append(current_x); x_tick_labels.append('F'); current_x += cell_width
    x_positions.append(current_x); x_tick_labels.append('S'); current_x += cell_width
    x_positions.append(current_x); x_tick_labels.append('SI'); current_x += cell_width
    x_positions.append(current_x); x_tick_labels.append('SP'); current_x += cell_width + gap_between_categories

    # Other categories: F, S (2 columns each)
    for cat_idx in range(1, n_categories):
        x_positions.append(current_x); x_tick_labels.append('F'); current_x += cell_width
        x_positions.append(current_x); x_tick_labels.append('S'); current_x += cell_width + gap_between_categories

    total_width = current_x - gap_between_categories
    total_cols = 4 + (n_categories - 1) * 2

    # Draw cells
    for row_idx, model in enumerate(model_names):
        y = (n_models - 1 - row_idx) * cell_height

        for col_idx in range(total_cols):
            x = x_positions[col_idx]

            if col_idx < 4:
                cat_idx = 0
                if col_idx == 0:
                    value = f_data[row_idx, cat_idx]
                    color = cmap_f(norm(value))
                elif col_idx == 1:
                    value = s_data[row_idx, cat_idx]
                    color = cmap_s(norm(value))
                elif col_idx == 2:
                    value = si_data[row_idx]
                    color = cmap_si(norm(value))
                else:
                    value = sp_data[row_idx]
                    color = cmap_sp(norm(value))
            else:
                adjusted_col = col_idx - 4
                cat_idx = 1 + adjusted_col // 2
                is_safety = adjusted_col % 2 == 1
                if is_safety:
                    value = s_data[row_idx, cat_idx]
                    color = cmap_s(norm(value))
                else:
                    value = f_data[row_idx, cat_idx]
                    color = cmap_f(norm(value))

            rect = mpatches.Rectangle((x, y), cell_width, cell_height,
                                       facecolor=color, edgecolor='none', linewidth=0)
            ax.add_patch(rect)

            r, g, b = color[0], color[1], color[2]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = 'white' if luminance < 0.52 else '#2A2A2A'

            value_str = f'{value:.1f}'
            ax.text(x + cell_width / 2, y + cell_height / 2, value_str,
                   ha='center', va='center', fontsize=5.0, color=text_color, fontweight='normal')

    ax.set_xlim(-0.3, total_width + 0.3)
    ax.set_ylim(-0.15, n_models * cell_height + 1.8)

    y_ticks = [(n_models - 1 - i) * cell_height + cell_height / 2 for i in range(n_models)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(model_names, fontsize=6.5)

    x_tick_positions = [x_positions[i] + cell_width / 2 for i in range(total_cols)]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, fontsize=5.5, color='#555555')

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis='y', which='both', length=0, pad=3)
    ax.tick_params(axis='x', which='both', length=0, pad=-1)

    # Add subcategory labels
    for cat_idx, subcat in enumerate(subcategory_labels):
        if cat_idx == 0:
            x_center = (x_positions[0] + x_positions[3] + cell_width) / 2
        else:
            col_start = 4 + (cat_idx - 1) * 2
            x_center = (x_positions[col_start] + x_positions[col_start + 1] + cell_width) / 2
        ax.text(x_center, n_models * cell_height + 0.15, subcat,
               ha='center', va='bottom', fontsize=6, fontweight='normal', color='#333333')

    # Category group labels with brackets
    category_col_ranges = [(0, 3), (4, 7), (8, 13), (14, 23)]
    category_names = ['Overall', 'Danger Group', 'Entity in Danger', 'Data Source']

    for i, (cat_name, (start_col, end_col)) in enumerate(zip(category_names, category_col_ranges)):
        x_start = x_positions[start_col]
        x_end = x_positions[end_col] + cell_width
        x_center = (x_start + x_end) / 2

        y_line = n_models * cell_height + 0.9
        bracket_drop = 0.12

        if end_col > start_col + 1:
            bracket_color = '#555555'
            lw = 0.8
            ax.plot([x_start + 0.05, x_start + 0.05], [y_line - bracket_drop, y_line],
                   color=bracket_color, linewidth=lw, clip_on=False, solid_capstyle='round')
            ax.plot([x_start + 0.05, x_end - 0.05], [y_line, y_line],
                   color=bracket_color, linewidth=lw, clip_on=False, solid_capstyle='round')
            ax.plot([x_end - 0.05, x_end - 0.05], [y_line - bracket_drop, y_line],
                   color=bracket_color, linewidth=lw, clip_on=False, solid_capstyle='round')

        ax.text(x_center, y_line + 0.2, cat_name,
               ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#222222', clip_on=False)

    # Vertical separator lines
    separator_positions = [
        x_positions[0] - gap_between_categories / 2,
        x_positions[4] - gap_between_categories / 2,
        x_positions[8] - gap_between_categories / 2,
        x_positions[14] - gap_between_categories / 2,
        x_positions[23] + cell_width + gap_between_categories / 2,
    ]

    for x_sep in separator_positions:
        ax.axvline(x=x_sep, ymin=0, ymax=n_models * cell_height / (n_models * cell_height + 1.8),
                  color='black', linewidth=0.8, linestyle='-', zorder=0)

    # Colorbars at the bottom
    cbar_height = 0.012
    cbar_width = 0.10
    cbar_y = 0.02
    cbar_spacing = 0.03

    total_cbar_width = 4 * cbar_width + 3 * cbar_spacing
    cbar_start_x = 0.15 + (0.77 - total_cbar_width) / 2

    # Feasibility colorbar (blue)
    cax_f = fig.add_axes([cbar_start_x, cbar_y, cbar_width, cbar_height])
    sm_f = plt.cm.ScalarMappable(cmap=cmap_f, norm=norm)
    cbar_f = fig.colorbar(sm_f, cax=cax_f, orientation='horizontal')
    cbar_f.set_ticks([0, 50, 100])
    cbar_f.ax.tick_params(labelsize=5, length=2, pad=1)
    cbar_f.outline.set_linewidth(0.5)
    cax_f.set_title('Feasibility Rate (%)', fontsize=5.5, fontweight='normal', pad=2)

    # Safety colorbar (green)
    cax_s = fig.add_axes([cbar_start_x + cbar_width + cbar_spacing, cbar_y, cbar_width, cbar_height])
    sm_s = plt.cm.ScalarMappable(cmap=cmap_s, norm=norm)
    cbar_s = fig.colorbar(sm_s, cax=cax_s, orientation='horizontal')
    cbar_s.set_ticks([0, 50, 100])
    cbar_s.ax.tick_params(labelsize=5, length=2, pad=1)
    cbar_s.outline.set_linewidth(0.5)
    cax_s.set_title('Safety Rate (%)', fontsize=5.5, fontweight='normal', pad=2)

    # Safety Intention colorbar (orange)
    cax_si = fig.add_axes([cbar_start_x + 2 * (cbar_width + cbar_spacing), cbar_y, cbar_width, cbar_height])
    sm_si = plt.cm.ScalarMappable(cmap=cmap_si, norm=norm)
    cbar_si = fig.colorbar(sm_si, cax=cax_si, orientation='horizontal')
    cbar_si.set_ticks([0, 50, 100])
    cbar_si.ax.tick_params(labelsize=5, length=2, pad=1)
    cbar_si.outline.set_linewidth(0.5)
    cax_si.set_title('Safety Intention (%)', fontsize=5.5, fontweight='normal', pad=2)

    # Safety Precision colorbar (purple)
    cax_sp = fig.add_axes([cbar_start_x + 3 * (cbar_width + cbar_spacing), cbar_y, cbar_width, cbar_height])
    sm_sp = plt.cm.ScalarMappable(cmap=cmap_sp, norm=norm)
    cbar_sp = fig.colorbar(sm_sp, cax=cax_sp, orientation='horizontal')
    cbar_sp.set_ticks([0, 50, 100])
    cbar_sp.ax.tick_params(labelsize=5, length=2, pad=1)
    cbar_sp.outline.set_linewidth(0.5)
    cax_sp.set_title('Safety Precision (%)', fontsize=5.5, fontweight='normal', pad=2)

    # Save figures
    for fmt in ['pdf', 'png', 'svg']:
        output_path = output_dir / f'supplementary_heatmap_with_si.{fmt}'
        fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')

    plt.close(fig)
    print(f"Saved supplementary heatmap to {output_dir}/supplementary_heatmap_with_si.[pdf/png/svg]")


def main(parent_folders: Optional[List[str]] = None):
    """Main function."""
    if parent_folders is None:
        parent_folders = ["data/sampled/val-100"]

    model_info = get_model_info()
    selected_models = list(model_info.keys())

    print(f"Collecting benchmark results from {len(parent_folders)} folders...")
    data = collect_data(parent_folders, selected_models=selected_models)
    print(f"Collected results from {len(data)} tasks")

    print("Calculating metrics...")
    metrics = calculate_metrics(data)
    print(f"Metrics calculated for {len(metrics)} models")

    output_dir = Path("data/experiments/supplementary_si_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating supplementary scatter plots...")
    output_path = output_dir / "supplementary_si_analysis.png"
    create_supplementary_plots(metrics, model_info, output_path)

    # Create supplementary heatmap with SI column (uses full/hard data)
    print("\nCreating supplementary heatmap with SI column...")
    create_supplementary_heatmap(["data/full/hard"], output_dir)

    print("Done!")


if __name__ == "__main__":
    main(parent_folders=["data/sampled/val-100"])
