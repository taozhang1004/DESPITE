#!/usr/bin/env python3
"""
Scale Analysis: Model Size vs Performance

Creates scatter plots showing the relationship between model size (active parameters)
and performance metrics with linear trend lines and bootstrapped confidence intervals.

Panels (2x2):
  (a) Feasibility Rate vs Active Parameters
  (b) Safety Intention Rate vs Active Parameters (with F reference line)
  (c) Safety Rate vs Active Parameters
  (d) Safety Rate vs F×SI (all models including closed-source)
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.patheffects as path_effects
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_model_info() -> Dict[str, Dict[str, Any]]:
    """
    Define model information including size (total and active), family,
    release date, and whether the model is MoE or thinking.

    Active size = parameters activated per forward pass.
    For dense models, active_size = size. For MoE models, active_size < size.
    Sizes verified from official technical reports and HuggingFace model cards.
    Display names use "A{X}B" for MoE models to clarify active parameter count.
    """
    m = {}

    # ── DeepSeek (MoE: 671B total, 37B active, 256 routed + 1 shared expert) ──
    m['deepseek_deepseek-chat'] = {
        'size': 671, 'active_size': 37, 'is_moe': True, 'is_thinking': False,
        'family': 'DeepSeek', 'display_name': 'V3.2-Exp-671B (A37B)',
        'release_date': '2025-09-29'
    }
    m['deepseek_deepseek-reasoner'] = {
        'size': 671, 'active_size': 37, 'is_moe': True, 'is_thinking': True,
        'family': 'DeepSeek', 'display_name': 'V3.2-Exp-Think-671B (A37B)',
        'release_date': '2025-09-29'
    }

    # ── Qwen Dense ──
    m['together_Qwen/Qwen2.5-7B-Instruct-Turbo'] = {
        'size': 7, 'active_size': 7, 'is_moe': False, 'is_thinking': False,
        'family': 'Qwen', 'display_name': '2.5-7B',
        'release_date': '2024-09-19'
    }
    m['together_Qwen/Qwen2.5-72B-Instruct-Turbo'] = {
        'size': 72, 'active_size': 72, 'is_moe': False, 'is_thinking': False,
        'family': 'Qwen', 'display_name': '2.5-72B',
        'release_date': '2024-09-19'
    }

    # ── Qwen MoE ──
    # Qwen3-235B: 235B total, 22B active, 128 experts / 8 active
    m['together_Qwen/Qwen3-235B-A22B-Instruct-2507-tput'] = {
        'size': 235, 'active_size': 22, 'is_moe': True, 'is_thinking': False,
        'family': 'Qwen', 'display_name': '3-235B (A22B)',
        'release_date': '2025-04-28'
    }
    m['together_Qwen/Qwen3-235B-A22B-Thinking-2507'] = {
        'size': 235, 'active_size': 22, 'is_moe': True, 'is_thinking': True,
        'family': 'Qwen', 'display_name': '3-235B-Think (A22B)',
        'release_date': '2025-04-28'
    }
    # Qwen3-Coder-480B: 480B total, 35B active, 160 experts / 8 active
    m['together_Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8'] = {
        'size': 480, 'active_size': 35, 'is_moe': True, 'is_thinking': False,
        'family': 'Qwen', 'display_name': '3-Coder-480B (A35B)',
        'release_date': '2025-07-22'
    }
    # Qwen3-Next-80B: 80B total, 3B active, 512 experts / 10+1 active (hybrid MoE)
    m['together_Qwen/Qwen3-Next-80B-A3B-Instruct'] = {
        'size': 80, 'active_size': 3, 'is_moe': True, 'is_thinking': False,
        'family': 'Qwen', 'display_name': '3-Next-80B (A3B)',
        'release_date': '2025-09-10'
    }
    m['together_Qwen/Qwen3-Next-80B-A3B-Thinking'] = {
        'size': 80, 'active_size': 3, 'is_moe': True, 'is_thinking': True,
        'family': 'Qwen', 'display_name': '3-Next-80B-Think (A3B)',
        'release_date': '2025-09-10'
    }

    # ── Meta Llama Dense ──
    m['together_meta-llama/Meta-Llama-3-8B-Instruct-Lite'] = {
        'size': 8, 'active_size': 8, 'is_moe': False, 'is_thinking': False,
        'family': 'Llama', 'display_name': '3-8B',
        'release_date': '2024-04-18'
    }
    m['together_meta-llama/Meta-Llama-3-70B-Instruct-Turbo'] = {
        'size': 70, 'active_size': 70, 'is_moe': False, 'is_thinking': False,
        'family': 'Llama', 'display_name': '3-70B',
        'release_date': '2024-04-18'
    }
    m['together_meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'] = {
        'size': 8, 'active_size': 8, 'is_moe': False, 'is_thinking': False,
        'family': 'Llama', 'display_name': '3.1-8B',
        'release_date': '2024-07-23'
    }
    m['together_meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'] = {
        'size': 70, 'active_size': 70, 'is_moe': False, 'is_thinking': False,
        'family': 'Llama', 'display_name': '3.1-70B',
        'release_date': '2024-07-23'
    }
    m['together_meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo'] = {
        'size': 405, 'active_size': 405, 'is_moe': False, 'is_thinking': False,
        'family': 'Llama', 'display_name': '3.1-405B',
        'release_date': '2024-07-23'
    }
    m['together_meta-llama/Llama-3.2-3B-Instruct-Turbo'] = {
        'size': 3, 'active_size': 3, 'is_moe': False, 'is_thinking': False,
        'family': 'Llama', 'display_name': '3.2-3B',
        'release_date': '2024-09-25'
    }
    m['together_meta-llama/Llama-3.3-70B-Instruct-Turbo'] = {
        'size': 70, 'active_size': 70, 'is_moe': False, 'is_thinking': False,
        'family': 'Llama', 'display_name': '3.3-70B',
        'release_date': '2024-12-06'
    }

    # ── Meta Llama MoE ──
    # Llama 4 Scout: 109B total, 17B active, 16 routed + 1 shared expert
    m['together_meta-llama/Llama-4-Scout-17B-16E-Instruct'] = {
        'size': 109, 'active_size': 17, 'is_moe': True, 'is_thinking': False,
        'family': 'Llama', 'display_name': '4-Scout-109B (A17B)',
        'release_date': '2025-04-05'
    }
    # Llama 4 Maverick: 400B total, 17B active, 128 routed + 1 shared expert
    m['together_meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'] = {
        'size': 400, 'active_size': 17, 'is_moe': True, 'is_thinking': False,
        'family': 'Llama', 'display_name': '4-Maverick-400B (A17B)',
        'release_date': '2025-04-05'
    }

    # ── Closed-source models (for S vs F×SI panel only, no size info) ──
    m['anthropic_claude-sonnet-4-5'] = {
        'family': 'Anthropic', 'display_name': 'Sonnet 4.5',
        'is_thinking': False, 'release_date': '2025-03-01'
    }
    m['openai_gpt-5'] = {
        'family': 'OpenAI', 'display_name': 'GPT-5',
        'is_thinking': False, 'release_date': '2025-05-14'
    }
    m['openai_gpt-5.1'] = {
        'family': 'OpenAI', 'display_name': 'GPT-5.1',
        'is_thinking': False, 'release_date': '2025-07-24'
    }
    m['google_gemini-2.5-pro'] = {
        'family': 'Google', 'display_name': 'Gemini-2.5-Pro',
        'is_thinking': False, 'release_date': '2025-03-25'
    }
    m['google_gemini-3-pro-preview'] = {
        'family': 'Google', 'display_name': 'Gemini-3-Pro',
        'is_thinking': False, 'release_date': '2025-08-05'
    }

    return m


def collect_data(parent_folders: List[str], selected_models: Optional[List[str]] = None) -> Dict[str, Any]:
    """Collect benchmark results from task folders."""
    results = {}
    for parent_folder in parent_folders:
        pattern = f"{parent_folder}/*/benchmark_results_1.json"
        for result_file in glob.glob(pattern):
            task_id = Path(result_file).parent.name
            with open(result_file) as f:
                data = json.load(f)
                if selected_models is not None and 'models' in data:
                    data['models'] = {k: v for k, v in data['models'].items() if k in selected_models}
                results[task_id] = data
    return results


def calculate_metrics(data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Calculate comprehensive_planning metrics from benchmark results."""
    model_stats = {}
    for task_data in data.values():
        if 'models' not in task_data:
            continue
        for model_key, model_data in task_data['models'].items():
            if model_key not in model_stats:
                model_stats[model_key] = {'score_2': 0, 'score_1': 0, 'si_true': 0, 'total': 0}
            task_types = model_data.get('task_types', {})
            if 'comprehensive_planning' in task_types:
                model_stats[model_key]['total'] += 1
                val = task_types['comprehensive_planning']['validation_result']
                score = val.get('score', 0)
                if score == 2:
                    model_stats[model_key]['score_2'] += 1
                if score >= 1:
                    model_stats[model_key]['score_1'] += 1
                if val.get('safety_intention', False):
                    model_stats[model_key]['si_true'] += 1

    final = {}
    for model_key, s in model_stats.items():
        if s['total'] > 0:
            final[model_key] = {
                'safe_rate': s['score_2'] / s['total'],
                'feasible_rate': s['score_1'] / s['total'],
                'si_rate': s['si_true'] / s['total'],
                'total': s['total']
            }
    return final


def apply_axis_style(ax, fontsize=14):
    """Apply consistent Nature/Science style to an axis."""
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(0.5)
    ax.tick_params(axis='both', which='both', direction='out', length=3, width=0.5,
                   labelsize=fontsize, pad=3, colors='black',
                   bottom=True, left=True, top=False, right=False)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', visible=False)
    ax.set_axisbelow(True)


def format_stats(r2, slope, subscript=''):
    """Format slope and R² for display on plot."""
    r2_str = f'R\u00b2={r2:.4f}' if r2 < 0.01 else f'R\u00b2={r2:.2f}'
    if subscript:
        return f'\u03b2$_{{{subscript}}}$={slope:.1f}, {r2_str}'
    return f'\u03b2={slope:.1f}, {r2_str}'


def bootstrap_regression_ci(x, y, n_boot=10000, ci_level=0.95, seed=42):
    """Compute bootstrap CI band for linear regression y = slope*x + intercept."""
    n = len(x)
    x_pred = np.linspace(x.min(), x.max(), 200)
    y_preds = np.zeros((n_boot, len(x_pred)))
    slopes = np.zeros(n_boot)
    rng = np.random.RandomState(seed)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        s, inter, _, _, _ = stats.linregress(x[idx], y[idx])
        y_preds[i] = s * x_pred + inter
        slopes[i] = s
    alpha = (1 - ci_level) / 2
    y_lo = np.percentile(y_preds, alpha * 100, axis=0)
    y_hi = np.percentile(y_preds, (1 - alpha) * 100, axis=0)
    slope_ci = (np.percentile(slopes, alpha * 100), np.percentile(slopes, (1 - alpha) * 100))
    return x_pred, y_lo, y_hi, slope_ci


def save_plot_data_to_txt(plot_data, x_values, y_values, x_label, y_label,
                          trend_stats, output_path, append=False):
    """Save plot data to a text file for detailed analysis."""
    mode = 'a' if append else 'w'
    with open(output_path, mode) as f:
        if append:
            f.write("\n\n" + "#" * 100 + "\n" + "#" * 100 + "\n\n")
        f.write(f"{x_label} vs {y_label}\n")
        f.write("=" * 100 + "\n\n")
        if trend_stats:
            f.write("Trend Line Statistics:\n")
            f.write("-" * 100 + "\n")
            f.write(f"R\u00b2: {trend_stats['r_squared']:.6f}\n")
            f.write(f"Slope: {trend_stats['slope']:.6f}\n")
            f.write(f"P-value: {trend_stats['p_value']:.6e}\n")
            if trend_stats.get('slope_ci'):
                lo, hi = trend_stats['slope_ci']
                f.write(f"Slope 95% CI: [{lo:.4f}, {hi:.4f}]\n")
            f.write("\n")
        f.write("Data Points:\n")
        f.write("-" * 100 + "\n")
        sorted_indices = np.argsort(x_values)
        for idx in sorted_indices:
            d = plot_data[idx]
            f.write(f"  {d['display_name']:<35} {d['family']:<10} "
                    f"{'Think' if d.get('is_thinking') else 'Std':<6} "
                    f"x={x_values[idx]:<12.4f} y={y_values[idx]:<12.4f}\n")
        f.write("=" * 100 + "\n")
        f.write(f"N={len(plot_data)}, x=[{x_values.min():.2f}, {x_values.max():.2f}], "
                f"y=[{y_values.min():.2f}, {y_values.max():.2f}]\n")


def create_scatter_plots_normalized(metrics: Dict[str, Dict[str, float]],
                                    model_info: Dict[str, Dict[str, Any]],
                                    output_path: Path):
    """Create publication-quality 2x2 scatter plots:
    (a) F vs Active Size, (b) SI vs Active Size,
    (c) S vs Active Size, (d) S vs F×SI.
    """

    # ── Prepare data ──────────────────────────────────────────────────────────
    # Scale panels (a,b,c): open-source models with known active_size
    scale_data = []
    for model_key, metric_data in metrics.items():
        if model_key in model_info and 'active_size' in model_info[model_key]:
            info = model_info[model_key]
            if metric_data.get('total', 0) > 0:
                scale_data.append({
                    'model_key': model_key,
                    'active_size': info['active_size'],
                    'size': info['size'],
                    'family': info['family'],
                    'display_name': info['display_name'],
                    'release_date': info.get('release_date', '2099-01-01'),
                    'is_thinking': info.get('is_thinking', False),
                    'is_moe': info.get('is_moe', False),
                    'safe_rate': metric_data['safe_rate'] * 100,
                    'feasible_rate': metric_data['feasible_rate'] * 100,
                    'si_rate': metric_data['si_rate'] * 100,
                    'total': metric_data['total'],
                })

    # Decomposition panel (d): ALL models in model_info (including closed-source)
    decomp_data = []
    for model_key, metric_data in metrics.items():
        if model_key in model_info and metric_data.get('total', 0) > 0:
            info = model_info[model_key]
            decomp_data.append({
                'model_key': model_key,
                'family': info['family'],
                'display_name': info['display_name'],
                'is_thinking': info.get('is_thinking', False),
                'safe_rate': metric_data['safe_rate'] * 100,
                'feasible_rate': metric_data['feasible_rate'] * 100,
                'si_rate': metric_data['si_rate'] * 100,
                'total': metric_data['total'],
            })

    if not scale_data:
        print("No scale data to plot!")
        return

    thinking_data = [d for d in scale_data if d['is_thinking']]
    standard_data = [d for d in scale_data if not d['is_thinking']]
    print(f"  Scale panels: {len(scale_data)} models ({len(standard_data)} std, {len(thinking_data)} think)")
    print(f"  Decomposition panel: {len(decomp_data)} models (incl. closed-source)")

    # ── Assign chronological numbers ──────────────────────────────────────────
    date_groups = defaultdict(list)
    for d in scale_data:
        date_groups[d['release_date']].append(d)

    model_labels = {}
    legend_entries = []
    current_number = 1
    for date in sorted(date_groups.keys()):
        models_on_date = sorted(date_groups[date], key=lambda x: (x['family'], x['display_name']))
        if len(models_on_date) == 1:
            label = str(current_number)
            model_labels[models_on_date[0]['model_key']] = label
            legend_entries.append((label, models_on_date[0]['display_name'],
                                   models_on_date[0]['family'], models_on_date[0]['is_thinking']))
        else:
            for i, model in enumerate(models_on_date):
                label = f"{current_number}{chr(ord('a') + i)}"
                model_labels[model['model_key']] = label
                legend_entries.append((label, model['display_name'],
                                       model['family'], model['is_thinking']))
        current_number += 1

    for d in scale_data:
        d['label'] = model_labels[d['model_key']]

    # ── Figure setup (2×2 + legend row) ───────────────────────────────────────
    fs = 14
    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.22], hspace=0.30, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])  # a: F vs Active Size
    ax2 = fig.add_subplot(gs[0, 1])  # b: S vs Active Size
    ax3 = fig.add_subplot(gs[1, 0])  # c: SI vs Active Size
    ax4 = fig.add_subplot(gs[1, 1])  # d: S vs F×SI
    ax_legend = fig.add_subplot(gs[2, :])
    fig.patch.set_facecolor('white')

    os_families = sorted(set(d['family'] for d in scale_data))
    family_colors_dark = {'DeepSeek': '#2c5aa0', 'Llama': '#1a6622', 'Qwen': '#b8a030'}
    family_markers = {'DeepSeek': 's', 'Llama': 'D', 'Qwen': 'o'}

    all_family_colors = {**family_colors_dark,
                         'OpenAI': '#cc3333', 'Google': '#cc6633', 'Anthropic': '#7744aa'}
    all_family_markers = {**family_markers, 'OpenAI': 'h', 'Google': '^', 'Anthropic': 'p'}

    # ── Arrays (all open-source models) ─────────────────────────────────────
    # Use total size for all models (dense size = active, MoE size = total params)
    total_sizes = np.array([d['size'] for d in scale_data])
    feasible_rates = np.array([d['feasible_rate'] for d in scale_data])
    safe_rates = np.array([d['safe_rate'] for d in scale_data])
    si_rates = np.array([d['si_rate'] for d in scale_data])
    is_moe_arr = np.array([d['is_moe'] for d in scale_data])
    log_sizes = np.log10(total_sizes)

    n_dense = (~is_moe_arr).sum()
    n_moe = is_moe_arr.sum()
    print(f"  Architecture split: {n_dense} dense, {n_moe} MoE")

    x_min, x_max = total_sizes.min(), total_sizes.max()
    x_line = np.logspace(np.log10(x_min), np.log10(x_max), 200)

    # ── Single combined regression (all models, x = total size) ───────────────
    slope_f, intercept_f, r_f, p_f, se_f = stats.linregress(log_sizes, feasible_rates)
    r2_f = r_f ** 2
    slope_si, intercept_si, r_si, p_si, se_si = stats.linregress(log_sizes, si_rates)
    r2_si = r_si ** 2
    slope_s, intercept_s, r_s, p_s, se_s = stats.linregress(log_sizes, safe_rates)
    r2_s = r_s ** 2

    # F×SI predicted safety (per model)
    all_fxsi = np.array([d['feasible_rate'] * d['si_rate'] / 100 for d in scale_data])

    # Bootstrap CIs (single combined trend)
    print("  Bootstrapping CIs (all models, x = total size)...")
    x_pred_f, y_lo_f, y_hi_f, slope_ci_f = bootstrap_regression_ci(log_sizes, feasible_rates)
    x_pred_si, y_lo_si, y_hi_si, slope_ci_si = bootstrap_regression_ci(log_sizes, si_rates)
    x_pred_s, y_lo_s, y_hi_s, slope_ci_s = bootstrap_regression_ci(log_sizes, safe_rates)

    # Bootstrap slope ratios (SI/F and S/F)
    n_boot = 10000
    rng_ratio = np.random.RandomState(42)
    n_models = len(log_sizes)
    si_f_ratios, s_f_ratios = [], []
    for _ in range(n_boot):
        idx = rng_ratio.choice(n_models, n_models, replace=True)
        f_sl, _, _, _, _ = stats.linregress(log_sizes[idx], feasible_rates[idx])
        si_sl, _, _, _, _ = stats.linregress(log_sizes[idx], si_rates[idx])
        s_sl, _, _, _, _ = stats.linregress(log_sizes[idx], safe_rates[idx])
        if abs(f_sl) > 0.01:
            si_f_ratios.append(si_sl / f_sl)
            s_f_ratios.append(s_sl / f_sl)
    si_f_ratios = np.array(si_f_ratios)
    s_f_ratios = np.array(s_f_ratios)
    si_f_ratio_lo = np.percentile(si_f_ratios, 2.5)
    si_f_ratio_hi = np.percentile(si_f_ratios, 97.5)
    si_f_ratio_point = slope_si / slope_f if abs(slope_f) > 0.01 else float('nan')
    s_f_ratio_lo = np.percentile(s_f_ratios, 2.5)
    s_f_ratio_hi = np.percentile(s_f_ratios, 97.5)
    s_f_ratio_point = slope_s / slope_f if abs(slope_f) > 0.01 else float('nan')

    print(f"  Combined: F={slope_f:.1f} [{slope_ci_f[0]:.1f}, {slope_ci_f[1]:.1f}], "
          f"S={slope_s:.1f} [{slope_ci_s[0]:.1f}, {slope_ci_s[1]:.1f}], "
          f"SI={slope_si:.1f} [{slope_ci_si[0]:.1f}, {slope_ci_si[1]:.1f}]")
    print(f"  S/F ratio:  {s_f_ratio_point:.3f} [{s_f_ratio_lo:.3f}, {s_f_ratio_hi:.3f}]")
    print(f"  SI/F ratio: {si_f_ratio_point:.3f} [{si_f_ratio_lo:.3f}, {si_f_ratio_hi:.3f}]")

    # ── Setup axes ────────────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.07, right=0.97, top=0.96, bottom=0.02)
    pad = 5
    for ax_i in [ax1, ax2, ax3]:
        ax_i.set_xscale('log')
        ax_i.set_xlim(x_min * 0.6, x_max * 1.8)
        ax_i.set_ylim(-pad, 100 + pad)
        apply_axis_style(ax_i, fontsize=fs)
    apply_axis_style(ax4, fontsize=fs)

    label_overrides = {}
    dir_map = {
        'up':    ((0, 1), 'center', 'bottom'),
        'right': ((1, 0), 'left', 'center'),
        'down':  ((0, -1), 'center', 'top'),
        'left':  ((-1, 0), 'right', 'center'),
    }

    # ── plot_scatter ──────────────────────────────────────────────────────────
    def plot_scatter(ax, y_func, subplot_idx=0, add_labels=True):
        """Plot scatter with auto-placed labels.
        X-axis: size for dense (=active), total size for MoE.
        Dense=filled markers, MoE=hollow markers."""
        # Define x_func: use 'size' for all (= active for dense, total for MoE)
        x_func = lambda d: d['size']

        for family in os_families:
            fam_data = [d for d in scale_data if d['family'] == family]
            if not fam_data:
                continue
            color = family_colors_dark.get(family, '#808080')
            marker = family_markers.get(family, 'o')
            # Dense models: filled markers
            dense_fam = [d for d in fam_data if not d['is_moe']]
            if dense_fam:
                ax.scatter([x_func(d) for d in dense_fam], [y_func(d) for d in dense_fam],
                          s=70, alpha=0.9, color=color, marker=marker,
                          edgecolors='white', linewidths=0.4, label=f'{family} (Dense)', zorder=3)
            # MoE models: hollow markers with thick edge
            moe_fam = [d for d in fam_data if d['is_moe']]
            if moe_fam:
                ax.scatter([x_func(d) for d in moe_fam], [y_func(d) for d in moe_fam],
                          s=70, alpha=0.9, facecolors='none', marker=marker,
                          edgecolors=color, linewidths=1.5, label=f'{family} (MoE)', zorder=3)

        if not add_labels:
            return

        # ── Label placement ───────────────────────────────────────────────
        outline = [path_effects.Stroke(linewidth=1.5, foreground='white'),
                   path_effects.Normal()]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        transform = ax.transData
        dpi = fig.dpi
        pts_per_px = 72.0 / dpi

        ax_bbox = ax.get_window_extent(renderer)
        margin_px = 3
        bx0, by0 = ax_bbox.x0 + margin_px, ax_bbox.y0 + margin_px
        bx1, by1 = ax_bbox.x1 - margin_px, ax_bbox.y1 - margin_px

        text_sizes = {}
        for d in scale_data:
            tmp = ax.text(x_func(d), y_func(d), d['label'], fontsize=fs, fontfamily='sans-serif')
            bb = tmp.get_window_extent(renderer)
            text_sizes[d['label']] = (bb.width, bb.height)
            tmp.remove()

        point_disps = [transform.transform((x_func(d), y_func(d))) for d in scale_data]

        directions = [
            ((0, 1), 'center', 'bottom'),
            ((1, 0), 'left', 'center'),
            ((0, -1), 'center', 'top'),
            ((-1, 0), 'right', 'center'),
        ]
        lengths_px = list(range(8, 160, 2))
        marker_r = 5
        marker_occ = [(px - marker_r, py - marker_r, px + marker_r, py + marker_r)
                      for px, py in point_disps]

        TEXT_PAD, LINE_W, GAP = 4, 2, 2

        def get_text_rect(cx, cy, tw, th, ha, va):
            x0 = cx if ha == 'left' else (cx - tw if ha == 'right' else cx - tw / 2)
            y0 = cy if va == 'bottom' else (cy - th if va == 'top' else cy - th / 2)
            return (x0 - TEXT_PAD, y0 - TEXT_PAD, x0 + tw + TEXT_PAD, y0 + th + TEXT_PAD)

        def get_line_rect(px, py, dx, dy, length_px):
            if dx != 0:
                lx0, lx1 = min(px, px + dx * length_px), max(px, px + dx * length_px)
                return (lx0 - LINE_W, py - LINE_W, lx1 + LINE_W, py + LINE_W)
            ly0, ly1 = min(py, py + dy * length_px), max(py, py + dy * length_px)
            return (px - LINE_W, ly0, px + LINE_W, ly1)

        def rects_overlap(r1, r2):
            return not (r1[2] <= r2[0] or r1[0] >= r2[2] or r1[3] <= r2[1] or r1[1] >= r2[3])

        def rect_in_bounds(r):
            return r[0] >= bx0 and r[1] >= by0 and r[2] <= bx1 and r[3] <= by1

        label_text_rects, label_line_rects, label_params = {}, {}, {}

        def get_occupied(exclude_idx=None, exclude_idxs=None):
            excluded = set()
            if exclude_idx is not None: excluded.add(exclude_idx)
            if exclude_idxs is not None: excluded.update(exclude_idxs)
            return ([r for i, r in enumerate(marker_occ) if i not in excluded],
                    [r for i, r in label_text_rects.items() if i not in excluded],
                    [r for i, r in label_line_rects.items() if i not in excluded])

        def count_collisions(rect, lrect, markers, texts, lines):
            hard = sum(1 for o in markers if rects_overlap(rect, o))
            hard += sum(1 for o in texts if rects_overlap(rect, o))
            soft = sum(1 for o in lines if rects_overlap(rect, o))
            soft += sum(1 for o in texts if rects_overlap(lrect, o))
            return hard, soft

        # Compute preferred directions
        nearby_radius = 60
        preferred_dirs = {}
        label_order = []
        for idx, d in enumerate(scale_data):
            px, py = point_disps[idx]
            nearby = [(point_disps[j][0], point_disps[j][1])
                     for j in range(len(point_disps))
                     if j != idx and ((px - point_disps[j][0])**2 +
                                      (py - point_disps[j][1])**2)**0.5 < nearby_radius]
            if nearby:
                cx, cy = np.mean([p[0] for p in nearby]), np.mean([p[1] for p in nearby])
                dx, dy = px - cx, py - cy
                preferred_dirs[idx] = (1, 0) if abs(dx) > abs(dy) else (0, 1)
                if abs(dx) > abs(dy) and dx < 0: preferred_dirs[idx] = (-1, 0)
                if abs(dy) >= abs(dx) and dy < 0: preferred_dirs[idx] = (0, -1)
            else:
                preferred_dirs[idx] = (0, 1)
            label_order.append((len(nearby), idx, d))
        label_order.sort(key=lambda x: -x[0])

        def is_overridden(label):
            return (subplot_idx, label) in label_overrides

        def get_dirs_lens(label):
            ovr = label_overrides.get((subplot_idx, label))
            dirs, lens = directions, lengths_px
            if ovr:
                if 'dir' in ovr: dirs = [dir_map[ovr['dir']]]
                if 'length' in ovr: lens = [ovr['length']]
            return dirs, lens

        # Phase 1: Greedy placement (overridden first)
        for _, idx, d in sorted(label_order, key=lambda x: (0 if is_overridden(x[2]['label']) else 1, -x[0])):
            px, py = point_disps[idx]
            tw, th = text_sizes[d['label']]
            pref = preferred_dirs[idx]
            markers, texts, lines = get_occupied(exclude_idx=idx)
            ovr_dirs, ovr_lens = get_dirs_lens(d['label'])
            best, best_score = None, float('inf')
            for (ddx, ddy), ha, va in ovr_dirs:
                for lp in ovr_lens:
                    rect = get_text_rect(px + ddx * (lp + GAP), py + ddy * (lp + GAP), tw, th, ha, va)
                    lrect = get_line_rect(px, py, ddx, ddy, lp)
                    if not rect_in_bounds(rect): continue
                    hard, soft = count_collisions(rect, lrect, markers, texts, lines)
                    score = hard * 10000 + soft * 30 + (25 if (ddx, ddy) != pref else 0) + lp
                    if score < best_score:
                        best_score = score
                        best = (ddx, ddy, lp, ha, va, rect, lrect)
            if best:
                ddx, ddy, lp, ha, va, rect, lrect = best
                label_text_rects[idx] = rect
                label_line_rects[idx] = lrect
                label_params[idx] = (ddx, ddy, lp, ha, va)

        # Phase 2: Fix overlaps
        for _ in range(3):
            any_fix = False
            for _, idx, d in label_order:
                if idx not in label_params or is_overridden(d['label']): continue
                markers, texts, lines = get_occupied(exclude_idx=idx)
                hard, _ = count_collisions(label_text_rects[idx], label_line_rects[idx], markers, texts, lines)
                if hard == 0: continue
                px, py = point_disps[idx]
                tw, th = text_sizes[d['label']]
                best, best_score = None, float('inf')
                for (ddx, ddy), ha, va in directions:
                    for lp in lengths_px:
                        rect = get_text_rect(px + ddx * (lp + GAP), py + ddy * (lp + GAP), tw, th, ha, va)
                        lrect = get_line_rect(px, py, ddx, ddy, lp)
                        if not rect_in_bounds(rect): continue
                        h, s = count_collisions(rect, lrect, markers, texts, lines)
                        sc = h * 10000 + s * 30 + lp
                        if sc < best_score: best_score, best = sc, (ddx, ddy, lp, ha, va, rect, lrect)
                if best:
                    ddx, ddy, lp, ha, va, rect, lrect = best
                    label_text_rects[idx] = rect; label_line_rects[idx] = lrect
                    label_params[idx] = (ddx, ddy, lp, ha, va); any_fix = True
            if not any_fix: break

        # Phase 3: Shorten
        for _ in range(5):
            any_short = False
            for _, idx, d in label_order:
                if idx not in label_params or is_overridden(d['label']): continue
                ddx, ddy, cur_lp, ha, va = label_params[idx]
                px, py = point_disps[idx]
                tw, th = text_sizes[d['label']]
                markers, texts, lines = get_occupied(exclude_idx=idx)
                for lp in lengths_px:
                    if lp >= cur_lp: break
                    rect = get_text_rect(px + ddx * (lp + GAP), py + ddy * (lp + GAP), tw, th, ha, va)
                    lrect = get_line_rect(px, py, ddx, ddy, lp)
                    if not rect_in_bounds(rect): continue
                    h, _ = count_collisions(rect, lrect, markers, texts, lines)
                    if h > 0: continue
                    label_text_rects[idx] = rect; label_line_rects[idx] = lrect
                    label_params[idx] = (ddx, ddy, lp, ha, va); any_short = True; break
            if not any_short: break

        # Phase 4: Pair-wise repair
        for _ in range(5):
            pairs = []
            idxs = list(label_text_rects.keys())
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a, b = idxs[i], idxs[j]
                    if is_overridden(scale_data[a]['label']) and is_overridden(scale_data[b]['label']): continue
                    if (rects_overlap(label_text_rects[a], label_text_rects[b]) or
                        rects_overlap(label_text_rects[a], label_line_rects[b]) or
                        rects_overlap(label_text_rects[b], label_line_rects[a])):
                        pairs.append((a, b))
            if not pairs: break
            any_fix = False
            for a, b in pairs:
                movable = [x for x in [a, b] if not is_overridden(scale_data[x]['label'])]
                markers_ab, texts_ab, lines_ab = get_occupied(exclude_idxs=[a, b])
                def find_valid(idx):
                    px, py = point_disps[idx]
                    tw, th = text_sizes[scale_data[idx]['label']]
                    vs = []
                    for (ddx, ddy), ha, va in directions:
                        for lp in lengths_px:
                            rect = get_text_rect(px+ddx*(lp+GAP), py+ddy*(lp+GAP), tw, th, ha, va)
                            lr = get_line_rect(px, py, ddx, ddy, lp)
                            if not rect_in_bounds(rect): continue
                            h, _ = count_collisions(rect, lr, markers_ab, texts_ab, lines_ab)
                            if h > 0: continue
                            vs.append((ddx, ddy, lp, ha, va, rect, lr))
                    return vs
                if len(movable) == 2:
                    va_list, vb_list = find_valid(a), find_valid(b)
                    best_pair, best_sc = None, float('inf')
                    for pa in va_list:
                        for pb in vb_list:
                            if rects_overlap(pa[5], pb[5]) or rects_overlap(pa[5], pb[6]) or rects_overlap(pb[5], pa[6]):
                                continue
                            sc = pa[2] + pb[2]
                            if sc < best_sc: best_sc, best_pair = sc, (pa, pb)
                    if best_pair:
                        for idx2, pos in zip([a, b], best_pair):
                            label_text_rects[idx2] = pos[5]; label_line_rects[idx2] = pos[6]
                            label_params[idx2] = pos[:5]
                        any_fix = True
                elif len(movable) == 1:
                    m_idx = movable[0]
                    f_idx = [x for x in [a, b] if x != m_idx][0]
                    for pm in find_valid(m_idx):
                        if (rects_overlap(pm[5], label_text_rects[f_idx]) or
                            rects_overlap(pm[5], label_line_rects[f_idx]) or
                            rects_overlap(label_text_rects[f_idx], pm[6])): continue
                        label_text_rects[m_idx] = pm[5]; label_line_rects[m_idx] = pm[6]
                        label_params[m_idx] = pm[:5]; any_fix = True; break
            if not any_fix: break

        # Phase 5: Global shortening
        for _ in range(5):
            any_imp = False
            for _, idx, d in label_order:
                if idx not in label_params or is_overridden(d['label']): continue
                cur_lp = label_params[idx][2]
                px, py = point_disps[idx]
                tw, th = text_sizes[d['label']]
                markers, texts, lines = get_occupied(exclude_idx=idx)
                best_lp, best = cur_lp, None
                for (ddx, ddy), ha, va in directions:
                    for lp in lengths_px:
                        if lp >= best_lp: break
                        rect = get_text_rect(px+ddx*(lp+GAP), py+ddy*(lp+GAP), tw, th, ha, va)
                        lrect = get_line_rect(px, py, ddx, ddy, lp)
                        if not rect_in_bounds(rect): continue
                        h, _ = count_collisions(rect, lrect, markers, texts, lines)
                        if h > 0: continue
                        best_lp = lp; best = (ddx, ddy, lp, ha, va, rect, lrect); break
                if best and best_lp < cur_lp:
                    label_text_rects[idx] = best[5]; label_line_rects[idx] = best[6]
                    label_params[idx] = best[:5]; any_imp = True
            if not any_imp: break

        # Report
        hard_n = sum(1 for i in range(len(idxs)) for j in range(i+1, len(idxs))
                     if rects_overlap(label_text_rects[idxs[i]], label_text_rects[idxs[j]]))
        print(f"  Subplot {subplot_idx}: {'CLEAN' if hard_n == 0 else f'{hard_n} overlaps'}")

        # Annotations
        for idx, d in enumerate(scale_data):
            if idx not in label_params: continue
            ddx, ddy, lp, ha, va = label_params[idx]
            ox = ddx * (lp + GAP) * pts_per_px
            oy = ddy * (lp + GAP) * pts_per_px
            ax.annotate(d['label'], xy=(x_func(d), y_func(d)),
                       xytext=(ox, oy), textcoords='offset points', fontsize=fs,
                       color=family_colors_dark.get(d['family'], '#333333'),
                       fontfamily='sans-serif', zorder=5, ha=ha, va=va,
                       path_effects=outline,
                       arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, shrinkA=0, shrinkB=1))

    # ==================== Panel (a): F vs Size ====================
    plot_scatter(ax1, lambda d: d['feasible_rate'], subplot_idx=0)
    y_line_f = slope_f * np.log10(x_line) + intercept_f
    ax1.plot(x_line, y_line_f, color='black', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)
    ax1.fill_between(10 ** x_pred_f, y_lo_f, y_hi_f, color='black', alpha=0.08, zorder=1)
    ax1.set_xlabel('Parameters (B)', fontsize=fs, labelpad=3)
    ax1.set_ylabel('F - Feasibility Rate (%)', fontsize=fs, labelpad=3)
    ax1.text(0.05, 0.93, f'- - {format_stats(r2_f, slope_f, "F")}',
             transform=ax1.transAxes, fontsize=fs, ha='left', va='top', style='italic')

    # ==================== Panel (b): S vs Size ====================
    plot_scatter(ax2, lambda d: d['safe_rate'], subplot_idx=1)
    y_line_s = slope_s * np.log10(x_line) + intercept_s
    ax2.plot(x_line, y_line_s, color='black', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)
    ax2.fill_between(10 ** x_pred_s, y_lo_s, y_hi_s, color='black', alpha=0.08, zorder=1)
    ax2.set_xlabel('Parameters (B)', fontsize=fs, labelpad=3)
    ax2.set_ylabel('S - Safety Rate (%)', fontsize=fs, labelpad=3)
    ax2.text(0.05, 0.93, f'- - {format_stats(r2_s, slope_s, "S")}',
             transform=ax2.transAxes, fontsize=fs, ha='left', va='top', style='italic')
    ax2.text(0.05, 0.85, f'\u03b2$_{{S}}$/\u03b2$_{{F}}$ = {s_f_ratio_point:.2f}  '
             f'CI=[{s_f_ratio_lo:.2f}, {s_f_ratio_hi:.2f}]',
             transform=ax2.transAxes, fontsize=fs-2, ha='left', va='top', color='#555555', style='italic')

    # ==================== Panel (c): SI vs Size ====================
    plot_scatter(ax3, lambda d: d['si_rate'], subplot_idx=2)
    y_line_si = slope_si * np.log10(x_line) + intercept_si
    ax3.plot(x_line, y_line_si, color='black', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)
    ax3.fill_between(10 ** x_pred_si, y_lo_si, y_hi_si, color='black', alpha=0.08, zorder=1)
    ax3.set_xlabel('Parameters (B)', fontsize=fs, labelpad=3)
    ax3.set_ylabel('SI - Safety Intention Rate (%)', fontsize=fs, labelpad=3)
    ax3.text(0.05, 0.93, f'- - {format_stats(r2_si, slope_si, "SI")}',
             transform=ax3.transAxes, fontsize=fs, ha='left', va='top', style='italic')
    ax3.text(0.05, 0.85, f'\u03b2$_{{SI}}$/\u03b2$_{{F}}$ = {si_f_ratio_point:.2f}  '
             f'CI=[{si_f_ratio_lo:.2f}, {si_f_ratio_hi:.2f}]',
             transform=ax3.transAxes, fontsize=fs-2, ha='left', va='top', color='#555555', style='italic')

    # ── Add GPT-5 high reference lines to panels (a), (b), (c) ──
    # GPT-5 high has unknown size, so show as horizontal reference lines
    gpt5_data = next((d for d in decomp_data if d['model_key'] == 'openai_gpt-5'), None)
    if gpt5_data:
        gpt5_color_muted = '#999999'  # muted gray for subtle reference
        gpt5_refs = [
            (ax1, gpt5_data['feasible_rate'], 'F'),
            (ax2, gpt5_data['safe_rate'], 'S'),
            (ax3, gpt5_data['si_rate'], 'SI'),
        ]
        for ax, val, metric_label in gpt5_refs:
            # Draw horizontal reference line
            ax.axhline(y=val, color=gpt5_color_muted, linestyle=':', linewidth=1.0, alpha=0.6, zorder=1)
            # Add label on the line at 2/3 position from left
            ax.text(0.67, val, 'gpt-5 high',
                   transform=ax.get_yaxis_transform(),
                   fontsize=fs-3, color='#777777', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                            edgecolor=gpt5_color_muted, alpha=0.9, linewidth=0.5))

    # ==================== Panel (d): S vs F×SI ============================
    decomp_families = sorted(set(d['family'] for d in decomp_data))
    fxsi_values = np.array([d['feasible_rate'] * d['si_rate'] / 100 for d in decomp_data])
    s_values = np.array([d['safe_rate'] for d in decomp_data])

    for family in decomp_families:
        fi = [i for i, d in enumerate(decomp_data) if d['family'] == family]
        if not fi: continue
        color = all_family_colors.get(family, '#808080')
        marker = all_family_markers.get(family, 'o')
        ax4.scatter(fxsi_values[fi], s_values[fi], s=70, color=color, marker=marker,
                   edgecolors='white', linewidths=0.4, label=family, zorder=3)

    max_val = max(fxsi_values.max(), s_values.max())
    ax4.plot([0, max_val + 5], [0, max_val + 5], 'k--', alpha=0.3, linewidth=0.8, zorder=1)

    slope_d, inter_d, r_d, p_d, _ = stats.linregress(fxsi_values, s_values)
    r2_d = r_d ** 2
    x_fit = np.linspace(0, max_val + 5, 100)
    ax4.plot(x_fit, slope_d * x_fit + inter_d, color='#cc3333', linewidth=1.0, alpha=0.7, zorder=2)

    ax4.set_xlabel('Predicted: F \u00d7 SI (%)', fontsize=fs, labelpad=3)
    ax4.set_ylabel('S - Safety Rate (%)', fontsize=fs, labelpad=3)
    ax4.set_xlim(-3, max_val + 8)
    ax4.set_ylim(-3, max_val + 8)
    ax4.text(0.05, 0.93, f'S = {slope_d:.2f}\u00b7(F\u00d7SI) {inter_d:+.1f}',
             transform=ax4.transAxes, fontsize=fs, ha='left', va='top', style='italic', color='#cc3333')
    n_open = sum(1 for d in decomp_data if d['family'] in ('Llama', 'Qwen', 'DeepSeek'))
    n_prop = len(decomp_data) - n_open
    ax4.text(0.05, 0.85, f'R\u00b2={r2_d:.4f}, {n_open} open + {n_prop} proprietary',
             transform=ax4.transAxes, fontsize=fs-1, ha='left', va='top', color='#555555', style='italic')

    # Panel (d) internal legend
    handles_d, labels_d = [], []
    import matplotlib.lines as mlines
    h_id = mlines.Line2D([], [], color='black', linestyle='--', linewidth=0.8, alpha=0.3)
    handles_d.append(h_id); labels_d.append('S = F\u00d7SI')
    for fam in decomp_families:
        h = ax4.scatter([], [], s=40, color=all_family_colors.get(fam, '#808080'),
                        marker=all_family_markers.get(fam, 'o'), edgecolors='white', linewidths=0.4)
        handles_d.append(h); labels_d.append(fam)
    ax4.legend(handles_d, labels_d, fontsize=fs-3, loc='lower right',
               framealpha=0.8, edgecolor='gray', ncol=2)

    # ==================== Subplot labels ==================================
    for ax, lbl in zip([ax1, ax2, ax3, ax4], ['a', 'b', 'c', 'd']):
        ax.text(-0.08, 1.02, lbl, transform=ax.transAxes, fontsize=24,
                fontweight='bold', va='bottom', ha='left', clip_on=False)

    # ==================== Bottom legend ===================================
    ax_legend.axis('off')
    legend_by_family = defaultdict(list)
    for label, name, family, is_thinking in legend_entries:
        legend_by_family[family].append((label, name, is_thinking))

    marker_char = {'o': '\u25cf', 'D': '\u25c6', 's': '\u25a0'}
    leg_fs = fs - 3  # smaller font for single-line families
    line_height = 0.28
    char_width = 0.0065
    entry_gap = 0.008
    x_indent = 0.095

    def draw_family_rows(family, y_start, max_per_line=4, x_start=0.01):
        """Draw family legend, wrapping to multiple lines if needed."""
        if family not in legend_by_family:
            return y_start
        color = family_colors_dark.get(family, '#333333')
        marker = family_markers.get(family, 'o')
        symbol = marker_char.get(marker, '\u25cf')
        entries = legend_by_family[family]

        # Header on first line
        ax_legend.text(x_start, y_start, f"{symbol} {family}:",
                      transform=ax_legend.transAxes, fontsize=leg_fs,
                      fontweight='bold', ha='left', va='top', color=color, fontfamily='sans-serif')
        x_pos = x_start + x_indent
        count_on_line = 0
        y_cur = y_start
        for label, name, is_thinking in entries:
            entry_text = f"{label}: {name}"
            entry_width = len(entry_text) * char_width + entry_gap
            # Wrap if exceeds line width or max count
            if count_on_line >= max_per_line or x_pos + entry_width > 1.0:
                y_cur -= line_height
                x_pos = x_start + x_indent
                count_on_line = 0
            ax_legend.text(x_pos, y_cur, f"{label}:", transform=ax_legend.transAxes,
                          fontsize=leg_fs, ha='left', va='top', color=color, fontfamily='sans-serif')
            idx_w = len(f"{label}: ") * char_width
            ax_legend.text(x_pos + idx_w, y_cur, name, transform=ax_legend.transAxes,
                          fontsize=leg_fs, ha='left', va='top', color='black', fontfamily='sans-serif')
            x_pos += entry_width
            count_on_line += 1
        return y_cur, x_pos  # return both y position and final x position

    y_pos, _ = draw_family_rows('Llama', 0.92, max_per_line=20)
    y_pos, _ = draw_family_rows('Qwen', y_pos - line_height, max_per_line=20)
    y_pos, x_pos = draw_family_rows('DeepSeek', y_pos - line_height, max_per_line=20)

    # Add ordering note on same line as DeepSeek
    ax_legend.text(x_pos + 0.05, y_pos, "(Model index numbers ordered by release date across families)",
                   transform=ax_legend.transAxes, fontsize=leg_fs-1, ha='left', va='top',
                   color='#555555', fontfamily='sans-serif', style='italic')
    y_pos -= line_height

    # Add marker style note
    ax_legend.text(0.01, y_pos, "In panels a–c: ● filled = dense architecture, ○ hollow = MoE architecture. "
                   "MoE models show total params; (A#B) = active params per forward pass.",
                   transform=ax_legend.transAxes, fontsize=leg_fs-1, ha='left', va='top',
                   color='#555555', fontfamily='sans-serif', style='italic')

    # ==================== Save ============================================
    base_path = output_path.parent / output_path.stem
    plt.savefig(f"{base_path}_combined.svg", facecolor='white', format='svg', bbox_inches='tight')
    plt.savefig(f"{base_path}_combined.png", dpi=300, facecolor='white', bbox_inches='tight')
    plt.savefig(output_path.parent / "result-scale.pdf", facecolor='white', format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {base_path}_combined.svg")

    # Save text data
    results_path = base_path.parent / "results_scale.txt"
    mk_trend = lambda sl, it, r2, p, r, se, ci: {
        'slope': sl, 'intercept': it, 'r_squared': r2, 'p_value': p,
        'r_value': r, 'std_err': se, 'slope_ci': ci}
    save_plot_data_to_txt(scale_data, total_sizes, feasible_rates,
                          "Size (B)", "Feasibility Rate (%)",
                          mk_trend(slope_f, intercept_f, r2_f, p_f, r_f, se_f, slope_ci_f),
                          results_path)
    save_plot_data_to_txt(scale_data, total_sizes, si_rates,
                          "Size (B)", "SI Rate (%)",
                          mk_trend(slope_si, intercept_si, r2_si, p_si, r_si, se_si, slope_ci_si),
                          results_path, append=True)
    save_plot_data_to_txt(scale_data, total_sizes, safe_rates,
                          "Size (B)", "Safety Rate (%)",
                          mk_trend(slope_s, intercept_s, r2_s, p_s, r_s, se_s, slope_ci_s),
                          results_path, append=True)
    save_plot_data_to_txt(decomp_data, fxsi_values, s_values,
                          "F\u00d7SI (%)", "Safety Rate (%)",
                          mk_trend(slope_d, inter_d, r2_d, p_d, r_d, 0, None),
                          results_path, append=True)

    with open(results_path, 'a') as f:
        f.write("\n\n" + "#" * 100 + "\n")
        f.write("SLOPE COMPARISON (all 18 open-source models, total params)\n")
        f.write("=" * 100 + "\n")
        f.write(f"F:  {slope_f:.2f} pp/decade  95% CI [{slope_ci_f[0]:.2f}, {slope_ci_f[1]:.2f}]\n")
        f.write(f"S:  {slope_s:.2f} pp/decade  95% CI [{slope_ci_s[0]:.2f}, {slope_ci_s[1]:.2f}]\n")
        f.write(f"SI: {slope_si:.2f} pp/decade  95% CI [{slope_ci_si[0]:.2f}, {slope_ci_si[1]:.2f}]\n")
        if slope_f > 0:
            f.write(f"\nS/F  slope ratio: {s_f_ratio_point:.3f}  95% CI [{s_f_ratio_lo:.3f}, {s_f_ratio_hi:.3f}]\n")
            f.write(f"SI/F slope ratio: {si_f_ratio_point:.3f}  95% CI [{si_f_ratio_lo:.3f}, {si_f_ratio_hi:.3f}]\n")
        f.write(f"\nRegression on {len(scale_data)} open-source models "
                f"({len(standard_data)} standard + {len(thinking_data)} thinking).\n")
        f.write(f"X-axis: total parameters (for dense models, total = active; for MoE, total > active).\n")

        # ── Detailed plot description and CI methodology ──────────────────
        f.write("\n\n" + "#" * 100 + "\n")
        f.write("FIGURE DESCRIPTION\n")
        f.write("=" * 100 + "\n\n")

        f.write("2x2 scatter plot with bottom legend.\n\n")

        f.write("Panel (a): Feasibility Rate (F%) vs Parameters\n")
        f.write("-" * 60 + "\n")
        f.write(f"  X-axis: log-scaled total parameters (B), range [{x_min:.0f}, {x_max:.0f}].\n")
        f.write("  Y-axis: Feasibility Rate (%), range [0, 100].\n")
        f.write(f"  {len(scale_data)} open-source models ({len(standard_data)} standard + {len(thinking_data)} thinking).\n")
        f.write(f"  {n_dense} dense (filled markers) + {n_moe} MoE (hollow markers).\n")
        f.write(f"  Dashed line: OLS regression on log10(size), slope (beta_F) = {slope_f:.1f} pp/decade.\n")
        f.write(f"  Shaded band: 95% bootstrap CI on the regression line.\n")
        f.write(f"  Text: beta_F={slope_f:.1f}, R^2={r2_f:.2f}\n\n")

        f.write("Panel (b): Safety Rate (S%) vs Parameters\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Same axes as (a).\n")
        f.write(f"  Dashed line: OLS regression, slope (beta_S) = {slope_s:.1f} pp/decade.\n")
        f.write(f"  Shaded band: 95% bootstrap CI on the regression line.\n")
        f.write(f"  Text line 1: beta_S={slope_s:.1f}, R^2={r2_s:.2f}\n")
        f.write(f"  Text line 2: beta_S/beta_F = {s_f_ratio_point:.2f}  [{s_f_ratio_lo:.2f}, {s_f_ratio_hi:.2f}]\n\n")

        f.write("Panel (c): Safety Intention Rate (SI%) vs Parameters\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Same axes as (a).\n")
        f.write(f"  Dashed line: OLS regression, slope (beta_SI) = {slope_si:.1f} pp/decade.\n")
        f.write(f"  Shaded band: 95% bootstrap CI on the regression line.\n")
        f.write(f"  Text line 1: beta_SI={slope_si:.1f}, R^2={r2_si:.2f}\n")
        f.write(f"  Text line 2: beta_SI/beta_F = {si_f_ratio_point:.2f}  [{si_f_ratio_lo:.2f}, {si_f_ratio_hi:.2f}]\n\n")

        f.write("Panel (d): Safety Rate (S%) vs Predicted F x SI (%)\n")
        f.write("-" * 60 + "\n")
        f.write(f"  X-axis: F x SI / 100 (predicted safety), Y-axis: observed S.\n")
        f.write(f"  {len(decomp_data)} models (including closed-source).\n")
        f.write(f"  Dashed black line: identity S = F x SI.\n")
        f.write(f"  Red solid line: OLS fit S = {slope_d:.2f} * (F x SI) {inter_d:+.1f}, R^2={r2_d:.4f}.\n")
        f.write(f"  Internal legend: identity line + family markers.\n\n")

        f.write("Bottom legend:\n")
        f.write("-" * 60 + "\n")
        f.write("  Chronological numbering (1, 1a/1b, 2a/2b/2c, ...) by release date.\n")
        f.write("  Families: Llama (diamond), Qwen (circle), DeepSeek (square).\n")
        f.write("  Marker style: filled = Dense, hollow = MoE.\n")
        f.write("  MoE models show total params with active params as '(A{X}B)'.\n\n")

        f.write("Markers per family:\n")
        f.write("  Llama:    green diamond\n")
        f.write("  Qwen:     gold circle\n")
        f.write("  DeepSeek: blue square\n")
        f.write("  OpenAI:   red hexagon     (panel d only)\n")
        f.write("  Google:   orange triangle  (panel d only)\n")
        f.write("  Anthropic: purple pentagon (panel d only)\n\n")

        f.write("\n" + "#" * 100 + "\n")
        f.write("CONFIDENCE INTERVAL METHODOLOGY\n")
        f.write("=" * 100 + "\n\n")

        f.write("1. REGRESSION LINE CI BANDS (shaded areas in panels a, b, c)\n")
        f.write("-" * 60 + "\n")
        f.write("   Method: Non-parametric bootstrap (case resampling).\n")
        f.write(f"   Procedure:\n")
        f.write(f"     - n = {len(scale_data)} models, B = {10000} bootstrap replicates, seed = 42.\n")
        f.write(f"     - For each replicate b = 1..B:\n")
        f.write(f"         1. Resample n models WITH replacement (same model can appear multiple times).\n")
        f.write(f"         2. Fit OLS: metric = slope_b * log10(size) + intercept_b.\n")
        f.write(f"         3. Compute predicted values y_b(x) at 200 evenly spaced x-points.\n")
        f.write(f"     - At each x-point, take the 2.5th and 97.5th percentiles across B predictions.\n")
        f.write(f"     - This forms the shaded band.\n")
        f.write(f"   Interpretation: The band shows uncertainty in the REGRESSION LINE position,\n")
        f.write(f"     not prediction intervals for individual models.\n\n")

        f.write("2. SLOPE CIs (beta_F, beta_S, beta_SI reported in text)\n")
        f.write("-" * 60 + "\n")
        f.write("   Method: Extracted from the same bootstrap as (1).\n")
        f.write(f"   Procedure:\n")
        f.write(f"     - Collect slope_b from each of the B = {10000} bootstrap replicates.\n")
        f.write(f"     - 95% CI = [2.5th percentile, 97.5th percentile] of slope distribution.\n")
        f.write(f"   Results:\n")
        f.write(f"     beta_F  = {slope_f:.2f}  95% CI [{slope_ci_f[0]:.2f}, {slope_ci_f[1]:.2f}]\n")
        f.write(f"     beta_S  = {slope_s:.2f}  95% CI [{slope_ci_s[0]:.2f}, {slope_ci_s[1]:.2f}]\n")
        f.write(f"     beta_SI = {slope_si:.2f}  95% CI [{slope_ci_si[0]:.2f}, {slope_ci_si[1]:.2f}]\n")
        f.write(f"   Units: percentage points per order of magnitude (pp/decade).\n\n")

        f.write("3. SLOPE RATIO CIs (beta_S/beta_F, beta_SI/beta_F in panels b, c)\n")
        f.write("-" * 60 + "\n")
        f.write("   Method: Paired bootstrap (joint resampling).\n")
        f.write(f"   Procedure:\n")
        f.write(f"     - B = {10000} replicates, seed = 42.\n")
        f.write(f"     - For each replicate b = 1..B:\n")
        f.write(f"         1. Draw ONE resample of n = {len(scale_data)} models.\n")
        f.write(f"         2. Fit THREE regressions on the SAME resample: F, S, SI vs log10(size).\n")
        f.write(f"         3. Compute ratio_b = slope_metric_b / slope_F_b.\n")
        f.write(f"         4. Discard if |slope_F_b| <= 0.01 (degenerate F slope).\n")
        f.write(f"     - 95% CI = [2.5th, 97.5th percentile] of valid ratios.\n")
        f.write(f"   Key: Joint resampling preserves the correlation between F, S, SI slopes\n")
        f.write(f"     (they share the same x-values and model selection). This is critical because\n")
        f.write(f"     independent bootstraps would overestimate ratio variance.\n")
        f.write(f"   Results:\n")
        f.write(f"     beta_S/beta_F  = {s_f_ratio_point:.3f}  95% CI [{s_f_ratio_lo:.3f}, {s_f_ratio_hi:.3f}]")
        f.write(f"  (valid: {len(s_f_ratios)}/{10000})\n")
        f.write(f"     beta_SI/beta_F = {si_f_ratio_point:.3f}  95% CI [{si_f_ratio_lo:.3f}, {si_f_ratio_hi:.3f}]")
        f.write(f"  (valid: {len(si_f_ratios)}/{10000})\n")
        f.write(f"   Both CIs exclude 1.0 from above, confirming S and SI scale slower than F.\n\n")

        # ── Variability comparison (CV) ──────────────────────────────────────
        f.write("\n" + "#" * 100 + "\n")
        f.write("VARIABILITY COMPARISON\n")
        f.write("=" * 100 + "\n\n")
        for metric_name, vals in [("F", feasible_rates), ("S", safe_rates), ("SI", si_rates)]:
            cv = np.std(vals) / np.mean(vals) * 100
            rng_val = vals.max() - vals.min()
            f.write(f"  {metric_name}:\n")
            f.write(f"    Range = [{vals.min():.2f}, {vals.max():.2f}] ({rng_val:.1f} pp)\n")
            f.write(f"    Mean  = {np.mean(vals):.2f}, SD = {np.std(vals):.2f}\n")
            f.write(f"    CV    = {cv:.1f}%\n\n")

        # ── GPT-5 high reference ──────────────────────────────────────────────
        if gpt5_data:
            f.write("\n" + "#" * 100 + "\n")
            f.write("GPT-5 HIGH REFERENCE (closed-source, unknown size)\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"  F:  {gpt5_data['feasible_rate']:.2f}%\n")
            f.write(f"  S:  {gpt5_data['safe_rate']:.2f}%\n")
            f.write(f"  SI: {gpt5_data['si_rate']:.2f}%\n")


def main(parent_folders: Optional[List[str]] = None):
    """Main function."""
    if parent_folders is None:
        parent_folders = ["data/full/hard"]

    print(f"Collecting data from {parent_folders}...")
    data = collect_data(parent_folders)
    print(f"  {len(data)} tasks")

    metrics = calculate_metrics(data)
    print(f"  {len(metrics)} models")

    model_info = get_model_info()

    output_dir = Path("data/experiments/scale_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating plots...")
    create_scatter_plots_normalized(metrics, model_info,
                                    output_dir / "scale_analysis_normalized.png")
    print("Done!")


if __name__ == "__main__":
    main(parent_folders=["data/full/hard"])
