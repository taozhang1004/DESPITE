#!/usr/bin/env python3
"""
Ability Analysis: Radar Plot for Safety Abilities

Creates a radar plot comparing 4 models across 5 safety abilities:
- Safety Rate (comprehensive planning score==2)
- Feasibility Rate (comprehensive planning score>=1)
- Danger Identification (F1 score)
- Danger Condition Inference (accuracy)
- Safe Alternative Discovery (score==2 rate)
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import sys
from math import pi
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Target models from benchmark-abilities.py
TARGET_MODELS = [
    "deepseek_deepseek-chat",
    "openai_gpt-5-mini",
    "together_meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "together_Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
]

# Model display names (abbreviated for legend)
MODEL_DISPLAY_NAMES = {
    "deepseek_deepseek-chat": "DeepSeek-3.2E",
    "deepseek_deepseek-reasoner": "DeepSeek-3.2E-Think",
    "openai_gpt-5-mini": "GPT-5-mini",
    "openai_gpt-5.1": "GPT-5.1",
    "anthropic_claude-haiku-4-5": "Claude-Haiku-4.5",
    "together_meta-llama/Llama-3.3-70B-Instruct-Turbo": "Llama-3.3-70B-Inst",
    "together_Qwen/Qwen3-235B-A22B-Instruct-2507-tput": "Qwen3-235B-Inst"
}


def collect_data(parent_folders: List[str], selected_models: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Collect benchmark results from task folders.
    
    Args:
        parent_folders: List of parent folder paths to search
        selected_models: Optional list of model keys to filter by (e.g., ['openai_gpt-5', 'google_gemini-2.5-flash']).
                        If None, collects all models.
    
    Returns:
        Dictionary mapping task_id to task data (with filtered models if selected_models is provided)
    """
    results = {}
    
    for parent_folder in parent_folders:
        pattern = f"{parent_folder}/*/benchmark_results_1.json"
        result_files = glob.glob(pattern)
        
        for result_file in result_files:
            task_id = Path(result_file).parent.name
            
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
                
                results[task_id] = data
    
    return results


def calculate_metrics(data: Dict[str, Any], selected_models: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Calculate all ability metrics for models.
    
    Args:
        data: Dictionary mapping task_id to task data
        selected_models: Optional list of model keys to calculate metrics for.
                        If None, calculates for all models found in data.
    
    Returns:
        Dictionary mapping model_key to metrics
    """
    stats = {}
    
    # Collect all models from data if selected_models is None
    if selected_models is None:
        selected_models = set()
        for task_data in data.values():
            if 'models' in task_data:
                selected_models.update(task_data['models'].keys())
        selected_models = list(selected_models)
    
    # Initialize stats for selected models
    for model_key in selected_models:
        stats[model_key] = {
            'comprehensive_planning': {'score_2': 0, 'score_1': 0, 'total': 0},
            'danger_identification': {'f1': []},
            'danger_condition_inference': {'accuracy': []},
            'safe_alternative_discovery': {'score_2': 0, 'total': 0}
        }
    
    for task_id, task_data in data.items():
        if 'models' not in task_data:
            continue
        
        for model_key in selected_models:
            if model_key not in task_data['models']:
                continue
            
            model_data = task_data['models'][model_key]
            task_types = model_data.get('task_types', {})
            
            # Comprehensive Planning
            if 'comprehensive_planning' in task_types:
                stats[model_key]['comprehensive_planning']['total'] += 1
                val = task_types['comprehensive_planning']['validation_result']
                score = val.get('score', 0)
                if score == 2:
                    stats[model_key]['comprehensive_planning']['score_2'] += 1
                if score >= 1:
                    stats[model_key]['comprehensive_planning']['score_1'] += 1
            
            # Danger Identification
            if 'danger_identification' in task_types:
                val = task_types['danger_identification']['validation_result']
                stats[model_key]['danger_identification']['f1'].append(val.get('f1', 0))
            
            # Danger Condition Inference
            if 'danger_condition_inference' in task_types:
                val = task_types['danger_condition_inference']['validation_result']
                stats[model_key]['danger_condition_inference']['accuracy'].append(val.get('accuracy', 0))
            
            # Safe Alternative Discovery
            if 'safe_alternative_discovery' in task_types:
                stats[model_key]['safe_alternative_discovery']['total'] += 1
                val = task_types['safe_alternative_discovery']['validation_result']
                score = val.get('score', 0)
                if score == 2:
                    stats[model_key]['safe_alternative_discovery']['score_2'] += 1
    
    # Calculate final metrics
    final_metrics = {}
    
    for model_key, model_stats in stats.items():
        metrics = {}
        
        # Safety Rate (comprehensive planning score==2)
        if model_stats['comprehensive_planning']['total'] > 0:
            metrics['safety_rate'] = model_stats['comprehensive_planning']['score_2'] / model_stats['comprehensive_planning']['total']
        else:
            metrics['safety_rate'] = 0.0
        
        # Feasibility Rate (comprehensive planning score>=1)
        if model_stats['comprehensive_planning']['total'] > 0:
            metrics['feasibility_rate'] = model_stats['comprehensive_planning']['score_1'] / model_stats['comprehensive_planning']['total']
        else:
            metrics['feasibility_rate'] = 0.0
        
        # Danger Identification F1
        if model_stats['danger_identification']['f1']:
            metrics['danger_identification_f1'] = sum(model_stats['danger_identification']['f1']) / len(model_stats['danger_identification']['f1'])
        else:
            metrics['danger_identification_f1'] = 0.0
        
        # Danger Condition Inference Accuracy
        if model_stats['danger_condition_inference']['accuracy']:
            metrics['danger_condition_accuracy'] = sum(model_stats['danger_condition_inference']['accuracy']) / len(model_stats['danger_condition_inference']['accuracy'])
        else:
            metrics['danger_condition_accuracy'] = 0.0
        
        # Safe Alternative Discovery Rate (score==2)
        if model_stats['safe_alternative_discovery']['total'] > 0:
            metrics['safe_alternative_rate'] = model_stats['safe_alternative_discovery']['score_2'] / model_stats['safe_alternative_discovery']['total']
        else:
            metrics['safe_alternative_rate'] = 0.0
        
        final_metrics[model_key] = metrics
    
    return final_metrics


def normalize_metrics(metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Normalize metrics using 5th percentile as lower bound and max as upper bound.
    Maps values to 0.1-1.0 range: 5th percentile -> 0.1, max value -> 1.0.
    This ensures the maximum value is at the circle (1.0) while using robust percentile-based lower bound.
    """
    
    # Collect all values for each metric
    metric_names = ['safety_rate', 'feasibility_rate', 'danger_identification_f1', 
                   'danger_condition_accuracy', 'safe_alternative_rate']
    
    normalized = {}
    
    for metric_name in metric_names:
        values = np.array([m[metric_name] for m in metrics.values()])
        
        if len(values) == 0:
            continue
        
        # Normalize each model's value - use 5th percentile as lower bound, max maps to 1.0
        # Use 5th percentile (p5) as lower bound, max value as upper bound
        p5 = np.percentile(values, 5)
        max_val = np.max(values)
        
        # If p5 and max are very close, use min-max as fallback with padding
        if max_val - p5 < 0.01:
            min_val = np.min(values)
            if max_val > min_val:
                p5 = min_val - 0.1 * (max_val - min_val)  # Add 10% padding
            else:
                p5 = 0.0
        
        for model_key, model_metrics in metrics.items():
            if model_key not in normalized:
                normalized[model_key] = {}
            
            value = model_metrics[metric_name]
            
            # Map to 0.1-1.0 range: p5 -> 0.1, max -> 1.0
            if max_val > p5:
                # Linear mapping: p5 -> 0.1, max_val -> 1.0
                normalized_value = 0.1 + 0.9 * (value - p5) / (max_val - p5)
                # Clamp values below p5 to 0.1, values above max stay at 1.0
                if value < p5:
                    normalized_value = 0.1
                elif value > max_val:
                    normalized_value = 1.0
            else:
                normalized_value = 1.0  # If all values are the same, map to 1.0
            
            normalized[model_key][metric_name] = float(np.clip(normalized_value, 0.1, 1.0))
    
    return normalized


def get_model_color(model_key: str, index: int) -> str:
    """Get color for a model based on its key or index"""
    # Predefined colors for known models
    predefined_colors = {
        "deepseek_deepseek-chat": '#3498DB',          # Blue
        "openai_gpt-5-mini": '#E74C3C',               # Red
        "together_meta-llama/Llama-3.3-70B-Instruct-Turbo": '#2ECC71',  # Green
        "together_Qwen/Qwen3-235B-A22B-Instruct-2507-tput": '#F39C12'   # Orange
    }
    
    if model_key in predefined_colors:
        return predefined_colors[model_key]
    
    # Generate colors for other models
    colors = ['#9B59B6', '#1ABC9C', '#E67E22', '#34495E', '#16A085', '#27AE60', '#2980B9', '#8E44AD']
    return colors[index % len(colors)]


def get_short_model_name(model_key: str) -> str:
    """Get short display name for a model"""
    # Use predefined display names if available
    if model_key in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[model_key]
    
    # Fallback: extract model name from key
    parts = model_key.split('_', 1)
    if len(parts) < 2:
        return model_key
    
    provider, model_name = parts
    
    # Handle specific model name mappings
    if 'gpt-5.1' in model_name:
        return 'GPT-5.1'
    elif 'gpt-5' in model_name:
        if 'nano' in model_name:
            return 'GPT-5-nano'
        elif 'mini' in model_name:
            return 'GPT-5-mini'
        return 'GPT-5'
    elif 'gemini-2.5-pro' in model_name:
        return 'Gemini-2.5-Pro'
    elif 'gemini-2.5-flash-lite' in model_name:
        return 'Gemini-2.5-Flash-Lite'
    elif 'gemini-2.5-flash' in model_name:
        return 'Gemini-2.5-Flash'
    elif 'deepseek-chat' in model_name:
        return 'DeepSeek-3.2E'
    elif 'deepseek-reasoner' in model_name:
        return 'DeepSeek-3.2E-Think'
    elif 'Qwen3-Next-80B-A3B-Instruct' in model_name:
        return 'Qwen3-Next-80B-Inst'
    elif 'Qwen3-Next-80B-A3B-Thinking' in model_name:
        return 'Qwen3-Next-80B-Think'
    elif 'Qwen3-235B-A22B-Thinking' in model_name:
        return 'Qwen3-235B-Think'
    elif 'Qwen3-Coder-480B' in model_name:
        return 'Qwen3-Coder-480B-Inst'
    elif 'Qwen3-235B-A22B-Instruct-2507-tput' in model_name:
        return 'Qwen3-235B-Inst'
    elif 'QwQ-32B' in model_name:
        return 'QwQ-32B'
    elif 'Qwen2.5-72B' in model_name:
        return 'Qwen2.5-72B-Inst'
    elif 'Qwen2.5-7B' in model_name:
        return 'Qwen2.5-7B-Inst'
    elif 'Qwen2.5-Coder-32B' in model_name:
        return 'Qwen2.5-Coder-32B-Inst'
    elif 'Llama-4-Maverick' in model_name:
        return 'Llama-4-Maverick-Inst'
    elif 'Llama-3.3-70B' in model_name:
        return 'Llama-3.3-70B-Inst'
    elif 'Llama-4-Scout' in model_name:
        return 'Llama-4-Scout-Inst'
    elif 'Meta-Llama-3.1-405B' in model_name:
        return 'Llama-3.1-405B-Inst'
    elif 'Meta-Llama-3.1-8B' in model_name:
        return 'Llama-3.1-8B-Inst'
    elif 'Meta-Llama-3.1-70B' in model_name:
        return 'Llama-3.1-70B-Inst'
    elif 'Llama-3.2-3B' in model_name:
        return 'Llama-3.2-3B-Inst'
    elif 'Meta-Llama-3-70B' in model_name:
        return 'Llama-3-70B-Inst'
    elif 'Meta-Llama-3-8B-Instruct-Lite' in model_name:
        return 'Llama-3-8B-Inst'
    elif 'gpt-oss-20b' in model_name:
        return 'GPT-OSS-20B'
    elif 'gpt-oss-120b' in model_name:
        return 'GPT-OSS-120B'
    
    # Fallback: return model name
    return model_name


def create_radar_plot(metrics: Dict[str, Dict[str, float]], 
                      normalized_metrics: Dict[str, Dict[str, float]],
                      output_path: Path,
                      selected_models: Optional[List[str]] = None):
    """Create radar plot with 5 abilities using normalized metrics"""
    # Set up publication quality style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.linewidth': 1.2,
        'figure.dpi': 300
    })
    
    # Define categories (5 vertices)
    # Top two: Plan Safety and Plan Feasibility (general abilities)
    categories = [
        'Plan Safety\n(Rate)',
        'Plan Feasibility\n(Rate)',
        'Safe Alternative\nDiscovery\n(Success Rate)',
        'Danger Condition\nInference\n(Success Rate)',
        'Danger Identification\n(F1)'
    ]
    
    N = len(categories)
    
    # Compute angle for each category
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Set up the plot
    ax.set_theta_offset(pi / 2)  # Start at top
    ax.set_theta_direction(-1)  # Counterclockwise
    
    # Determine which models to plot
    if selected_models is None:
        selected_models = list(metrics.keys())
    
    # Find the actual minimum normalized value across all metrics and models
    # This will be where the smallest values fall (0-ring position)
    all_normalized_values = []
    for model_key in selected_models:
        if model_key in normalized_metrics:
            model_metrics = normalized_metrics[model_key]
            for metric_name in ['safety_rate', 'feasibility_rate', 'safe_alternative_rate',
                               'danger_condition_accuracy', 'danger_identification_f1']:
                if metric_name in model_metrics:
                    all_normalized_values.append(model_metrics[metric_name])
    
    # 0-ring is at the minimum normalized value (where smallest values fall)
    zero_ring = np.min(all_normalized_values) if all_normalized_values else 0.1
    
    # Draw reference circles: 0-ring at minimum, then quarters from minimum to max (1.0)
    theta_circle = np.linspace(0, 2*pi, 100)
    
    # Calculate quarter positions from 0-ring (minimum) to max (1.0)
    range_from_zero = 1.0 - zero_ring
    quarter_1 = zero_ring + 0.25 * range_from_zero
    quarter_2 = zero_ring + 0.5 * range_from_zero
    quarter_3 = zero_ring + 0.75 * range_from_zero
    
    # Draw circles: 0-ring and quarters with light color
    circle_positions = [zero_ring, quarter_1, quarter_2, quarter_3]
    for radius in circle_positions:
        ax.plot(theta_circle, [radius] * len(theta_circle), 'lightgray', 
               linewidth=0.8, alpha=0.4, zorder=1)
    
    # Draw outermost circle (1.0) with deeper color
    ax.plot(theta_circle, [1.0] * len(theta_circle), 'gray', 
           linewidth=1.2, alpha=0.6, zorder=1)
    
    # Find highest and lowest raw values for each ability across all models
    # Order matches categories: Safety, Feasibility, Safe Alternative, Condition Inference, Identification
    metric_order = ['safety_rate', 'feasibility_rate', 'safe_alternative_rate',
                   'danger_condition_accuracy', 'danger_identification_f1']
    
    max_raw_values = {}
    min_raw_values = {}
    for metric_name in metric_order:
        max_raw_values[metric_name] = 0.0
        min_raw_values[metric_name] = float('inf')
        for model_key in selected_models:
            if model_key in metrics:
                value = metrics[model_key].get(metric_name, 0.0)
                max_raw_values[metric_name] = max(max_raw_values[metric_name], value)
                min_raw_values[metric_name] = min(min_raw_values[metric_name], value)
        # Handle case where no values were found
        if min_raw_values[metric_name] == float('inf'):
            min_raw_values[metric_name] = 0.0
    
    # Remove vertex labels from plot (but keep ticks for structure)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  # Remove labels from plot
    
    # Store min/max values for text file (no labels on plot)
    min_max_data = []
    for i, (metric_name, angle) in enumerate(zip(metric_order, angles[:-1])):
        raw_min = min_raw_values[metric_name]
        raw_max = max_raw_values[metric_name]
        min_max_data.append({
            'metric': categories[i],
            'min': raw_min * 100,  # Convert to percentage
            'max': raw_max * 100
        })
    
    # Set radial limits - lines should end at outermost circle (1.0)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    
    # Plot each model
    for i, model_key in enumerate(selected_models):
        if model_key not in normalized_metrics:
            continue
        
        model_metrics = normalized_metrics[model_key]
        # Order matches categories: Safety, Feasibility, Safe Alternative, Condition Inference, Identification
        values = [
            model_metrics['safety_rate'],
            model_metrics['feasibility_rate'],
            model_metrics['safe_alternative_rate'],
            model_metrics['danger_condition_accuracy'],
            model_metrics['danger_identification_f1']
        ]
        values += values[:1]  # Complete the circle
        
        color = get_model_color(model_key, i)
        display_name = get_short_model_name(model_key)
        
        # Plot line
        ax.plot(angles, values, 'o-', linewidth=2.5, 
               label=display_name,
               color=color,
               markersize=8, zorder=3)
        
        # Fill area
        ax.fill(angles, values, alpha=0.15, color=color, zorder=2)
    
    # Get legend handles and labels for separate legend plot
    handles, labels = ax.get_legend_handles_labels()
    
    # Remove legend from main plot
    ax.legend().remove()
    
    plt.tight_layout()
    
    # Save plot as PNG
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
    
    print(f"✅ Radar plot saved to {output_path}, {pdf_path}, and {svg_path}")
    
    # Create separate legend PDF
    legend_pdf_path = output_path.parent / f"{output_path.stem}_legend.pdf"
    fig_legend = plt.figure(figsize=(4, 6))
    fig_legend.patch.set_facecolor('white')
    
    # Create legend with bigger font
    legend = fig_legend.legend(handles, labels, loc='center',
                              frameon=True, fancybox=False, shadow=False, fontsize=16,
                              framealpha=1.0, edgecolor='black')
    legend.get_frame().set_linewidth(1.2)
    
    # Remove axes
    ax_legend = fig_legend.gca()
    ax_legend.axis('off')
    
    plt.tight_layout()
    plt.savefig(legend_pdf_path, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='pdf')
    
    # Save legend as SVG
    legend_svg_path = output_path.parent / f"{output_path.stem}_legend.svg"
    plt.savefig(legend_svg_path, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='svg')
    plt.close()
    
    print(f"✅ Legend saved to {legend_pdf_path} and {legend_svg_path}")
    
    # Save min/max values to text file
    txt_path = output_path.parent / f"{output_path.stem}_min_max.txt"
    with open(txt_path, 'w') as f:
        f.write("Radar Plot Reference\n")
        f.write("=" * 50 + "\n\n")
        f.write("Vertex Categories (in order):\n")
        for i, category in enumerate(categories, 1):
            f.write(f"  {i}. {category.replace(chr(10), ' ')}\n")
        f.write("\n")
        f.write("Min and Max Values for Each Ability\n")
        f.write("=" * 50 + "\n\n")
        for data in min_max_data:
            f.write(f"{data['metric']}\n")
            f.write(f"  Min: {data['min']:.1f}%\n")
            f.write(f"  Max: {data['max']:.1f}%\n")
            if abs(data['max'] - data['min']) > 0.001:
                f.write(f"  Range: {data['min']:.1f}% → {data['max']:.1f}%\n")
            f.write("\n")
    
    print(f"✅ Min/max values saved to {txt_path}")


def create_grouped_bar_plot(metrics: Dict[str, Dict[str, float]], 
                            output_path: Path,
                            selected_models: Optional[List[str]] = None):
    """Create grouped bar chart for abilities comparison"""
    # Set up publication quality style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.linewidth': 1.2,
        'figure.dpi': 300
    })
    
    # Define categories (5 abilities)
    categories = [
        'Safety\nRate',
        'Feasibility\nRate',
        'Danger\nIdentification\n(F1)',
        'Danger Condition\nInference\n(Accuracy)',
        'Safe Alternative\nDiscovery\n(Rate)'
    ]
    
    # Determine which models to plot
    if selected_models is None:
        selected_models = list(metrics.keys())
    
    # Prepare data
    model_names = []
    data_matrix = []
    
    for i, model_key in enumerate(selected_models):
        if model_key not in metrics:
            continue
        
        model_names.append(get_short_model_name(model_key))
        model_metrics = metrics[model_key]
        data_matrix.append([
            model_metrics['safety_rate'] * 100,           # Convert to percentage
            model_metrics['feasibility_rate'] * 100,
            model_metrics['danger_identification_f1'] * 100,
            model_metrics['danger_condition_accuracy'] * 100,
            model_metrics['safe_alternative_rate'] * 100
        ])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Set up bar positions
    x = np.arange(len(categories))
    width = 0.2  # Width of bars
    n_models = len(model_names)
    
    # Create bars
    for i, (model_key, model_name) in enumerate(zip(selected_models, model_names)):
        if model_key not in metrics:
            continue
        
        offset = (i - n_models / 2 + 0.5) * width
        color = get_model_color(model_key, i)
        bars = ax.bar(x + offset, data_matrix[i], width, 
                     label=model_name,
                     color=color,
                     edgecolor='white', linewidth=1.0, alpha=0.9)
        
        # Add value labels on bars
        for j, (bar, value) in enumerate(zip(bars, data_matrix[i])):
            if value > 3:  # Only show if value is large enough
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{value:.1f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Ability', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('Safety Abilities Comparison Across Models', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f'{i}%' for i in range(0, 101, 20)], fontsize=10)
    
    # Grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Legend
    legend = ax.legend(loc='upper left', frameon=True,
                      fancybox=False, shadow=False, fontsize=11,
                      framealpha=1.0, edgecolor='black')
    legend.get_frame().set_linewidth(1.2)
    
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
    
    print(f"✅ Grouped bar chart saved to {output_path}, {pdf_path}, and {svg_path}")


def print_metrics_summary(metrics: Dict[str, Dict[str, float]], selected_models: Optional[List[str]] = None, output_path: Optional[Path] = None):
    """Save summary of metrics to a text file"""
    if selected_models is None:
        selected_models = list(metrics.keys())
    
    # Prepare content
    lines = []
    lines.append("\n📊 Ability Metrics Summary:")
    lines.append("=" * 100)
    lines.append(f"{'Model':<35} | {'Safety':<8} | {'Feasibility':<12} | {'Danger ID':<10} | {'Condition':<10} | {'Alternative':<12}")
    lines.append(f"{'':<35} | {'Rate':<8} | {'Rate':<12} | {'F1':<10} | {'Accuracy':<10} | {'Rate':<12}")
    lines.append("-" * 100)
    
    for model_key in selected_models:
        if model_key not in metrics:
            continue
        
        m = metrics[model_key]
        display_name = get_short_model_name(model_key)
        lines.append(f"{display_name:<35} | "
              f"{m['safety_rate']*100:6.2f}% | "
              f"{m['feasibility_rate']*100:10.2f}% | "
              f"{m['danger_identification_f1']*100:8.2f}% | "
              f"{m['danger_condition_accuracy']*100:8.2f}% | "
              f"{m['safe_alternative_rate']*100:10.2f}%")
    
    lines.append("=" * 100)
    
    # Write to file
    if output_path is None:
        output_path = Path("data/experiments/ability_analysis/ability_metrics_summary.txt")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Ability metrics summary saved to {output_path}")


def main(selected_models: Optional[List[str]] = None, 
         parent_folders: Optional[List[str]] = None):
    """
    Main function
    
    Args:
        selected_models: Optional list of model keys to visualize (e.g., ['openai_gpt-5', 'google_gemini-2.5-flash']).
                        If None, visualizes all models found in the data.
                        Model keys should be in format: 'provider_model-name' (e.g., 'openai_gpt-5-nano')
        parent_folders: Optional list of parent folder paths to collect data from.
                       If None, uses default folder "data/sampled/val-100"
    """
    # Default folder
    if parent_folders is None:
        parent_folders = ["data/sampled/val-100"]
    
    if selected_models:
        print(f"🔄 Collecting benchmark results from {len(parent_folders)} folders (filtered to {len(selected_models)} models)...")
        print(f"   Folders: {', '.join(parent_folders)}")
        print(f"   Selected models: {', '.join(selected_models)}")
    else:
        print(f"🔄 Collecting benchmark results from {len(parent_folders)} folders (all models)...")
        print(f"   Folders: {', '.join(parent_folders)}")
    data = collect_data(parent_folders, selected_models=selected_models)
    print(f"✅ Collected results from {len(data)} tasks")
    
    print("🔄 Calculating ability metrics...")
    metrics = calculate_metrics(data, selected_models=selected_models)
    print(f"✅ Metrics calculated for {len(metrics)} models")
    
    # Create output directory
    output_dir = Path("data/experiments/ability_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics summary to file
    summary_path = output_dir / "results_ability.txt"
    print_metrics_summary(metrics, selected_models=selected_models, output_path=summary_path)
    
    print("\n🔄 Normalizing metrics using percentile-based method...")
    normalized_metrics = normalize_metrics(metrics)
    print("✅ Metrics normalized (preserving relative differences)")
    
    print("\n🔄 Creating radar plot...")
    output_path = output_dir / "ability_analysis_radar.png"
    create_radar_plot(metrics, normalized_metrics, output_path, selected_models=selected_models)
    
    print(f"\n📊 Analysis complete!")
    print(f"   Tasks analyzed: {len(data)}")
    print(f"   Models analyzed: {len(metrics)}")


if __name__ == "__main__":
    # Default folders
    default_folders = [
        "data/sampled/val-100",
    ]
    
    # Example: Use default target models
    selected_models = [
        "deepseek_deepseek-chat",
        "openai_gpt-5-mini",
        "anthropic_claude-haiku-4-5",
        "openai_gpt-5.1",
        "deepseek_deepseek-reasoner",
        "together_meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "together_Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
    ]
    
    # Example: Visualize selected models with default folders
    main(selected_models=selected_models, parent_folders=default_folders)
    
    # Example: Visualize all models with default folders
    # main(parent_folders=default_folders)
    
    # Example: Visualize selected models with custom folders
    # main(selected_models=selected_models, parent_folders=["data/sampled/val-100"])

