#!/usr/bin/env python3
"""
Redundancy analysis showing how redundant objects and actions affect LLM planning performance.
Tracks performance trends across different redundancy levels for multiple models.
"""

import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


class RedundancyAnalysis:
    def __init__(self, output_dir: Path, run_numbers: Optional[List[int]] = None, selected_models: Optional[List[str]] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # If run_numbers is None, use all available runs
        # Otherwise, filter to only use the specified run numbers
        self.run_numbers = run_numbers
        # If selected_models is None, use all models
        # Otherwise, filter to only use the specified models (model keys like 'openai_gpt-5', 'together_meta-llama/Llama-3.3-70B-Instruct-Turbo')
        self.selected_models = selected_models

    def collect_scores(self, baseline_dir: str, experiments_dir: str) -> pd.DataFrame:
        """Collect scores from benchmark_results_1.json files for all models"""
        all_results = []
        
        baseline_path = Path(baseline_dir)
        experiments_path = Path(experiments_dir)
        
        # Collect baseline results
        print(f"📂 Collecting baseline from {baseline_path}")
        if self.run_numbers is not None:
            print(f"   Using runs: {self.run_numbers}")
        if self.selected_models is not None:
            print(f"   Filtering to {len(self.selected_models)} models: {', '.join(self.selected_models)}")
        baseline_results = self._collect_from_directory(baseline_path, redundancy_type="baseline", redundancy_level=0)
        all_results.extend(baseline_results)
        
        # Collect experiment results
        print(f"📂 Collecting experiments from {experiments_path}")
        
        # Find all experiment directories (obj2, obj4, ..., act2, act4, ...)
        for exp_dir in sorted(experiments_path.iterdir()):
            if not exp_dir.is_dir():
                continue
            
            exp_name = exp_dir.name
            if exp_name.startswith("obj"):
                redundancy_type = "objects"
                redundancy_level = int(exp_name[3:])  # Extract number after "obj"
            elif exp_name.startswith("act"):
                redundancy_type = "actions"
                redundancy_level = int(exp_name[3:])  # Extract number after "act"
            else:
                continue  # Skip unknown directories
            
            print(f"   Processing {exp_name}...")
            exp_results = self._collect_from_directory(
                exp_dir, 
                redundancy_type=redundancy_type, 
                redundancy_level=redundancy_level
            )
            all_results.extend(exp_results)
        
        return pd.DataFrame(all_results)

    def _collect_from_directory(self, data_path: Path, redundancy_type: str, redundancy_level: int) -> List[Dict]:
        """Collect results from a single directory (baseline or experiment)
        Handles multiple benchmark_results files (benchmark_results_1.json, benchmark_results_2.json, etc.)
        Returns per-task data to allow proper run-level aggregation later
        """
        results = []
        
        if not data_path.exists():
            print(f"⚠️  Directory not found: {data_path}")
            return results
        
        task_count = 0
        # Structure: model -> run_number -> list of task results
        all_model_runs = {}
        
        for task_dir in data_path.iterdir():
            if not task_dir.is_dir():
                continue
            
            # Find all benchmark_results files
            all_benchmark_files = sorted(task_dir.glob("benchmark_results_*.json"))
            
            # Filter by run numbers if specified
            if self.run_numbers is not None:
                benchmark_files = [
                    f for f in all_benchmark_files
                    if self._extract_run_number(f.name) in self.run_numbers
                ]
            else:
                benchmark_files = all_benchmark_files
            
            if not benchmark_files:
                continue
            
            task_id = task_dir.name
            
            for benchmark_file in benchmark_files:
                try:
                    run_number = self._extract_run_number(benchmark_file.name)
                    with open(benchmark_file) as f:
                        data = json.load(f)
                    
                    # Extract results for all models
                    models = data.get("models", {})
                    
                    for model_key, model_data in models.items():
                        # Filter by selected_models if specified
                        if self.selected_models is not None and model_key not in self.selected_models:
                            continue
                        
                        # Get comprehensive_planning results
                        task_types = model_data.get("task_types", {})
                        comprehensive = task_types.get("comprehensive_planning", {})
                        
                        if not comprehensive:
                            continue
                        
                        validation_result = comprehensive.get("validation_result", {})
                        if not validation_result:
                            continue
                        
                        feasible = validation_result.get("feasible", False)
                        safe = validation_result.get("safe", False)
                        score = validation_result.get("score", 0)
                        safety_intention = validation_result.get("safety_intention", False)
                        
                        # Extract model name for display
                        model_display = self._extract_model_display_name(model_key)
                        
                        # Initialize structure
                        if model_display not in all_model_runs:
                            all_model_runs[model_display] = {}
                        if run_number not in all_model_runs[model_display]:
                            all_model_runs[model_display][run_number] = []
                        
                        # Store per-task data for this run
                        all_model_runs[model_display][run_number].append({
                            "redundancy_type": redundancy_type,
                            "redundancy_level": redundancy_level,
                            "task_id": task_id,
                            "model_key": model_key,
                            "model_display": model_display,
                            "score": score,
                            "feasible": feasible,
                            "safe": safe,
                            "safety_intention": safety_intention
                        })
                
                except Exception as e:
                    print(f"⚠️  Error reading {benchmark_file}: {e}")
                    continue
            
            task_count += 1
        
        # Now calculate run-level rates: for each run, calculate rate across all tasks
        # Then we can compute mean and std of those run-level rates
        for model_display, runs_dict in all_model_runs.items():
            if not runs_dict:
                continue
            
            # Collect run-level rates
            run_rates_feasible = []
            run_rates_safe = []
            run_rates_score = []
            run_rates_si = []
            model_key = None  # Will be set from first task result
            
            for run_number, task_results in runs_dict.items():
                if not task_results:
                    continue
                
                # Save model_key from first task result (same for all runs)
                if model_key is None:
                    model_key = task_results[0]['model_key']
                
                # Calculate rate across all tasks for this run
                feasible_rate = sum(r['feasible'] for r in task_results) / len(task_results)
                safe_rate = sum(r['safe'] for r in task_results) / len(task_results)
                score_avg = sum(r['score'] for r in task_results) / len(task_results)
                si_rate = sum(r['safety_intention'] for r in task_results) / len(task_results)

                run_rates_feasible.append(feasible_rate)
                run_rates_safe.append(safe_rate)
                run_rates_score.append(score_avg)
                run_rates_si.append(si_rate)
            
            # Calculate mean and std across runs
            if not run_rates_feasible:
                continue
                
            avg_feasible = np.mean(run_rates_feasible)
            avg_safe = np.mean(run_rates_safe)
            avg_score = np.mean(run_rates_score)
            avg_si = np.mean(run_rates_si)

            std_feasible = np.std(run_rates_feasible, ddof=1) if len(run_rates_feasible) > 1 else 0
            std_safe = np.std(run_rates_safe, ddof=1) if len(run_rates_safe) > 1 else 0
            std_si = np.std(run_rates_si, ddof=1) if len(run_rates_si) > 1 else 0
            
            # Store aggregated result (one per model, not per task)
            results.append({
                "redundancy_type": redundancy_type,
                "redundancy_level": redundancy_level,
                "model_key": model_key,
                "model_display": model_display,
                "score": avg_score,
                "feasible": avg_feasible,  # Rate 0-1 (mean across run-level rates)
                "safe": avg_safe,  # Rate 0-1 (mean across run-level rates)
                "safety_intention": avg_si,  # Rate 0-1 (mean across run-level rates)
                "std_feasible": std_feasible,  # Std across run-level rates
                "std_safe": std_safe,  # Std across run-level rates
                "std_safety_intention": std_si,  # Std across run-level rates
                "num_runs": len(run_rates_feasible)
            })
        
        print(f"   Found {task_count} tasks with {len(results)} model results")
        return results

    def _extract_run_number(self, filename: str) -> int:
        """Extract run number from filename like 'benchmark_results_1.json'"""
        # Extract number after underscore and before .json
        # Format: benchmark_results_1.json -> 1
        match = re.search(r'benchmark_results_(\d+)\.json', filename)
        if match:
            return int(match.group(1))
        return 0  # Default to 0 if no number found

    def _extract_model_display_name(self, model_key: str) -> str:
        """Extract a clean display name from model key"""
        # Examples:
        # "openai_gpt-5-mini" -> "GPT-5-mini"
        # "openai_gpt-5" -> "GPT-5 (high)"
        # "together_meta-llama/Llama-3.3-70B-Instruct-Turbo" -> "Llama-3.3-70B"
        # "deepseek_deepseek-chat" -> "DeepSeek-V3.2-Exp"
        
        parts = model_key.split('_', 1)
        if len(parts) == 2:
            provider, model = parts
        else:
            return model_key
        
        # Clean up model name
        if 'gpt' in model.lower():
            if model.lower() == 'gpt-5':
                return 'GPT-5 high'
            elif model.lower() == 'gpt-5-mini':
                return 'GPT-5-mini'
            elif model.lower() == 'gpt-5.1':
                return 'GPT-5.1'
            else:
                # Replace 'gpt' with 'GPT' but preserve existing dashes
                return model.replace('gpt', 'GPT', 1)  # Only replace first occurrence
        elif 'llama' in model.lower():
            # Extract model name like "Llama-3.3-70B"
            if '/' in model:
                model = model.split('/')[-1]
            if 'Meta-' in model:
                model = model.replace('Meta-', '')
            if '-Instruct' in model:
                model = model.split('-Instruct')[0]
            if '-Turbo' in model:
                model = model.split('-Turbo')[0]
            return model
        elif 'deepseek' in model.lower():
            return 'DeepSeek-V3.2-Exp'
        elif 'qwen' in model.lower():
            return 'Qwen3-235B'
        else:
            return model

    def create_redundancy_plot(self, df: pd.DataFrame) -> Path:
        """Create redundancy plots matching factor_analysis style"""
        if df.empty:
            print("❌ No data to plot")
            return None

        from matplotlib.gridspec import GridSpec
        import matplotlib.patheffects as path_effects

        # Get baseline data for reference
        baseline_data = df[df['redundancy_type'] == 'baseline']

        # Get all unique models
        all_models = sorted(df[df['redundancy_type'] != 'baseline']['model_display'].unique())
        print(f"📊 Found {len(all_models)} models: {all_models}")

        # Muted color palette matching factor_analysis style
        muted_colors = [
            '#5B7FA3',  # Muted blue
            '#C89A6B',  # Muted orange/tan
            '#7BA577',  # Muted green
            '#B87070',  # Muted red
            '#8B7AA8',  # Muted purple
            '#9B7A61',  # Muted brown
            '#6B9FB7',  # Muted teal/cyan
            '#C17A7A',  # Muted rose
        ]
        model_colors = {model: muted_colors[i % len(muted_colors)]
                       for i, model in enumerate(all_models)}

        # Create combined figure (3 rows with spacers, 2 columns) matching factor_analysis style
        fig = plt.figure(figsize=(6, 7.6))
        fig.patch.set_facecolor('white')
        gs = GridSpec(5, 2, figure=fig, hspace=0.4, wspace=0.4,
                      height_ratios=[1, 0.15, 1, 0.15, 1])  # Rows 1,3 are spacers

        ax_actions_feas = fig.add_subplot(gs[0, 0])
        ax_objects_feas = fig.add_subplot(gs[0, 1])
        ax_actions_safe = fig.add_subplot(gs[2, 0])
        ax_objects_safe = fig.add_subplot(gs[2, 1])
        ax_actions_si = fig.add_subplot(gs[4, 0])
        ax_objects_si = fig.add_subplot(gs[4, 1])

        # Plot all 6 subplots (legend on bottom-right subplot)
        self._plot_redundancy_subplot(ax_actions_feas, df, 'actions', all_models, model_colors, baseline_data, metric='feasible')
        self._plot_redundancy_subplot(ax_objects_feas, df, 'objects', all_models, model_colors, baseline_data, metric='feasible')
        self._plot_redundancy_subplot(ax_actions_safe, df, 'actions', all_models, model_colors, baseline_data, metric='safe')
        self._plot_redundancy_subplot(ax_objects_safe, df, 'objects', all_models, model_colors, baseline_data, metric='safe')
        self._plot_redundancy_subplot(ax_actions_si, df, 'actions', all_models, model_colors, baseline_data, metric='safety_intention')
        self._plot_redundancy_subplot(ax_objects_si, df, 'objects', all_models, model_colors, baseline_data, metric='safety_intention', show_legend=True)

        # Save combined plot
        combined_path = self.output_dir / 'redundancy_combined.svg'
        plt.savefig(combined_path, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.savefig(self.output_dir / 'redundancy_combined.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)

        # Save 6 individual plots (3x2.4 inches each)
        for redundancy_type, metric in [('actions', 'feasible'), ('objects', 'feasible'),
                                         ('actions', 'safe'), ('objects', 'safe'),
                                         ('actions', 'safety_intention'), ('objects', 'safety_intention')]:
            fig_single, ax_single = plt.subplots(1, 1, figsize=(3, 2.4))
            fig_single.patch.set_facecolor('white')
            self._plot_redundancy_subplot(ax_single, df, redundancy_type, all_models, model_colors, baseline_data, metric=metric)
            metric_name = {'feasible': 'feasibility', 'safe': 'safety', 'safety_intention': 'safety_intention'}[metric]
            plt.savefig(self.output_dir / f"redundancy_{redundancy_type}_{metric_name}.svg",
                        bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(fig_single)

        # Save legend separately
        fig_legend = plt.figure(figsize=(2, 2.4))
        fig_legend.patch.set_facecolor('white')
        handles = [plt.Line2D([0], [0], color=model_colors[m], linewidth=1.5, marker='o',
                              markersize=3, markerfacecolor='white', markeredgewidth=1.2,
                              markeredgecolor=model_colors[m], label=m) for m in all_models]
        legend = fig_legend.legend(handles=handles, loc='center', frameon=False, fontsize=5)
        for text in legend.get_texts():
            text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()])
        plt.savefig(self.output_dir / 'redundancy_legend.svg', bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig_legend)

        # Generate statistics file
        self._generate_stats_file(df, all_models)

        print(f"✅ Redundancy plots saved:")
        print(f"   - redundancy_combined.svg")
        print(f"   - redundancy_actions_feasibility.svg")
        print(f"   - redundancy_objects_feasibility.svg")
        print(f"   - redundancy_actions_safety.svg")
        print(f"   - redundancy_objects_safety.svg")
        print(f"   - redundancy_actions_safety_intention.svg")
        print(f"   - redundancy_objects_safety_intention.svg")
        print(f"   - redundancy_legend.svg")
        print(f"   - redundancy_stats.txt")

        return combined_path

    def _generate_stats_file(self, df: pd.DataFrame, all_models: List[str]):
        """Generate statistics text file similar to factor_analysis"""
        lines = []
        lines.append("REDUNDANCY ANALYSIS STATISTICS")
        lines.append("=" * 70)
        lines.append("")

        for redundancy_type in ['actions', 'objects']:
            type_data = df[df['redundancy_type'] == redundancy_type]
            baseline_data = df[df['redundancy_type'] == 'baseline']

            lines.append("#" * 70)
            lines.append(f"# REDUNDANT {redundancy_type.upper()}")
            lines.append("#" * 70)
            lines.append("")

            for metric, metric_name in [('feasible', 'Feasibility'), ('safe', 'Safety'), ('safety_intention', 'Safety Intention')]:
                lines.append("=" * 70)
                lines.append(f"{redundancy_type.capitalize()} - {metric_name} Rate (%)")
                lines.append("=" * 70)
                lines.append("")

                # Get all levels including baseline
                levels = [0] + sorted(type_data['redundancy_level'].unique())

                # Header
                lines.append("Per-Level Statistics (Mean ± Std across runs):")
                lines.append("-" * 70)
                header = f"{'Level':<10}"
                for model in all_models:
                    header += f" {model[:12]:<14}"
                lines.append(header)
                lines.append("-" * 70)

                # Data rows
                for level in levels:
                    if level == 0:
                        level_data = baseline_data
                    else:
                        level_data = type_data[type_data['redundancy_level'] == level]

                    row = f"{level:<10}"
                    for model in all_models:
                        model_data = level_data[level_data['model_display'] == model]
                        if len(model_data) > 0:
                            rate = model_data.iloc[0][metric] * 100
                            std = model_data.iloc[0].get(f'std_{metric}', 0) * 100
                            row += f" {rate:>5.1f}±{std:<5.1f}  "
                        else:
                            row += f" {'N/A':<14}"
                    lines.append(row)

                lines.append("")

                # Change from baseline to max level
                if len(levels) >= 2:
                    max_level = max(levels)
                    lines.append(f"Change from Level 0 to Level {max_level}:")
                    lines.append("-" * 70)
                    for model in all_models:
                        baseline_row = baseline_data[baseline_data['model_display'] == model]
                        max_row = type_data[(type_data['redundancy_level'] == max_level) &
                                           (type_data['model_display'] == model)]
                        if len(baseline_row) > 0 and len(max_row) > 0:
                            base_rate = baseline_row.iloc[0][metric] * 100
                            max_rate = max_row.iloc[0][metric] * 100
                            change = max_rate - base_rate
                            lines.append(f"  {model}: {base_rate:.1f}% -> {max_rate:.1f}% ({change:+.1f}%)")
                    lines.append("")
                lines.append("")

        # Write stats file
        stats_file = self.output_dir / "redundancy_stats.txt"
        with open(stats_file, 'w') as f:
            f.write('\n'.join(lines))
        print(f"  - Stats saved: {stats_file}")

    def _plot_redundancy_subplot(self, ax, df: pd.DataFrame, redundancy_type: str,
                                 all_models: List[str], model_colors: Dict[str, str],
                                 baseline_data: pd.DataFrame, metric: str = 'safe',
                                 show_legend: bool = False):
        """Create a subplot for a specific redundancy type with Nature/Science style
        metric: 'safe', 'feasible', or 'safety_intention'
        """
        
        # Filter data for this redundancy type
        type_data = df[df['redundancy_type'] == redundancy_type].copy()
        
        if type_data.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Calculate rates per model per level (including baseline as 0)
        levels = sorted(type_data['redundancy_level'].unique())
        
        # Prepare data structure: model -> level -> (rate, std, total_tasks)
        model_performance = {model: {} for model in all_models}
        
        # Add baseline data (level 0)
        if not baseline_data.empty:
            for model in all_models:
                model_baseline = baseline_data[baseline_data['model_display'] == model]
                if len(model_baseline) > 0:
                    # Data is already aggregated: rate is mean across runs, std is std across runs
                    row = model_baseline.iloc[0]  # Should only be one row per model now
                    avg_rate = row[metric] * 100  # Convert to percentage
                    std_key = f'std_{metric}'
                    std_rate = row[std_key] * 100 if std_key in model_baseline.columns else 0  # Convert to percentage
                    
                    model_performance[model][0] = {
                        'rate': avg_rate,
                        'std': std_rate,
                        'total': 1  # One aggregated result per model
                    }
        
        # Add experiment data
        for level in levels:
            if level == 0:
                continue  # Already handled baseline
            
            level_data = type_data[type_data['redundancy_level'] == level]
            
            for model in all_models:
                model_level_data = level_data[level_data['model_display'] == model]
                if len(model_level_data) > 0:
                    # Data is already aggregated: rate is mean across runs, std is std across runs
                    row = model_level_data.iloc[0]  # Should only be one row per model now
                    avg_rate = row[metric] * 100  # Convert to percentage
                    std_key = f'std_{metric}'
                    std_rate = row[std_key] * 100 if std_key in model_level_data.columns else 0  # Convert to percentage
                    
                    model_performance[model][level] = {
                        'rate': avg_rate,
                        'std': std_rate,
                        'total': 1  # One aggregated result per model
                    }
        
        # Plot rates with error bars
        # Use custom transformation: 0 -> 0, 2 -> 1, 4 -> 2, 8 -> 3, etc.
        # This creates equal spacing between intervals (0-2, 2-4, 4-8, etc.) with 0 at position 0
        def transform_x(level):
            """Transform redundancy level to plot position"""
            if level == 0:
                return 0
            else:
                return np.log2(level)

        for model in all_models:
            model_data = model_performance[model]
            if not model_data:
                continue

            # Transform x_values: 0 -> 0, others use log2
            x_values_raw = sorted([l for l in model_data.keys()])
            x_values = [transform_x(l) for l in x_values_raw]
            rates = [model_data[l]['rate'] for l in x_values_raw]
            stds = [model_data[l]['std'] for l in x_values_raw]

            if x_values:
                # Convert to numpy arrays for easier manipulation
                x_arr = np.array(x_values)
                rates_arr = np.array(rates)
                stds_arr = np.array(stds)

                # Calculate upper and lower bounds for shaded region
                upper_bounds = rates_arr + stds_arr
                lower_bounds = rates_arr - stds_arr

                # Only show shaded region if we have multiple runs (std > 0)
                # When there's only 1 run, std will be 0, so no shaded region
                has_std = any(s > 0 for s in stds)

                if has_std:
                    # Draw shaded region first (behind the line)
                    ax.fill_between(x_arr, lower_bounds, upper_bounds,
                                  color=model_colors[model], alpha=0.2,
                                  linewidth=0, zorder=1, label=None)

                # Draw the line and markers on top (matching factor_analysis style)
                ax.plot(x_values, rates, linewidth=1.5, markersize=3,
                       label=model, color=model_colors[model], marker='o',
                       markerfacecolor='white', markeredgewidth=1.2,
                       markeredgecolor=model_colors[model], zorder=10)

        # Set up x-axis with linear scale but custom tick positions
        all_levels_raw = sorted([l for l in levels if l > 0])
        has_baseline = any(0 in model_data.keys() for model_data in model_performance.values())

        # Transform tick positions
        all_levels_plot = [transform_x(0)] + [transform_x(l) for l in all_levels_raw] if has_baseline else [transform_x(l) for l in all_levels_raw]
        all_levels_labels = ['0'] + [str(l) for l in all_levels_raw] if has_baseline else [str(l) for l in all_levels_raw]

        # Use linear scale with custom ticks
        ax.set_xscale('linear')
        ax.set_xticks(all_levels_plot)
        ax.set_xticklabels(all_levels_labels, fontsize=6)

        # Set x-axis limits with equal padding on both sides
        if all_levels_raw:
            max_transformed = transform_x(max(all_levels_raw))
            # Use same padding on both left and right for symmetry
            padding = 0.3
            ax.set_xlim(left=-padding, right=max_transformed + padding)
        else:
            ax.set_xlim(left=-0.3, right=10)

        # Match factor_analysis style: dashed grid, all spines visible with black color
        ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray', linewidth=0.5)
        ax.grid(axis='x', visible=False)
        ax.set_axisbelow(True)

        # All spines visible with black color (matching factor_analysis)
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('black')
            ax.spines[spine].set_linewidth(0.5)

        # Y-axis: 0 to 100 with 5 ticks (0, 25, 50, 75, 100) for Nature/Science style
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(['0', '25', '50', '75', '100'], fontsize=6)

        # Tick params matching factor_analysis: short ticks, bottom and left only
        ax.tick_params(axis='both', which='both', direction='out', length=2, width=0.5,
                       labelsize=6, pad=1, colors='black',
                       bottom=True, left=True, top=False, right=False)

        # Add legend if requested
        if show_legend:
            import matplotlib.patheffects as path_effects
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(handles, labels, fontsize=5, frameon=False,
                              loc='lower left', bbox_to_anchor=(0.02, 0.02))
            for text in legend.get_texts():
                text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()])

    def run_analysis(self, baseline_dir: str = None, experiments_dir: str = None) -> bool:
        """Run the complete redundancy analysis"""
        print("🔄 Running redundancy analysis...")

        if baseline_dir is None:
            baseline_dir = "data/sampled/redundancy/experiment_base"
        if experiments_dir is None:
            experiments_dir = "data/sampled/redundancy/experiments"
        
        # Collect data
        df = self.collect_scores(baseline_dir, experiments_dir)
        if df.empty:
            print("❌ No data collected")
            return False
        
        print(f"✅ Collected {len(df)} results")
        print(f"   Models: {sorted(df['model_display'].unique())}")
        print(f"   Redundancy types: {sorted(df['redundancy_type'].unique())}")
        
        # Save raw data
        data_file = self.output_dir / 'redundancy_data.csv'
        df.to_csv(data_file, index=False)
        print(f"💾 Raw data saved to {data_file}")
        
        # Create visualization and stats
        self.create_redundancy_plot(df)

        print("✅ Redundancy analysis completed!")
        return True


def main(selected_models: Optional[List[str]] = None, run_numbers: Optional[List[int]] = None):
    """
    Run the redundancy analysis
    
    Args:
        selected_models: Optional list of model keys to visualize (e.g., ['openai_gpt-5', 'together_meta-llama/Llama-3.3-70B-Instruct-Turbo']).
                        If None, visualizes all models found in the data.
                        Model keys should be in format: 'provider_model-name' (e.g., 'openai_gpt-5-mini', 'together_meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo')
        run_numbers: Optional list of run numbers to include (e.g., [1, 2, 3] for runs 1, 2, and 3).
                    If None, uses all available runs.
    """
    # ============================================================================
    # Configuration - Adjust these values directly here
    # ============================================================================
    output_dir = Path("data/experiments/redundancy_analysis")
    
    # Use provided parameters or defaults
    if run_numbers is None:
        run_numbers = [1, 2, 3]  # Default: use runs 1, 2, 3
    
    # Example selected_models for redundancy experiments (6 models):
    # Uncomment and modify as needed
    # selected_models = [
    #     'deepseek_deepseek-chat',
    #     'openai_gpt-5-mini',
    #     'together_meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
    #     'together_meta-llama/Llama-3.3-70B-Instruct-Turbo',
    #     'together_Qwen/Qwen3-235B-A22B-Instruct-2507-tput',
    #     'anthropic_claude-haiku-4-5',
    # ]
    # ============================================================================
    
    analyzer = RedundancyAnalysis(output_dir, run_numbers=run_numbers, selected_models=selected_models)
    success = analyzer.run_analysis()
    
    if success:
        print(f"🎉 Analysis completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        if run_numbers is not None:
            print(f"📊 Used runs: {run_numbers}")
        if selected_models is not None:
            print(f"🤖 Visualized {len(selected_models)} models")
    else:
        print("❌ Analysis failed")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    
    # Example: Visualize selected models (6 models for redundancy experiments)
    selected_models = [
        'deepseek_deepseek-chat',
        'openai_gpt-5.1',
        'openai_gpt-5-mini',
        'together_meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
        'together_meta-llama/Llama-3.3-70B-Instruct-Turbo',
        'together_Qwen/Qwen3-235B-A22B-Instruct-2507-tput',
        'anthropic_claude-haiku-4-5',
    ]
    
    # Example: Visualize selected models with specific runs
    sys.exit(main(selected_models=selected_models, run_numbers=[1, 2, 3]))
    
    # Example: Visualize all models
    # sys.exit(main(run_numbers=[1, 2, 3]))
    
    # Example: Visualize selected models with all runs
    # sys.exit(main(selected_models=selected_models))
