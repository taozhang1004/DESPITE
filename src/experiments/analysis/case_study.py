#!/usr/bin/env python3
"""
Case Study: Detailed Analysis of Specific Model-Task Cases

Allows filtering and examining specific cases where models got certain scores.
Shows the generated plan and GT safe feasible plan for each case.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import glob
import re
import sys
import random
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def collect_cases(
    parent_folders: List[str],
    model_keys: Optional[List[str]] = None,
    score_filter: Optional[int] = None,
    run_number: int = 1
) -> List[Dict[str, Any]]:
    """
    Collect cases matching the specified criteria.
    
    Args:
        parent_folders: List of parent folders to search (e.g., ["data/sampled/val-100"])
        model_keys: List of model keys to filter by (None = all models)
        score_filter: Score to filter by (None = all scores)
        run_number: Which benchmark_results_{run}.json file to read
    
    Returns:
        List of case dictionaries with task_id, model_key, score, generated_plan, etc.
    """
    cases = []
    
    for parent_folder in parent_folders:
        # Find all benchmark_results_{run}.json files
        pattern = f"{parent_folder}/*/benchmark_results_{run_number}.json"
        result_files = glob.glob(pattern)
        
        for result_file in result_files:
            task_dir = Path(result_file).parent
            task_id = task_dir.name
            
            try:
                with open(result_file) as f:
                    benchmark_data = json.load(f)
                
                if 'models' not in benchmark_data:
                    continue
                
                # Load GT safe plan and unsafe plan from result.json
                gt_safe_plan = None
                gt_unsafe_plan = None
                result_json_path = task_dir / "result.json"
                if result_json_path.exists():
                    try:
                        with open(result_json_path) as rf:
                            result_data = json.load(rf)
                            if 'generated_plans' in result_data:
                                gt_safe_plan = result_data['generated_plans'].get('safe_plan', None)
                                gt_unsafe_plan = result_data['generated_plans'].get('unsafe_plan', None)
                    except Exception as e:
                        print(f"Warning: Could not read result.json for {task_id}: {e}")
                
                # Process each model
                for model_key, model_data in benchmark_data['models'].items():
                    # Filter by model if specified
                    if model_keys is not None and model_key not in model_keys:
                        continue
                    
                    task_types = model_data.get('task_types', {})
                    
                    # Check comprehensive_planning results
                    if 'comprehensive_planning' in task_types:
                        cp_result = task_types['comprehensive_planning']
                        validation_result = cp_result.get('validation_result', {})
                        score = validation_result.get('score')
                        
                        # Filter by score if specified
                        if score_filter is not None and score != score_filter:
                            continue
                        
                        # Extract generated plan
                        cp_data = cp_result.get('comprehensive_planning_result', {})
                        generated_plan = cp_data.get('response', '')
                        
                        # Format plan nicely (remove markdown code blocks if present)
                        if generated_plan:
                            generated_plan = generated_plan.strip()
                            if generated_plan.startswith('```'):
                                lines = generated_plan.split('\n')
                                generated_plan = '\n'.join(lines[1:-1]) if len(lines) > 2 else generated_plan
                        
                        case = {
                            'task_id': task_id,
                            'task_dir': str(task_dir),
                            'model_key': model_key,
                            'score': score,
                            'feasible': validation_result.get('feasible', None),
                            'safe': validation_result.get('safe', None),
                            'generated_plan': generated_plan,
                            'gt_safe_plan': gt_safe_plan,
                            'gt_unsafe_plan': gt_unsafe_plan,
                            'validation_output': validation_result.get('output', ''),
                        }
                        cases.append(case)
            
            except Exception as e:
                print(f"Warning: Error processing {result_file}: {e}")
                continue
    
    return cases


def format_plan(plan: Optional[str]) -> str:
    """Format a plan string for readable display - actions on separate lines"""
    if plan is None:
        return "N/A"
    
    plan = plan.strip()
    if not plan:
        return "Empty plan"
    
    # Remove markdown code blocks if present
    if plan.startswith('```'):
        lines = plan.split('\n')
        plan = '\n'.join(lines[1:-1]) if len(lines) > 2 else plan
    
    # Extract actions (things in parentheses)
    actions = re.findall(r'\([^)]+\)', plan)
    
    if actions:
        # Put each action on a separate line
        return '\n'.join(actions)
    else:
        # Fallback: return as-is, preserving line breaks
        return plan


def print_case(case: Dict[str, Any], case_number: int, total_cases: int):
    """Print a single case in a readable format"""
    print("\n" + "=" * 100)
    print(f"CASE {case_number}/{total_cases}")
    print("=" * 100)
    print(f"Task ID: {case['task_dir']}")
    print(f"Model: {case['model_key']}")
    print(f"GENERATED PLAN: Score={case['score']}, Feasible={case['feasible']}, Safe={case['safe']}")
    print("-" * 100)
    print(format_plan(case['generated_plan']))
    print("-" * 100)
    # Use unsafe plan for score 0, safe plan for other scores
    if case['score'] == 0:
        print("GT UNSAFE FEASIBLE PLAN:")
        print("-" * 100)
        print(format_plan(case['gt_unsafe_plan']))
    else:
        print("GT SAFE FEASIBLE PLAN:")
        print("-" * 100)
        print(format_plan(case['gt_safe_plan']))


def save_cases_to_file(cases: List[Dict[str, Any]], output_path: Path):
    """Save cases to a text file for easy reading"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("CASE STUDY RESULTS\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Total cases found: {len(cases)}\n\n")
        
        for i, case in enumerate(cases, 1):
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"CASE {i}/{len(cases)}\n")
            f.write("=" * 100 + "\n")
            f.write(f"Task ID: {case['task_dir']}\n")
            f.write(f"Model: {case['model_key']}\n")
            f.write(f"GENERATED PLAN: Score={case['score']}, Feasible={case['feasible']}, Safe={case['safe']}\n")
            f.write("-" * 100 + "\n")
            f.write(format_plan(case['generated_plan']) + "\n")
            f.write("-" * 100 + "\n")
            # Use unsafe plan for score 0, safe plan for other scores
            if case['score'] == 0:
                f.write("GT UNSAFE FEASIBLE PLAN:\n")
                f.write("-" * 100 + "\n")
                f.write(format_plan(case['gt_unsafe_plan']) + "\n")
            else:
                f.write("GT SAFE FEASIBLE PLAN:\n")
                f.write("-" * 100 + "\n")
                f.write(format_plan(case['gt_safe_plan']) + "\n")
    
    print(f"✅ Cases saved to {output_path}")


def get_short_model_name(model_key: str) -> str:
    """Get short display name for a model (same as in general_analysis.py)"""
    parts = model_key.split('_', 1)
    if len(parts) < 2:
        return model_key
    
    provider, model_name = parts
    
    # Handle specific model name mappings
    if 'gpt-5' in model_name:
        if 'mini' in model_name:
            return 'GPT-5-mini'
        return 'GPT-5'
    elif 'deepseek-chat' in model_name:
        return 'DeepSeek-V3.2'
    elif 'deepseek-reasoner' in model_name:
        return 'DeepSeek-V3.2-Thinking'
    elif 'Qwen3-Next-80B-A3B-Instruct' in model_name:
        return 'Qwen3-Next-80B-Instruct'
    elif 'Qwen3-Next-80B-A3B-Thinking' in model_name:
        return 'Qwen3-Next-80B-Thinking'
    elif 'Qwen3-235B-A22B-Thinking' in model_name:
        return 'Qwen3-235B-Thinking'
    elif 'Qwen3-Coder-480B' in model_name:
        return 'Qwen3-Coder-480B-Instruct'
    elif 'Qwen3-235B-A22B-Instruct-2507-tput' in model_name:
        return 'Qwen3-235B-Instruct'
    elif 'QwQ-32B' in model_name:
        return 'QwQ-32B'
    elif 'Qwen2.5-72B' in model_name:
        return 'Qwen2.5-72B-Instruct'
    elif 'Qwen2.5-7B' in model_name:
        return 'Qwen2.5-7B-Instruct'
    elif 'Qwen2.5-Coder-32B' in model_name:
        return 'Qwen2.5-Coder-32B-Instruct'
    elif 'Llama-4-Maverick' in model_name:
        return 'Llama-4-Maverick-Instruct'
    elif 'Llama-3.3-70B' in model_name:
        return 'Llama-3.3-70B-Instruct'
    elif 'Llama-4-Scout' in model_name:
        return 'Llama-4-Scout-Instruct'
    elif 'Meta-Llama-3.1-405B' in model_name:
        return 'Llama-3.1-405B-Instruct'
    elif 'Meta-Llama-3.1-8B' in model_name:
        return 'Llama-3.1-8B-Instruct'
    elif 'Meta-Llama-3.1-70B' in model_name:
        return 'Llama-3.1-70B-Instruct'
    elif 'Llama-3.2-3B' in model_name:
        return 'Llama-3.2-3B-Instruct'
    elif 'Meta-Llama-3-70B' in model_name:
        return 'Llama-3-70B-Instruct'
    elif 'Meta-Llama-3-8B-Instruct-Lite' in model_name:
        return 'Llama-3-8B-Instruct'
    elif 'gpt-oss-20b' in model_name:
        return 'GPT-OSS-20B'
    elif 'gpt-oss-120b' in model_name:
        return 'GPT-OSS-120B'
    
    # Fallback: return model name
    return model_name


def main():
    """Main function - configure your case study here"""
    
    # ============================================================================
    # CONFIGURATION - Adjust these parameters as needed
    # ============================================================================
    
    # Which task folders to search
    parent_folders = ["data/full/hard"]
    
    # Which models to study (None = all models)
    # Examples:
    #   model_keys = ["openai_gpt-5"]  # Single model
    #   model_keys = ["openai_gpt-5", "openai_gpt-5-mini"]  # Multiple models
    #   model_keys = None  # All models
    model_keys = None  # All models - we want random cases from random models
    
    # Which scores to process (0 = infeasible, 1 = feasible but not safe, 2 = safe)
    scores_to_process = [0, 1, 2]
    
    # Number of random cases to sample per score (None = use all cases, no sampling)
    num_samples_per_score = None
    
    # Which benchmark_results_{run}.json file to read
    run_number = 1
    
    # Output options
    print_to_console = False  # Print cases to console
    save_to_file = True  # Save cases to a text file
    
    # Output file path (if save_to_file is True)
    output_dir = Path("data/experiments/case_study")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # END CONFIGURATION
    # ============================================================================
    
    # Process each score separately
    for score_filter in scores_to_process:
        print("\n" + "=" * 100)
        print(f"Processing Score {score_filter}")
        print("=" * 100)
        
        print("🔄 Collecting cases...")
        print(f"   Task folders: {parent_folders}")
        print(f"   Models: All models")
        print(f"   Score filter: {score_filter}")
        print(f"   Run number: {run_number}")
        
        cases = collect_cases(
            parent_folders=parent_folders,
            model_keys=model_keys,
            score_filter=score_filter,
            run_number=run_number
        )
        
        print(f"✅ Found {len(cases)} cases matching criteria")
        
        if len(cases) == 0:
            print(f"⚠️  No cases found for score {score_filter}. Skipping...")
            continue
        
        # Randomly sample cases (if num_samples_per_score is None, use all cases)
        if num_samples_per_score is None:
            print(f"📋 Using all {len(cases)} cases (no sampling)")
            sampled_cases = cases
        elif len(cases) > num_samples_per_score:
            print(f"🎲 Randomly sampling {num_samples_per_score} cases from {len(cases)} total cases...")
            sampled_cases = random.sample(cases, num_samples_per_score)
        else:
            print(f"⚠️  Only {len(cases)} cases found (less than {num_samples_per_score}). Using all cases.")
            sampled_cases = cases
        
        # Print summary statistics
        print("\n📊 Summary Statistics:")
        print("=" * 80)
        
        # Count by model
        model_counts = {}
        for case in sampled_cases:
            model_key = case['model_key']
            short_name = get_short_model_name(model_key)
            model_counts[short_name] = model_counts.get(short_name, 0) + 1
        
        print("\nCases by model:")
        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: {count} cases")
        
        # Generate output filename
        score_str = f"score_{score_filter}"
        output_filename = f"cases_all_models_{score_str}_run{run_number}.txt"
        output_path = output_dir / output_filename
        
        # Print or save cases
        if print_to_console:
            print("\n" + "=" * 100)
            print("DETAILED CASES")
            print("=" * 100)
            for i, case in enumerate(sampled_cases, 1):
                print_case(case, i, len(sampled_cases))
        
        if save_to_file:
            save_cases_to_file(sampled_cases, output_path)
        
        print(f"\n📊 Score {score_filter} case study complete!")
        print(f"   Total cases analyzed: {len(sampled_cases)}")
        if save_to_file:
            print(f"   Results saved to: {output_path}")
    
    print("\n" + "=" * 100)
    print("✅ All case studies complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()

