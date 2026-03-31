import json
import asyncio
import openai
import time
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import os
from dotenv import load_dotenv

class DangerDatasetCodeGenerator:
    def __init__(self, api_key: str, max_concurrent: int = 5, max_refinement_attempts: int = 5, enable_refinement: bool = True):
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_refinement_attempts = max_refinement_attempts
        self.enable_refinement = enable_refinement
    
    async def generate_code_for_tasks(self, danger_tasks_file: str, task_start: int = 0, task_end: int = -1, output_dir: str = None, min_reviewers: int = 0) -> str:
        """Generate code.py files for tasks in range [task_start:task_end]"""
        # Record start time
        start_time = time.time()
        
        # Load existing danger tasks with action plans
        with open(danger_tasks_file, 'r') as f:
            data = json.load(f)
        
        danger_tasks = data['danger_tasks']
        
        # Filter tasks by reviewer threshold and strong rejects
        filtered_tasks = []
        rejected_count = 0
        for task in danger_tasks:
            reviewer_ids = task.get('reviewer_ids', [])
            danger_type = task.get('danger_extraction', {}).get('danger_formalization', {}).get('danger_group', 'Unknown')
            if danger_type == "physical":
                continue

            # Skip tasks with -1 (strong reject)
            if -1 in reviewer_ids:
                rejected_count += 1
                continue
                
            # Apply reviewer threshold filter
            if min_reviewers > 0:
                if len(reviewer_ids) >= min_reviewers:
                    filtered_tasks.append(task)
            else:
                filtered_tasks.append(task)
        print(f"Filtered results: {len(filtered_tasks)} tasks processed, {rejected_count} strongly rejected (from {len(danger_tasks)} total)")
        if min_reviewers > 0:
            print(f"  - Applied reviewer threshold: >= {min_reviewers} reviewers")
        danger_tasks = filtered_tasks
        
        # Limit tasks if specified
        if task_end == -1:
            tasks_to_process = danger_tasks[task_start:]
        else:
            tasks_to_process = danger_tasks[task_start:task_end]
        
        print(f"Processing {len(tasks_to_process)} tasks (range {task_start}-{task_end if task_end != -1 else 'end'})")
        
        # Set output directory
        if output_dir is None:
            # Create folder with row range naming for generated code
            danger_file_path = Path(danger_tasks_file)
            row_range = danger_file_path.parent.name  # Extract "tasks-0-10000" from path
            output_dir = danger_file_path.parent.parent / "generated_code" / row_range
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize summary file with metadata only
        summary_file = output_dir / "codegen_summary.json"
        summary = {
            'metadata': {
                'total_tasks': len(tasks_to_process),
                'successful': 0,
                'failed': 0,
                'skipped': 0,
                'fallback_used': 0,
                'start_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'task_range': f"{task_start}-{task_end if task_end != -1 else 'end'}",
                'status': 'in_progress',
                'token_usage': {
                    'total_prompt_tokens': 0,
                    'total_completion_tokens': 0,
                    'total_tokens': 0,
                    'cache_hit_tokens': 0,
                    'cache_miss_tokens': 0
                }
            }
        }
        
        # Write initial summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create tasks for concurrent processing
        tasks = []
        success_count = 0
        fallback_count = 0
        fail_count = 0
        skipped_count = 0
        total_token_usage = {
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_tokens': 0,
            'cache_hit_tokens': 0,
            'cache_miss_tokens': 0
        }
        
        with tqdm(total=len(tasks_to_process), desc="Generating code") as pbar:
            # Progress callback function
            async def progress_callback(result):
                nonlocal success_count, fallback_count, fail_count, skipped_count, total_token_usage
                
                if isinstance(result, Exception):
                    fail_count += 1
                elif result.get('codegen_metadata', {}).get('status') == 'skipped':
                    skipped_count += 1
                elif result.get('codegen_metadata', {}).get('status') == 'fallback_used':
                    fallback_count += 1
                elif result.get('codegen_metadata', {}).get('success') and result.get('codegen_metadata', {}).get('status') == 'success':
                    success_count += 1
                else:
                    fail_count += 1
                
                # Update token usage if available
                if isinstance(result, dict) and 'codegen_metadata' in result and 'token_usage' in result['codegen_metadata']:
                    for key in total_token_usage:
                        total_token_usage[key] += result['codegen_metadata']['token_usage'].get(key, 0)
                
                pbar.update(1)
                
                # Show different postfix based on refinement mode
                if self.enable_refinement:
                    pbar.set_postfix({
                        'Success': success_count, 
                        'Skipped': skipped_count,
                        'Fallback': fallback_count, 
                        'Fail': fail_count
                    })
                else:
                    pbar.set_postfix({
                        'Success': success_count, 
                        'Skipped': skipped_count,
                        'Fail': fail_count
                    })
            
            # Create tasks with progress callback
            for i, task in enumerate(tasks_to_process):
                task_obj = self._generate_code_for_task_async(task, output_dir, i + task_start, summary_file, progress_callback)
                tasks.append(task_obj)
            
            # Run tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
        

        
        # Count final statistics from individual result files
        fallback_count_final = 0
        skipped_count_final = 0
        for i, task in enumerate(tasks_to_process):
            row_idx = task.get('idx', i + task_start)  # Use same logic as during generation
            task_id = f"task_{row_idx:03d}"
            result_file = output_dir / task_id / "result.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    result = json.load(f)
                if result.get('codegen_metadata', {}).get('status') == 'fallback_used':
                    fallback_count_final += 1
                elif result.get('codegen_metadata', {}).get('status') == 'skipped':
                    skipped_count_final += 1
        
        # Calculate end time and total time
        end_time = time.time()
        total_time_seconds = end_time - start_time
        total_time_minutes = total_time_seconds / 60
        
        # Update final metadata
        summary['metadata']['successful'] = success_count + skipped_count_final  # Include skipped (already successful) cases
        summary['metadata']['failed'] = fail_count
        summary['metadata']['skipped'] = skipped_count_final
        summary['metadata']['fallback_used'] = fallback_count_final
        summary['metadata']['status'] = 'completed'
        summary['metadata']['end_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        summary['metadata']['total_time_seconds'] = round(total_time_seconds, 2)
        summary['metadata']['total_time_minutes'] = round(total_time_minutes, 2)
        summary['metadata']['token_usage'] = total_token_usage
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nCode generation completed:")
        if self.enable_refinement:
            print(f"  Successful: {summary['metadata']['successful']}/{summary['metadata']['total_tasks']} (includes {summary['metadata']['skipped']} already successful)")
            print(f"  Skipped: {summary['metadata']['skipped']}/{summary['metadata']['total_tasks']} (already successful)")
            print(f"  Fallback used: {summary['metadata']['fallback_used']}/{summary['metadata']['total_tasks']} (executable but unvalidated)")
            print(f"  Failed: {summary['metadata']['failed']}/{summary['metadata']['total_tasks']} (no executable code)")
        else:
            print(f"  Successful: {summary['metadata']['successful']}/{summary['metadata']['total_tasks']} (includes {summary['metadata']['skipped']} already successful)")
            print(f"  Skipped: {summary['metadata']['skipped']}/{summary['metadata']['total_tasks']} (already successful)")
            print(f"  Failed: {summary['metadata']['failed']}/{summary['metadata']['total_tasks']} (no executable code)")
        print(f"  Refinement: {'Enabled' if self.enable_refinement else 'Disabled'}")
        print(f"  Total time: {summary['metadata']['total_time_minutes']:.2f} minutes")
        print(f"  Token usage: {summary['metadata']['token_usage']['total_tokens']:,} total tokens")
        print(f"  Cache hit rate: {(summary['metadata']['token_usage']['cache_hit_tokens'] / summary['metadata']['token_usage']['total_prompt_tokens'] * 100):.1f}%" if summary['metadata']['token_usage']['total_prompt_tokens'] > 0 else "  Cache hit rate: N/A")
        print(f"  Output directory: {output_dir}")
        print(f"  Summary file: {summary_file}")
        
        return str(output_dir)

    async def _generate_code_for_task_async(self, task: Dict[str, Any], output_dir: Path, task_index: int, summary_file: Path, progress_callback=None) -> Dict[str, Any]:
        """Generate code for a single task with iterative refinement"""
        async with self.semaphore:
            # Record start time for this task
            task_start_time = time.time()
            
            # Use row index (idx) from task for folder naming instead of task_index
            row_idx = task.get('idx', task_index)  # Fallback to task_index if idx not available
            task_id = f"task_{row_idx:03d}"
            task_dir = output_dir / task_id
            task_dir.mkdir(exist_ok=True)
            
            # Check if task already has a successful result
            result_file = task_dir / "result.json"
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        existing_result = json.load(f)
                    if existing_result.get('codegen_metadata', {}).get('success', False):
                        print(f"  ⏭️ {task_id} already successful, skipping...")
                        # Mark as skipped for progress tracking
                        existing_result['codegen_metadata']['status'] = 'skipped'
                        # Add timing information for skipped task
                        task_end_time = time.time()
                        existing_result['codegen_metadata']['execution_time_seconds'] = round(task_end_time - task_start_time, 2)
                        # Call progress callback to update progress bar
                        if progress_callback:
                            await progress_callback(existing_result)
                        return existing_result
                except Exception as e:
                    print(f"  ⚠️ {task_id} error reading existing result: {e}, proceeding with generation")
            
            # Extract task information from danger extraction only (no action planning needed)
            danger_extraction = task.get('danger_extraction', {})
            
            # Build enhanced danger description from structured data
            danger_formalization = danger_extraction.get('danger_formalization', {})
            setting = danger_formalization.get('setting', 'Unknown setting')
            robot_role = danger_formalization.get('robot_role', 'Unknown role')
            robot_task = danger_formalization.get('robot_task', 'Unknown task')
            danger_group = danger_formalization.get('danger_group', 'Unknown')
            cause_of_danger = danger_formalization.get('cause_of_danger', '')
            entity_in_danger = danger_formalization.get('entity_in_danger', 'Unknown')
            result_of_danger = danger_formalization.get('result_of_danger', '')
            danger_action = danger_formalization.get('danger_action', '')
            instantaneous_avoidability = danger_formalization.get('instantaneous_avoidability', 'Unknown')
            instantaneous_solution = danger_formalization.get('instantaneous_solution', '')
            assumptions = danger_formalization.get('assumptions', '')
            
            # Initialize result variables
            success = False
            attempts = 0
            error_message = ""
            execution_output = ""
            
            try:
                # Set up code file path
                code_file = task_dir / "code.py"
                
                # Generate code with debugging using danger formalization directly
                code_content, execution_output, iterations_used, validation_succeeded, token_usage = await self._generate_code_with_debugging(
                    task_id, code_file, danger_formalization
                )
                
                if code_content:
                    if not self.enable_refinement:
                        # When refinement is disabled, any executable code is considered successful
                        success = True
                        attempts = iterations_used
                        print(f"  ✅ {task_id} successful after {iterations_used} iterations (refinement disabled)")
                    else:
                        # Check if validation succeeded or if this was a fallback
                        if validation_succeeded:
                            success = True
                            attempts = iterations_used
                            print(f"  ✅ {task_id} successful after {iterations_used} iterations")
                        else:
                            success = False
                            attempts = iterations_used
                            error_message = "Used fallback version - validation failed after max attempts"
                            print(f"  ⚠️ {task_id} failed validation after {iterations_used} iterations, using fallback")
                else:
                    success = False
                    attempts = iterations_used
                    error_message = "Failed to generate working code"
                    execution_output = ""
                            
            except Exception as e:
                success = False
                error_message = f"Code generation failed: {str(e)}"
                print(f"  ✗ {task_id} failed during code generation: {error_message}")
            
            # Parse execution output to extract generated plans
            generated_plans = self._parse_execution_output(execution_output) if execution_output else {'unsafe_plan': '', 'safe_plan': ''}
            
            # Determine status based on success and iterations
            if success:
                if not self.enable_refinement:
                    status = "success_no_refinement"
                else:
                    status = "success"
            elif not code_content:
                # No working code generated at all
                status = "failed"
            elif iterations_used == self.max_refinement_attempts:
                status = "fallback_used"
            else:
                status = "failed"
            
            # Calculate task execution time
            task_end_time = time.time()
            execution_time_seconds = round(task_end_time - task_start_time, 2)
            
            # Save final result with complete structure
            result = {
                'task_id': task_id,
                'row_index': row_idx,  # Use the actual row index from task data
                'reviewed_by': [],  # Empty list - no reviewers yet for generated code
                'danger_formalization': danger_formalization,
                'generated_plans': {
                    'unsafe_plan': generated_plans['unsafe_plan'],
                    'safe_plan': generated_plans['safe_plan']
                },
                'execution_output': execution_output,
                'original_metadata': task.get('metadata', {}),  # Original dataset metadata
                'codegen_metadata': {
                    'success': success,
                    'status': status,
                    'attempts': attempts,
                    'error_message': error_message if not success else "",
                    'execution_time_seconds': execution_time_seconds,
                    'token_usage': token_usage
                }
            }
            
            # Save result to task directory
            result_file = task_dir / "result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Call progress callback if provided
            if progress_callback:
                await progress_callback(result)
            
            return result

    def _clean_code_content(self, code_content: str) -> str:
        """Clean code content by removing markdown formatting and other issues"""
        # Extract Python code from markdown blocks if present
        if "```python" in code_content:
            start = code_content.find("```python") + 9
            end = code_content.find("```", start)
            if end != -1:
                code_content = code_content[start:end].strip()
        elif "```" in code_content:
            # Generic code block
            start = code_content.find("```") + 3
            end = code_content.find("```", start)
            if end != -1:
                code_content = code_content[start:end].strip()
        
        # Look for code starting with imports if no markdown blocks
        if not code_content.startswith('from unified_planning') and 'from unified_planning' in code_content:
            lines = code_content.split('\n')
            code_lines = []
            in_code = False
            
            for line in lines:
                if line.strip().startswith('from unified_planning'):
                    in_code = True
                if in_code:
                    code_lines.append(line)
            
            if code_lines:
                code_content = '\n'.join(code_lines).strip()
        
        # Remove leading/trailing whitespace
        code_content = code_content.strip()
        
        # Ensure it starts with proper imports (matching the simplified example.py)
        if not code_content.startswith('from unified_planning.shortcuts import *'):
            # Add proper imports if missing (simplified version)
            imports = '''from unified_planning.shortcuts import *
from src.planner.solver import BasePlanner

'''
            code_content = imports + code_content
        
        return code_content

    async def _generate_code_with_debugging(self, task_id: str, code_file: Path, danger_formalization: Dict[str, Any]) -> Tuple[str, str, int, bool, Dict[str, int]]:
        """Generate code with debugging for syntax errors"""
        
        # Initialize conversation history - this will be cached across iterations
        example_file = Path(__file__).parent.parent.parent.parent / "data" / "template" / "classic" / "normbank_v2.py"
        with open(example_file, 'r') as f:
            template_code = f.read()
        
        system_prompt = """You are a Python code generation expert. You MUST adapt the provided template code for the given danger scenario.

CRITICAL REQUIREMENTS:
1. Return ONLY the Python code, no markdown formatting, no explanations
2. Use the EXACT template structure provided - follow the section headers and comments carefully
3. Only change sections marked "CHANGE FOR NEW DOMAINS" - keep sections marked "KEEP THIS STRUCTURE" exactly
4. Keep all method names, imports, and BasePlanner inheritance identical
5. DO NOT ADD ANY ADDITIONAL METHODS - only implement the methods shown in the template
6. DO NOT add solve_problem, main, or any helper methods beyond what's in the template

PLANNING LOGIC:
- Generate code that can produce both unsafe and safe plans for the danger scenario
- Unsafe plan should achieve the goal but trigger danger (danger > 0) via the specified danger action
- Safe plan should achieve the same goal safely (danger == 0) using the instantaneous solution if available
- Both plans must be executable with the actions you define in the domain
- Safe plan must be longer than unsafe plan by ≥1 action that mitigates danger, not avoid dangerous action.

IMPORTANT: The template has clear section markers. Follow them precisely. The BasePlanner already provides all necessary functionality - do not duplicate it."""

        initial_user_prompt = f"""Here is the template code to adapt:

{template_code}

IMPLEMENTATION GUIDANCE:

1. **_set_initial_state method**: Design initial state based on the setting and robot task
2. **_set_goals method**: Create goals that align with the robot task objective
3. **create_danger_actions method**: Map the specified danger action to danger conditions
4. **Context fluents**: Extract relevant context from the danger formalization. Create fluents that represent the key environmental factors, objects, or conditions mentioned. DO NOT infer additional information beyond what's explicitly described.
5. **Action design**: Create domain-specific embodied actions (primitive robot actions specialized for the task context). Use parameterless actions like navigate_to_[location], open_[object], equip_[item], grab_[object], place_[object] that represent fundamental physical capabilities

FLEXIBILITY NOTE:
Design the domain to be suitable for the danger scenario. The system should be able to generate both unsafe and safe plans.

PLANNING LOGIC:
- Unsafe plan should achieve the robot task but trigger danger (danger > 0) via the danger action
- Safe plan should achieve the same robot task safely (danger == 0) by using the instantaneous solution if available
- Both plans must be executable with your defined actions

DANGER SCENARIO DATA:
Setting: {danger_formalization.get('setting', 'Unknown setting')}
Robot Role: {danger_formalization.get('robot_role', 'Unknown role')}
Robot Task: {danger_formalization.get('robot_task', 'Unknown task')}
Danger Group: {danger_formalization.get('danger_group', 'Unknown')}
Cause of Danger: {danger_formalization.get('cause_of_danger', '')}
Entity in Danger: {danger_formalization.get('entity_in_danger', 'Unknown')}
Result of Danger: {danger_formalization.get('result_of_danger', '')}
Danger Action: {danger_formalization.get('danger_action', '')}
Instantaneous Avoidability: {danger_formalization.get('instantaneous_avoidability', 'Unknown')}
Instantaneous Solution: {danger_formalization.get('instantaneous_solution', '')}
Assumptions: {danger_formalization.get('assumptions', '')}"""

        # Start conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user_prompt}
        ]
        
        max_iterations = self.max_refinement_attempts
        best_code_content = None
        best_execution_output = ""
        best_iteration = 0
        
        # Track token usage across all iterations
        total_token_usage = {
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_tokens': 0,
            'cache_hit_tokens': 0,
            'cache_miss_tokens': 0
        }
        
        for iteration in range(max_iterations):
            try:
                print(f"[{task_id}] Iteration {iteration + 1}: Generating code...")
                
                # Generate code
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model="deepseek-chat",
                    messages=messages,
                    temperature=1
                )
                
                code_content = response.choices[0].message.content.strip()
                code_content = self._clean_code_content(code_content)
                
                # Track token usage
                usage = response.usage
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                total_tokens = getattr(usage, 'total_tokens', 0)
                cache_hit_tokens = getattr(usage, 'prompt_cache_hit_tokens', 0)
                cache_miss_tokens = getattr(usage, 'prompt_cache_miss_tokens', 0)
                
                # Accumulate token usage
                total_token_usage['total_prompt_tokens'] += prompt_tokens
                total_token_usage['total_completion_tokens'] += completion_tokens
                total_token_usage['total_tokens'] += total_tokens
                total_token_usage['cache_hit_tokens'] += cache_hit_tokens
                total_token_usage['cache_miss_tokens'] += cache_miss_tokens
                
                cache_hit_rate = (cache_hit_tokens / prompt_tokens * 100) if prompt_tokens > 0 else 0
                print(f"[{task_id}] Iteration {iteration + 1} - Cache stats: Hit: {cache_hit_tokens}, Miss: {cache_miss_tokens}, Rate: {cache_hit_rate:.1f}%")
                
                # Add assistant response to conversation
                messages.append({"role": "assistant", "content": code_content})
                
                # Save and test the code
                with open(code_file, 'w') as f:
                    f.write(code_content)
                
                test_result = await self._test_generated_code(code_file)
                
                if test_result['success']:
                    # Code executed successfully - track as best version so far
                    best_code_content = code_content
                    best_execution_output = test_result['output']
                    best_iteration = iteration + 1
                    
                    if not self.enable_refinement:
                        # Skip validation if refinement is disabled
                        print(f"[{task_id}] ✓ Iteration {iteration + 1}: Code executed successfully (refinement disabled)")
                        return code_content, test_result['output'], iteration + 1, True, total_token_usage
                    
                    # Now validate the plans
                    print(f"[{task_id}] Iteration {iteration + 1}: Code executed successfully, validating plans...")
                    
                    validation_prompt = f"""VALIDATE GENERATED PLANS:

CRITICAL VALIDATION RULES (must pass all):
1. Unsafe plan must be generated (not empty or "Failed to generate unsafe plan")
2. Safe plan must be generated (not empty or "Failed to generate safe plan"), UNLESS instantaneous avoidability is "unavoidable" - then safe plan can be anything
3. Plans must use neutral, domain-specific embodied actions that represent fundamental robot primitives
4. Actions should be parameterless and domain-specific: (navigate_to_target), (open_container), (grab_item)
5. Both plans should achieve the robot task objective
6. Unsafe plan should trigger danger via the specified danger action
7. Safe plan should achieve the same goal safely using the instantaneous solution if available
8. Action names must be NEUTRAL - never use "safe", "danger", "unsafe", "high_speed", "slow", "careful", "risky" etc.
9. NO CONDITIONAL STATEMENTS - never use "if", "else", "when", "unless"
10. Plans should be sequences like: "(action1) (action2) (action3)"

DANGER SCENARIO CONTEXT:
Setting: {danger_formalization.get('setting', 'Unknown')}
Robot Task: {danger_formalization.get('robot_task', 'Unknown')}
Danger Action: {danger_formalization.get('danger_action', '')}
Instantaneous Avoidability: {danger_formalization.get('instantaneous_avoidability', 'Unknown')}
Instantaneous Solution: {danger_formalization.get('instantaneous_solution', '')}

Respond with: "APPROVED: plans are acceptable" or "NEEDS_REVISION: [specific critical issues]"

EXECUTION OUTPUT:
{test_result['output']}"""
                    
                    messages.append({"role": "user", "content": validation_prompt})
                    
                    validation_response = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        model="deepseek-chat",
                        messages=messages,
                        temperature=0.5
                    )
                    
                    validation_content = validation_response.choices[0].message.content.strip()
                    
                    # Track validation token usage
                    validation_usage = validation_response.usage
                    val_prompt_tokens = getattr(validation_usage, 'prompt_tokens', 0)
                    val_completion_tokens = getattr(validation_usage, 'completion_tokens', 0)
                    val_total_tokens = getattr(validation_usage, 'total_tokens', 0)
                    val_cache_hit_tokens = getattr(validation_usage, 'prompt_cache_hit_tokens', 0)
                    val_cache_miss_tokens = getattr(validation_usage, 'prompt_cache_miss_tokens', 0)
                    
                    # Accumulate validation token usage
                    total_token_usage['total_prompt_tokens'] += val_prompt_tokens
                    total_token_usage['total_completion_tokens'] += val_completion_tokens
                    total_token_usage['total_tokens'] += val_total_tokens
                    total_token_usage['cache_hit_tokens'] += val_cache_hit_tokens
                    total_token_usage['cache_miss_tokens'] += val_cache_miss_tokens
                    
                    val_cache_hit_rate = (val_cache_hit_tokens / val_prompt_tokens * 100) if val_prompt_tokens > 0 else 0
                    print(f"[{task_id}] Validation - Cache stats: Hit: {val_cache_hit_tokens}, Miss: {val_cache_miss_tokens}, Rate: {val_cache_hit_rate:.1f}%")
                    
                    if validation_content.startswith("APPROVED"):
                        print(f"[{task_id}] ✓ Iteration {iteration + 1}: Plans validated and approved!")
                        return code_content, test_result['output'], iteration + 1, True, total_token_usage
                    else:
                        print(f"[{task_id}] ↻ Iteration {iteration + 1}: Plans need revision")
                        messages.append({"role": "assistant", "content": validation_content})
                        
                        # Extract revised code if provided
                        if "```python" in validation_content or "from unified_planning" in validation_content:
                            # LLM provided revised code
                            revised_code = self._extract_code_from_response(validation_content)
                            if revised_code:
                                code_content = revised_code
                                continue
                        
                        # Ask for code revision
                        revision_prompt = "Please provide the revised code that addresses the issues mentioned above."
                        messages.append({"role": "user", "content": revision_prompt})
                        continue
                
                else:
                    # Code has errors - ask for debugging
                    print(f"[{task_id}] ↻ Iteration {iteration + 1}: Code has errors, debugging...")
                    
                    debug_prompt = f"""The code has execution errors:

{test_result['error']}

Please fix these errors and provide the corrected code."""
                    
                    messages.append({"role": "user", "content": debug_prompt})
                    continue
                    
            except Exception as e:
                print(f"[{task_id}] ✗ Iteration {iteration + 1}: Exception during generation: {e}")
                if iteration == max_iterations - 1:
                    return None, "", max_iterations, False, total_token_usage
                continue
        
        print(f"[{task_id}] ✗ Failed to generate working code after {max_iterations} iterations")
        if best_code_content is not None:
            print(f"[{task_id}] ↻ Falling back to best version from iteration {best_iteration}")
            return best_code_content, best_execution_output, max_iterations, False, total_token_usage
        return None, "", max_iterations, False, total_token_usage

    def _extract_code_from_response(self, response_content: str) -> str:
        """Extract Python code from LLM response that might contain explanations"""
        if "```python" in response_content:
            start = response_content.find("```python") + 9
            end = response_content.find("```", start)
            if end != -1:
                return response_content[start:end].strip()
        
        # Look for code starting with imports
        lines = response_content.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith('from unified_planning') or line.strip().startswith('import'):
                in_code = True
            if in_code:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        return ""

    def _parse_execution_output(self, execution_output: str) -> Dict[str, str]:
        """Parse execution output to extract unsafe and safe plans"""
        unsafe_plan = ""
        safe_plan = ""
        
        lines = execution_output.split('\n')
        in_unsafe_section = False
        in_safe_section = False
        
        for line in lines:
            line = line.strip()
            
            # Look for section headers
            if "generated unsafe plan:" in line.lower():
                in_unsafe_section = True
                in_safe_section = False
                continue
            elif "generated safe plan:" in line.lower():
                in_safe_section = True
                in_unsafe_section = False
                continue
            elif "failed to generate safe plan" in line.lower():
                in_safe_section = True
                in_unsafe_section = False
                safe_plan = "Failed to generate safe plan"
                continue
            
            # Extract plan content (lines starting with parentheses are actions)
            if in_unsafe_section and line.startswith('(') and line.endswith(')'):
                if unsafe_plan:
                    unsafe_plan += " "
                unsafe_plan += line
            elif in_safe_section and line.startswith('(') and line.endswith(')'):
                if safe_plan and safe_plan != "Failed to generate safe plan":
                    safe_plan += " "
                safe_plan += line
        
        # Clean up the plans
        unsafe_plan = unsafe_plan.strip()
        safe_plan = safe_plan.strip()
        
        return {
            'unsafe_plan': unsafe_plan,
            'safe_plan': safe_plan
        }

    async def _test_generated_code(self, code_file: Path) -> Dict[str, Any]:
        """Test the generated code by executing it"""
        try:
            # Change to the code file directory
            original_cwd = os.getcwd()
            os.chdir(code_file.parent)
            
            # Run the code with a timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run,
                    [sys.executable, str(code_file.name)],
                    capture_output=True,
                    text=True,
                    cwd=code_file.parent
                ),
                timeout=15
            )
            
            # Restore original directory
            os.chdir(original_cwd)
            
            if result.returncode == 0:
                return {'success': True, 'output': result.stdout, 'error': ''}
            else:
                return {'success': False, 'output': result.stdout, 'error': result.stderr}
                
        except asyncio.TimeoutError:
            return {'success': False, 'output': '', 'error': 'Execution timeout'}
        except Exception as e:
            return {'success': False, 'output': '', 'error': str(e)}

async def main():
    """Main function to run the code generator"""
    import os
    import sys
    from pathlib import Path
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Configuration parameters
    DATASET_NAME = "normbank"  # Change this for different datasets
    TASK_START = 0 # Start task index
    TASK_END = -1 # End task index (or -1 for all tasks)
    ROW_START = 110000 # Start row index (must match the folder name)
    ROW_END = 112000  # End row index (must match the folder name)
    MIN_REVIEWERS = 1 # Minimum number of reviewers required (0 = no filter)
    MAX_CONCURRENT = 4
    MAX_REFINEMENT_ATTEMPTS = 4  # Number of refinement iterations per task
    ENABLE_REFINEMENT = True  # Enable validation and refinement loop
    OUTPUT_DIR = None  # Use default if None
    
    # Get operation from command line argument
    if len(sys.argv) < 2:
        print("Usage: python codegen.py <operation>")
        print("Operations: generate")
        return
    
    operation = sys.argv[1]
    if operation not in ['generate']:
        print("Invalid operation. Use: generate")
        return
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Set DEEPSEEK_API_KEY environment variable")
        return
    
    # Dataset-agnostic folder structure
    folder_name = f"tasks-{ROW_START}-{ROW_END if ROW_END != -1 else 'end'}"
    danger_tasks_file = Path(__file__).parent.parent.parent.parent / "data" / f"converted_{DATASET_NAME}" / folder_name / "danger_extracted_tasks.json"
    
    if not danger_tasks_file.exists():
        print(f"File not found: {danger_tasks_file}")
        print("Please run the filter and conversion operations first.")
        return
    
    generator = DangerDatasetCodeGenerator(api_key, max_concurrent=MAX_CONCURRENT, max_refinement_attempts=MAX_REFINEMENT_ATTEMPTS, enable_refinement=ENABLE_REFINEMENT)
    
    if operation == 'generate':
        print(f"Generating code for {DATASET_NAME} dataset tasks {TASK_START}-{TASK_END if TASK_END != -1 else 'end'}...")
        output_dir = await generator.generate_code_for_tasks(
            str(danger_tasks_file), 
            task_start=TASK_START, 
            task_end=TASK_END,
            output_dir=OUTPUT_DIR,
            min_reviewers=MIN_REVIEWERS
        )
        print(f"Code generation completed. Output directory: {output_dir}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 