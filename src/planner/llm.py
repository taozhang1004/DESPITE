import os
import json
import re
import warnings
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import openai
from anthropic import Anthropic
from google import genai
from mistralai import Mistral
from together import Together
from .prompts import PLANNING_PROMPTS, DEFAULT_PLANNING_TYPE

# LLM Provider Default Models
DEFAULT_MODELS = {
    "openai": "gpt-4-turbo-preview",
    "anthropic": "claude-3-7-sonnet-latest",
    "google": "gemini-pro",
    "mistral": "mistral-medium-latest",
    "deepseek": "deepseek-chat",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
}

# LLM Generation Parameters
MAX_TOKENS = 4096
MAX_PLANNING_TOKENS = 512  # Lower limit for plan generation to prevent repetition
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
from ..utils.logic_utils import compare_logic_expressions

class LLMProvider:
    """Base class for LLM providers.
    
    This is a low-level interface for making raw LLM API calls.
    The generate_response method simply sends a prompt to the LLM and returns the raw response.
    For higher-level planning workflows, use LLMPlanner.generate_plan() instead.
    """
    def __init__(self):
        self._current_task_dir = None
        self._current_model = None
        self._current_ability = None
    
    def set_task_context(self, task_dir: Optional[str], model: Optional[str] = None, ability: Optional[str] = None):
        """Set the current task directory, model, and ability for error reporting."""
        self._current_task_dir = task_dir
        self._current_model = model
        self._current_ability = ability
    
    def _format_error_message(self, error_msg: str) -> str:
        """Format error message with task, model, and ability context if available."""
        context_parts = []
        
        # Add task context
        if self._current_task_dir:
            # Convert to relative path from workspace root
            try:
                task_dir_str = self._current_task_dir
                task_path = Path(task_dir_str)
                
                # Get workspace root (3 levels up from this file: src/core/llm_planner.py -> workspace)
                workspace_root = Path(__file__).parent.parent.parent.resolve()
                
                task_rel_path = None
                # If task_dir is already a relative path that looks valid (contains path separators)
                # and doesn't start with '..', use it as-is
                if not task_path.is_absolute() and '/' in task_dir_str and not task_dir_str.startswith('..'):
                    task_rel_path = task_dir_str
                else:
                    # Make absolute if it's not already
                    if not task_path.is_absolute():
                        # If it's relative, resolve it relative to current working directory
                        task_path = task_path.resolve()
                    
                    # Use os.path.relpath for more reliable relative path calculation
                    try:
                        rel_path = os.path.relpath(str(task_path), str(workspace_root))
                        # If relpath returns a path starting with '..', the task is outside workspace
                        if not rel_path.startswith('..'):
                            task_rel_path = rel_path
                        else:
                            # Task is outside workspace, use the task name
                            task_rel_path = task_path.name
                    except (ValueError, OSError):
                        # If relpath fails (e.g., different drives on Windows), use task name
                        task_rel_path = task_path.name
                
                if task_rel_path:
                    context_parts.append(f"task: {task_rel_path}")
            except Exception:
                # If anything fails, just use the original task_dir string
                context_parts.append(f"task: {self._current_task_dir}")
        
        # Add model context
        if self._current_model:
            context_parts.append(f"model: {self._current_model}")
        
        # Add ability context
        if self._current_ability:
            context_parts.append(f"ability: {self._current_ability}")
        
        # Format the error message with context
        if context_parts:
            context_str = ", ".join(context_parts)
            return f"{error_msg} ({context_str})"
        
        return error_msg
    
    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate a raw response from the LLM for the given prompt.
        
        This is a low-level method that directly calls the LLM API.
        For task-based planning (reading PDDL files, cleaning plans, etc.), 
        use LLMPlanner.generate_plan(task_dir) instead.
        
        Args:
            prompt: The prompt string to send to the LLM
            
        Returns:
            The raw response text from the LLM, or None if generation failed
        """
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = DEFAULT_MODELS["openai"]):
        super().__init__()
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate_response(self, prompt: str) -> Optional[str]:
        try:
            # GPT-5 (exactly "gpt-5") uses a different API endpoint with high reasoning effort
            if self.model.lower() == "gpt-5":
                result = self.client.responses.create(
                    model="gpt-5",
                    input=prompt,
                    reasoning={"effort": "high"},
                    text={"verbosity": "low"}
                )
                return result.output_text.strip()
            else:
                # Standard OpenAI models (including gpt-5-mini, gpt-4, etc.) use chat.completions.create
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                    # No temperature - use model defaults
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            print(self._format_error_message(f"OpenAI error: {str(e)}"))
            return None

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = DEFAULT_MODELS["anthropic"]):
        super().__init__()
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate_response(self, prompt: str) -> Optional[str]:
        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS
                # No temperature - use model defaults
            )
            # Check if response has content
            if not response.content or len(response.content) == 0:
                print(self._format_error_message("Anthropic error: Empty response content"))
                return None
            
            # Check if the first content block has text attribute
            first_block = response.content[0]
            if not hasattr(first_block, 'text') or not first_block.text:
                print(self._format_error_message("Anthropic error: Response content block has no text"))
                return None
            
            return first_block.text.strip()
        except Exception as e:
            print(self._format_error_message(f"Anthropic error: {str(e)}"))
            return None

class GoogleProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = DEFAULT_MODELS["google"]):
        super().__init__()
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate_response(self, prompt: str) -> Optional[str]:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            # Extract text from response, handling cases where response contains non-text parts
            # (e.g., thought_signature in newer Gemini models)
            # Access text parts directly to avoid SDK warning
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    # Extract only text parts, ignoring non-text parts like thought_signature
                    text_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    if text_parts:
                        return ' '.join(text_parts).strip()
            
            # Fallback to response.text if direct access doesn't work
            # Suppress warning about non-text parts (e.g., thought_signature) in response
            # The warning is harmless - we only need the text content
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*non-text parts.*")
                warnings.filterwarnings("ignore", category=UserWarning)
                return response.text.strip()
        except Exception as e:
            print(self._format_error_message(f"Google error: {str(e)}"))
            return None

class MistralProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = DEFAULT_MODELS["mistral"]):
        super().__init__()
        self.client = Mistral(api_key=api_key)
        self.model = model

    def generate_response(self, prompt: str) -> Optional[str]:
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
                # No temperature - use model defaults
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(self._format_error_message(f"Mistral error: {str(e)}"))
            return None

class DeepSeekProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = DEFAULT_MODELS["deepseek"]):
        super().__init__()
        self.client = openai.OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.model = model

    def generate_response(self, prompt: str) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
                # No temperature - use model defaults
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(self._format_error_message(f"DeepSeek error: {str(e)}"))
            return None

class TogetherProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = DEFAULT_MODELS["together"], max_tokens: int = MAX_TOKENS):
        super().__init__()
        self.client = Together(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate_response(self, prompt: str, max_tokens: Optional[int] = None) -> Optional[str]:
        try:
            tokens = max_tokens if max_tokens is not None else self.max_tokens
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=tokens
                # No temperature - use model defaults
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(self._format_error_message(f"Together AI error: {str(e)}"))
            return None

class LLMPlanner:
    def __init__(self, provider: str = "openai", model: Optional[str] = None, planning_type: str = DEFAULT_PLANNING_TYPE, temperature: float = 0, max_tokens: int = MAX_TOKENS, timeout: int = 300):
        """Initialize the LLM planner with the specified provider, model, and planning type.

        Args:
            provider: LLM provider name (openai, anthropic, etc.)
            model: Specific model to use (defaults to provider's default)
            planning_type: Type of planning prompt to use (default, safety_first, efficient, temporal)
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens for LLM generation
            timeout: Timeout for LLM calls in seconds
        """
        self.planning_type = planning_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.provider_name = provider.lower()
        # Store the model name for context
        if model is None:
            model = DEFAULT_MODELS.get(self.provider_name)
        self.model_name = model
        self.provider = self._create_provider(provider, model)
    
    def set_task_context(self, task_dir: str, ability: Optional[str] = None):
        """Set the task directory and ability context for error reporting."""
        # Format model name as provider/model for clarity
        model_str = f"{self.provider_name}/{self.model_name}" if self.model_name else None
        self.provider.set_task_context(task_dir, model=model_str, ability=ability)
    
    def _get_relative_task_path(self, task_dir: Optional[str]) -> Optional[str]:
        """Get relative path of task directory from workspace root.
        
        Args:
            task_dir: Task directory path (can be absolute or relative)
            
        Returns:
            Relative path from workspace root, or task name if outside workspace, or None
        """
        if not task_dir:
            return None
        
        try:
            task_path = Path(task_dir)
            
            # Get workspace root (3 levels up from this file: src/core/llm_planner.py -> workspace)
            workspace_root = Path(__file__).parent.parent.parent.resolve()
            
            # If task_dir is already a relative path that looks valid
            if not task_path.is_absolute() and '/' in task_dir and not task_dir.startswith('..'):
                return task_dir
            
            # Make absolute if it's not already
            if not task_path.is_absolute():
                task_path = task_path.resolve()
            
            # Use os.path.relpath for more reliable relative path calculation
            try:
                rel_path = os.path.relpath(str(task_path), str(workspace_root))
                # If relpath returns a path starting with '..', the task is outside workspace
                if not rel_path.startswith('..'):
                    return rel_path
                else:
                    # Task is outside workspace, use the task name
                    return task_path.name
            except (ValueError, OSError):
                # If relpath fails (e.g., different drives on Windows), use task name
                return task_path.name
        except Exception:
            # If anything fails, just use the original task_dir string
            return task_dir

    def _create_provider(self, provider: str, model: Optional[str] = None) -> LLMProvider:
        """Create an LLM provider based on the specified name and model."""
        provider = provider.lower()
        
        if model is None:
            model = DEFAULT_MODELS.get(provider)
            if model is None:
                raise ValueError(f"No default model specified for provider: {provider}")
        
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return OpenAIProvider(api_key, model)
            
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            return AnthropicProvider(api_key, model)
            
        elif provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            return GoogleProvider(api_key, model)
            
        elif provider == "mistral":
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY environment variable not set")
            return MistralProvider(api_key, model)
            
        elif provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable not set")
            return DeepSeekProvider(api_key, model)
            
        elif provider == "together":
            api_key = os.getenv("TOGETHER_API_KEY")
            if not api_key:
                raise ValueError("TOGETHER_API_KEY environment variable not set")
            return TogetherProvider(api_key, model, self.max_tokens)
            
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _read_pddl_files(self, task_dir: str) -> tuple[str, str]:
        """Read domain and problem PDDL files from a task directory."""
        domain_path = os.path.join(task_dir, "domain.pddl")
        problem_path = os.path.join(task_dir, "problem.pddl")
        
        with open(domain_path, 'r') as f:
            domain = f.read()
        with open(problem_path, 'r') as f:
            problem = f.read()
            
        return domain, problem

    @staticmethod
    def _read_task_info(task_dir: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Read task information from code.py and result.json files.
        
        Args:
            task_dir: Path to the task directory containing code.py and result.json
            
        Returns:
            Tuple containing:
            - unsafe_plan: The unsafe plan string from result.json, or None if not found
            - dangerous_actions: Dictionary of dangerous actions from code.py, or None if not found
        """
        try:
            # Read unsafe plan from result.json
            result_path = os.path.join(task_dir, "result.json")
            unsafe_plan = None
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    result_data = json.load(f)
                    unsafe_plan = result_data.get("generated_plans", {}).get("unsafe_plan")
            
            # Read dangerous actions from code.py
            code_path = os.path.join(task_dir, "code.py")
            dangerous_actions = None
            if os.path.exists(code_path):
                dangerous_actions = LLMPlanner._extract_dangerous_actions_static(code_path)
            
            return unsafe_plan, dangerous_actions
            
        except Exception as e:
            print(f"Error reading task info from {task_dir}: {str(e)}")
            return None, None

    @staticmethod
    def _extract_dangerous_actions_static(code_path: str) -> Optional[Dict[str, Any]]:
        """Extract dangerous actions dictionary from code.py file.
        
        Args:
            code_path: Path to the code.py file
            
        Returns:
            Dictionary of dangerous actions or None if extraction fails
        """
        try:
            with open(code_path, 'r') as f:
                code_content = f.read()
            
            # Find the create_danger_actions method
            method_pattern = r'def create_danger_actions\([^)]*\):.*?(?=\n    def|\nif __name__|\Z)'
            method_match = re.search(method_pattern, code_content, re.DOTALL)
            
            if not method_match:
                print(f"create_danger_actions method not found in {code_path}")
                return None
            
            method_content = method_match.group(0)
            
            # Find the dangerous_actions dictionary assignment
            dict_pattern = r'dangerous_actions\s*=\s*\{([^}]+)\}'
            dict_match = re.search(dict_pattern, method_content, re.DOTALL)
            
            if not dict_match:
                print(f"dangerous_actions dictionary not found in {code_path}")
                return None
            
            # Extract the dictionary content
            dict_content = dict_match.group(1)
            
            # Parse the dictionary content to extract action names and their conditions
            dangerous_actions = {}
            
            # More sophisticated parsing to handle multi-line expressions
            lines = dict_content.split('\n')
            current_action = None
            current_condition = []
            brace_count = 0
            bracket_count = 0
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Check for action name pattern: "action_name":
                action_match = re.match(r'"([^"]+)"\s*:\s*(.*)', line)
                if action_match and brace_count == 0 and bracket_count == 0:
                    # Save previous action if exists
                    if current_action and current_condition:
                        condition_str = '\n'.join(current_condition).strip()
                        # Remove trailing comma if present
                        condition_str = condition_str.rstrip(',').strip()
                        # Convert to simplified format for easier LLM generation
                        simplified_condition = LLMPlanner._simplify_condition_format_static(condition_str)
                        dangerous_actions[current_action] = simplified_condition
                    
                    # Start new action
                    current_action = action_match.group(1)
                    remaining_line = action_match.group(2)
                    current_condition = [remaining_line] if remaining_line else []
                    
                    # Count braces and brackets in the remaining part
                    brace_count = remaining_line.count('{') - remaining_line.count('}')
                    bracket_count = remaining_line.count('[') - remaining_line.count(']')
                else:
                    # Continue building current condition
                    if current_action:
                        current_condition.append(line)
                        brace_count += line.count('{') - line.count('}')
                        bracket_count += line.count('[') - line.count(']')
            
            # Don't forget the last action
            if current_action and current_condition:
                condition_str = '\n'.join(current_condition).strip()
                # Remove trailing comma if present
                condition_str = condition_str.rstrip(',').strip()
                # Convert to simplified format for easier LLM generation
                simplified_condition = LLMPlanner._simplify_condition_format_static(condition_str)
                dangerous_actions[current_action] = simplified_condition
            
            return dangerous_actions
            
        except Exception as e:
            print(f"Error extracting dangerous actions from {code_path}: {str(e)}")
            return None
    
    @staticmethod
    def _simplify_condition_format_static(condition_str: str) -> str:
        """Convert Python logic format to simplified format that logic_utils can parse.
        
        Converts Python UP logic to string format compatible with AST parsing:
        - And([cond1, cond2]) -> And([cond1, cond2])
        - Not(self.condition) -> Not(condition)
        - self.condition -> condition
        - object_at(self.obj, self.location) -> object_at(obj, location)
        - Removes comments (# ...) from the condition string
        """
        import re
        
        # Remove trailing comma if present (from Python code)
        condition_str = condition_str.rstrip(',').strip()
        
        # Remove comments (Python-style # comments) - do this before other processing
        lines = condition_str.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove comments (everything after #)
            comment_pos = line.find('#')
            if comment_pos != -1:
                line = line[:comment_pos]
            cleaned_lines.append(line.rstrip())
        condition_str = '\n'.join(cleaned_lines)
        
        # Remove self. prefixes from fluents (handles both simple and nested in function calls)
        simplified = re.sub(r'self\.(\w+)', r'\1', condition_str)
        
        # Keep the Python And()/Or()/Not() structure for logic_utils to parse
        # The format should be parseable by ast.parse
        return simplified

    def _compare_logic_expressions(self, expr1: str, expr2: str) -> bool:
        """Compare two logical expressions for equivalence using AST parsing.
        
        Args:
            expr1: First logical expression as string
            expr2: Second logical expression as string
            
        Returns:
            True if expressions are logically equivalent, False otherwise
        """
        return compare_logic_expressions(expr1, expr2)

    def _create_prompt(self, domain: str, problem: str) -> str:
        """Create a prompt for the LLM to generate a plan based on the configured planning type."""
        template = PLANNING_PROMPTS.get(self.planning_type, PLANNING_PROMPTS[DEFAULT_PLANNING_TYPE])
        return template.format(domain=domain, problem=problem)

    def _clean_plan(self, plan_text: str) -> List[str]:
        """Clean the plan text to ensure only valid actions are included."""
        # Split into lines and filter
        lines = [line.strip() for line in plan_text.split('\n')]
        # Only keep lines that start with '(' and end with ')'
        actions = [line for line in lines if line.startswith('(') and line.endswith(')')]
        return actions

    def _clean_response_with_deepseek(self, raw_response: str, output_format: str) -> Optional[str]:
        """Use deepseek-chat to clean/extract the desired output format from a raw LLM response.
        
        This method helps handle cases where LLMs don't follow the output format exactly
        and include extra content like reasoning, explanations, or repeated content.
        
        Args:
            raw_response: The raw response from the LLM that may contain extra content
            output_format: Type of output format expected ("plan", "action_names", "json")
            
        Returns:
            Cleaned response containing only the desired format, or None if cleaning fails
        """
        try:
            # Create a deepseek provider for cleaning (only use if not already deepseek)
            deepseek_provider = None
            if self.provider.__class__.__name__ != "DeepSeekProvider":
                deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
                if deepseek_api_key:
                    deepseek_provider = DeepSeekProvider(deepseek_api_key)
                    # Set task context if available (copy from main provider)
                    if self.provider._current_task_dir:
                        deepseek_provider.set_task_context(
                            self.provider._current_task_dir,
                            model=self.provider._current_model,
                            ability=self.provider._current_ability
                        )
                else:
                    print("Warning: DEEPSEEK_API_KEY not set, skipping response cleaning")
                    return None
            else:
                # Already using deepseek, use the same provider
                deepseek_provider = self.provider
            
            # Create format-specific cleaning prompts
            if output_format == "plan":
                cleaning_prompt = f"""Extract the PDDL action plan from the following LLM response.

The response may contain reasoning, explanations, or other text. Extract ONLY the list of PDDL actions.

REQUIRED OUTPUT FORMAT:
Each action must be on its own line and enclosed in parentheses like this:
(action_name)
(action_name param1 param2)

IMPORTANT:
- Extract only the FIRST occurrence of the plan (do not repeat actions)
- Remove all reasoning, explanations, markdown formatting, and extra text
- Output ONLY the actions, nothing else
- Each action should be on its own line

RAW RESPONSE:
{raw_response}

EXTRACTED PLAN (output only the actions, nothing else):"""
            
            elif output_format == "action_names":
                cleaning_prompt = f"""Extract the action names from the following LLM response.

The response may contain reasoning, explanations, or other text. Extract ONLY the action names.

REQUIRED OUTPUT FORMAT:
One action name per line, without parentheses or parameters, like this:
action1
action2
action3

IMPORTANT:
- Extract only the action names (without parentheses or parameters)
- Remove all reasoning, explanations, markdown formatting, and extra text
- Output ONLY the action names, one per line, nothing else
- Do not repeat action names

RAW RESPONSE:
{raw_response}

EXTRACTED ACTION NAMES (output only the action names, nothing else):"""
            
            elif output_format == "json":
                cleaning_prompt = f"""Extract the JSON object from the following LLM response.

The response may contain reasoning, explanations, or other text. Extract ONLY the JSON object.

REQUIRED OUTPUT FORMAT:
A valid JSON object with no extra text before or after it. Each value in the JSON should be a non-empty string.

IMPORTANT:
- Extract only the JSON object
- Remove all reasoning, explanations, markdown code block markers, and extra text
- Output ONLY the JSON object, nothing else
- Ensure the JSON is valid and all values are non-empty strings
- If any value is empty or invalid, exclude that key-value pair

RAW RESPONSE:
{raw_response}

EXTRACTED JSON (output only the JSON object, nothing else):"""
            
            elif output_format == "logic_expression":
                cleaning_prompt = f"""Extract and format the logic expression from the following response.

REQUIRED FORMAT:
Python format using And([...]), Or([...]), Not(...), Equals(...):
- And([condition1, condition2, condition3])
- Or([condition1, condition2])
- And([condition1, Not(condition2), condition3])
- And([condition1, Equals(variable, value), condition2])
- Or([And([condition1, condition2]), condition3])

RULES:
- Use And([...]) for AND operations
- Use Or([...]) for OR operations
- Use Not(...) for NOT operations
- Use Equals(variable, value) for equality checks (NOT "variable = value")
- Function calls use parentheses: info_verified(financing_info)
- Remove all explanations, markdown, and extra text
- Output ONLY the expression

EXAMPLES:
- "condition1 AND condition2" → "And([condition1, condition2])"
- "NOT condition1 AND condition2" → "And([Not(condition1), condition2])"
- "volume_setting = 0" → "Equals(volume_setting, 0)"
- "condition1 AND volume_setting = 0" → "And([condition1, Equals(volume_setting, 0)])"
- "condition1 OR condition2" → "Or([condition1, condition2])"
- "(condition1 AND condition2) OR condition3" → "Or([And([condition1, condition2]), condition3])"

RAW RESPONSE:
{raw_response}

EXTRACTED EXPRESSION:"""
            
            else:
                print(f"Unknown output format for cleaning: {output_format}")
                return None
            
            # Use deepseek to clean the response
            cleaned_response = deepseek_provider.generate_response(cleaning_prompt)
            return cleaned_response
            
        except Exception as e:
            print(f"Error cleaning response with deepseek: {str(e)}")
            return None

    def _convert_plan_format(self, original_response: str) -> Optional[str]:
        """Use the LLM to convert plan format to proper PDDL format."""
        conversion_prompt = f"""Convert the following plan to proper PDDL format. 

REQUIRED OUTPUT FORMAT:
Each action must be on its own line and enclosed in parentheses like this:
(action_name)
(action_name param1 param2)

INPUT PLAN TO CONVERT:
{original_response}

CONVERTED PLAN (output only the actions, nothing else):"""
        
        try:
            converted_response = self.provider.generate_response(conversion_prompt)
            return converted_response
        except Exception as e:
            print(f"Error converting plan format: {str(e)}")
            return None

    def generate_plan(self, task_dir: str) -> tuple[Optional[List[str]], Optional[str], Optional[str]]:
        """Generate a safe plan for a given task using the LLM.
        
        This is a high-level method that handles the full planning workflow:
        - Reads PDDL domain and problem files from task_dir
        - Creates a planning prompt based on the configured planning_type
        - Calls the LLM provider to generate a response
        - Cleans and parses the response into a list of actions
        - Handles format conversion if needed
        
        For low-level raw LLM calls (used by other benchmarks like danger_identification),
        use self.provider.generate_response(prompt) directly.
        
        Args:
            task_dir: Path to task directory containing domain.pddl and problem.pddl
            
        Returns:
            tuple: (plan, cleaned_response, llm_response) where:
                - plan is a list of action strings (PDDL format) or None if generation failed
                - cleaned_response is the cleaned response from deepseek or None if cleaning failed
                - llm_response is the raw response from the LLM or None if generation failed
        """
        try:
            # Read PDDL files
            domain, problem = self._read_pddl_files(task_dir)
            
            # Create prompt
            prompt = self._create_prompt(domain, problem)
            
            # Generate plan using the provider
            # For Together AI, use limited tokens to prevent repetition
            if self.provider.__class__.__name__ == "TogetherProvider":
                llm_response = self.provider.generate_response(prompt, max_tokens=MAX_PLANNING_TOKENS)
            else:
                llm_response = self.provider.generate_response(prompt)
            if not llm_response:
                return None, None, None
            
            # Toggle for cleaning response with deepseek
            ENABLE_CLEAN_RESPONSE = True  # Set to False to skip cleaning
            # ENABLE_CLEAN_RESPONSE = False  # Set to False to skip cleaning
            
            # Clean response using deepseek to extract only the plan (if enabled)
            cleaned_response = None
            if ENABLE_CLEAN_RESPONSE:
                cleaned_response = self._clean_response_with_deepseek(llm_response, "plan")
                if cleaned_response:
                    # Use cleaned response for parsing
                    plan = self._clean_plan(cleaned_response)
                else:
                    # Fall back to original response if cleaning fails
                    plan = self._clean_plan(llm_response)
            else:
                # Skip cleaning, use original response directly
                plan = self._clean_plan(llm_response)
            
            # If no valid actions found, try format conversion
            if not plan:
                # Get relative path and model for error message
                task_rel_path = self._get_relative_task_path(task_dir)
                model_str = f"{self.provider_name}/{self.model_name}"
                print(f"No valid PDDL actions found in response for {task_rel_path} (model: {model_str}), attempting format conversion...")
                converted_response = self._convert_plan_format(llm_response)
                if converted_response:
                    plan = self._clean_plan(converted_response)
                    if plan:
                        print(f"Successfully converted format for {task_rel_path} (model: {model_str})")
                        # Return converted response as cleaned_response
                        return plan, converted_response, llm_response
                    else:
                        print(f"Format conversion failed for {task_rel_path} (model: {model_str})")
                else:
                    print(f"Format conversion request failed for {task_rel_path} (model: {model_str})")
            
            return plan, cleaned_response, llm_response
            
        except Exception as e:
            task_rel_path = self._get_relative_task_path(task_dir)
            model_str = f"{self.provider_name}/{self.model_name}"
            if task_rel_path:
                print(f"Error generating plan for {task_rel_path} (model: {model_str}): {str(e)}")
            else:
                print(f"Error generating plan for {task_dir} (model: {model_str}): {str(e)}")
            return None, None, None
    
    def generate_raw_response(self, prompt: str) -> str:
        """Generate raw response from LLM for a given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            str: Raw response from the LLM
        """
        try:
            response = self.provider.generate_response(prompt)
            return response if response else ""
        except Exception as e:
            print(f"Error generating raw response: {str(e)}")
            return ""