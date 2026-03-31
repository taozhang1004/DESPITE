"""
Logic utilities for comparing and analyzing logical expressions.

This module provides utilities for parsing, normalizing, and comparing logical expressions
used in planning domains, particularly for dangerous action conditions.
"""

import ast
import os
from pathlib import Path
from typing import Optional, Dict, Any


def _get_relative_task_path(task_dir: Optional[str]) -> Optional[str]:
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
        
        # Get workspace root (3 levels up from this file: src/utils/logic_utils.py -> workspace)
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


def normalize_ast(node: ast.AST) -> str:
    """Normalize AST to handle commutative operations like And/Or.
    
    This function normalizes abstract syntax trees to handle commutative logical
    operations, ensuring that expressions like And([A, B]) and And([B, A]) are
    considered equivalent.
    
    Args:
        node: The AST node to normalize
        
    Returns:
        A normalized string representation of the AST
    """
    # Handle Expression nodes (from ast.parse with mode='eval')
    if isinstance(node, ast.Expression):
        return normalize_ast(node.body)
    
    # Handle Tuple nodes (can occur from trailing commas like "And([...]),")
    # Unwrap single-element tuples
    if isinstance(node, ast.Tuple):
        if len(node.elts) == 1:
            # Unwrap the tuple and process the inner expression
            return normalize_ast(node.elts[0])
        else:
            # Multiple elements in tuple - shouldn't happen in our use case
            return ast.dump(node)
    
    if isinstance(node, ast.Call):
        # Handle function calls like And([...]) or Or([...])
        if isinstance(node.func, ast.Name) and node.func.id in ['And', 'Or']:
            # Sort arguments for commutative operations
            if len(node.args) == 1 and isinstance(node.args[0], ast.List):
                # Extract and sort the list elements
                elements = node.args[0].elts
                sorted_elements = sorted([ast.dump(elem) for elem in elements])
                return f"{node.func.id}({sorted_elements})"
        return ast.dump(node)
    else:
        return ast.dump(node)


def compare_logic_expressions(expr1: str, expr2: str, task_dir: Optional[str] = None, model: Optional[str] = None) -> bool:
    """Compare two logical expressions for equivalence using AST parsing.
    
    This function parses logical expressions into abstract syntax trees and compares
    them for logical equivalence, handling commutative operations like And/Or.
    
    Supports both formats:
    - Simple: "condition1 AND condition2 AND NOT condition3"
    - Python: "And([condition1, condition2, Not(condition3)])"
    
    Args:
        expr1: First logical expression as string
        expr2: Second logical expression as string
        task_dir: Optional task directory path for error reporting
        model: Optional model name for error reporting (format: "provider/model")
        
    Returns:
        True if expressions are logically equivalent, False otherwise
        
    Examples:
        >>> compare_logic_expressions(
        ...     "And([Not(using_gentle_grip), object_at_egg_tray_shelf])",
        ...     "And([object_at_egg_tray_shelf, Not(using_gentle_grip)])"
        ... )
        True
        
        >>> compare_logic_expressions(
        ...     "condition1 AND condition2",
        ...     "condition2 AND condition1"
        ... )
        True
    """
    try:
        # Convert simple format to Python format for parsing
        expr1_python = _convert_to_python_format(expr1)
        expr2_python = _convert_to_python_format(expr2)
        
        # Parse both expressions into AST
        tree1 = ast.parse(expr1_python, mode='eval')
        tree2 = ast.parse(expr2_python, mode='eval')
        
        # Normalize and compare the AST structures
        return normalize_ast(tree1) == normalize_ast(tree2)
        
    except SyntaxError as e:
        task_path = _get_relative_task_path(task_dir)
        context_parts = []
        if task_path:
            context_parts.append(f"task: {task_path}")
        if model:
            context_parts.append(f"model: {model}")
        if context_parts:
            print(f"Syntax error comparing expressions: {str(e)} ({', '.join(context_parts)})")
        else:
            print(f"Syntax error comparing expressions: {str(e)}")
        return False
    except Exception as e:
        task_path = _get_relative_task_path(task_dir)
        context_parts = []
        if task_path:
            context_parts.append(f"task: {task_path}")
        if model:
            context_parts.append(f"model: {model}")
        if context_parts:
            print(f"Error comparing logic expressions: {str(e)} ({', '.join(context_parts)})")
        else:
            print(f"Error comparing logic expressions: {str(e)}")
        return False


def _convert_to_python_format(expr: str) -> str:
    """Clean and normalize Python format logic expressions.
    
    Cleans up Python format expressions by:
    - Removing comments
    - Normalizing whitespace
    - Removing trailing commas
    
    Also supports converting simple format (AND/OR/NOT) to Python format as fallback.
    
    Args:
        expr: Logic expression in Python format (or simple format as fallback)
        
    Returns:
        Cleaned Python format expression that can be parsed by ast.parse
    """
    import re
    
    # Remove comments (Python-style # comments)
    lines = expr.split('\n')
    cleaned_lines = []
    for line in lines:
        comment_pos = line.find('#')
        if comment_pos != -1:
            line = line[:comment_pos]
        cleaned_lines.append(line.rstrip())
    expr = '\n'.join(cleaned_lines)
    
    # Check if already in Python format
    expr_stripped = expr.strip()
    is_python_format = (expr_stripped.startswith(('And([', 'Or([', 'Not(')) and 
                      ' AND ' not in expr.upper() and ' OR ' not in expr.upper())
    
    if is_python_format:
        # Clean up Python format: normalize whitespace and remove trailing commas
        expr = ' '.join(expr.split('\n'))  # Remove newlines
        expr = ' '.join(expr.split())  # Normalize spaces
        
        # Convert equality operators (=) to Equals() function calls inside Python format
        # This handles cases like "And([condition1, variable = value, condition2])"
        # Pattern: match "variable = value" but not "==" or ":=" or "Equals(...)"
        if ' = ' in expr:
            # Convert all "variable = value" patterns (but not == or :=)
            expr = re.sub(r'(\w+)\s*=\s*(\d+|\w+)(?![=:])', r'Equals(\1, \2)', expr)
        
        expr = re.sub(r',\s*([\]\)])', r'\1', expr)  # Remove trailing commas before brackets/parens
        expr = re.sub(r',\s*$', '', expr)  # Remove trailing comma at end
        return expr
    
    # Fallback: Convert simple AND/OR/NOT format to Python format (for backward compatibility)
    # This is a simplified version - most expressions should already be in Python format
    
    # Convert NOT expressions
    expr = re.sub(r'NOT\s+(\w+)', r'Not(\1)', expr, flags=re.IGNORECASE)
    # Convert equality operators
    expr = re.sub(r'(\w+)\s*=\s*(\d+|\w+)', r'Equals(\1, \2)', expr)
    
    # Convert AND/OR to Python format
    if ' AND ' in expr.upper() or ' OR ' in expr.upper():
        # Simple conversion: split by AND/OR and wrap
        if ' AND ' in expr.upper():
            parts = [p.strip() for p in re.split(r'\s+AND\s+', expr, flags=re.IGNORECASE)]
            return f"And([{', '.join(parts)}])"
        elif ' OR ' in expr.upper():
            parts = [p.strip() for p in re.split(r'\s+OR\s+', expr, flags=re.IGNORECASE)]
            return f"Or([{', '.join(parts)}])"
    
    return expr
