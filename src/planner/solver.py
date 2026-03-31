from unified_planning.shortcuts import *
from unified_planning.engines import PlanGenerationResultStatus, ValidationResultStatus
from unified_planning.io import PDDLWriter, PDDLReader, ANMLWriter
from unified_planning.engines import SequentialPlanValidator, TimeTriggeredPlanValidator
from unified_planning.plans import SequentialPlan, ActionInstance, Plan
import os
import argparse
import inspect
import signal
from typing import Tuple, List, Dict, Any, Callable
from ..utils.planning_utils import compile_conditional_effects, convert_compiled_plan_to_original, get_unsafe_plan, get_safe_plan, validate_custom_plan, export_problem_files

class BasePlanner:
    def __init__(self):
        """Initialize the base planner with common attributes."""
        # Common attributes that examples can override
        self.safety_fluents: List = []
        self.objects: List = []

        # Fluent organization - automatically populated by subclasses
        self.normal_fluents: List = []
        self.context_fluents: List = []

    # Abstract interface that subclasses must implement
    def create_base_actions(self) -> Tuple:
        """Return tuple of base actions for the domain."""
        raise NotImplementedError("Subclasses must implement create_base_actions()")
    
    def create_danger_actions(self, base_actions: Tuple) -> Tuple:
        """Apply danger effects to base actions and return modified actions."""
        raise NotImplementedError("Subclasses must implement create_danger_actions()")
    
    def _set_initial_state(self, problem: Problem) -> None:
        """Set domain-specific initial state on the problem."""
        raise NotImplementedError("Subclasses must implement _set_initial_state()")
    
    def _set_goals(self, problem: Problem) -> None:
        """Set domain-specific goals on the problem."""
        raise NotImplementedError("Subclasses must implement _set_goals()")

    # Problem creation methods
    def create_problem_base(self, name: str, actions: List, 
                           initial_state_setter: Callable[[Problem], None], 
                           goal_setter: Callable[[Problem], None], 
                           goal_danger: int = 0, basic: bool = False) -> Problem:
        """
        Base method for creating planning problems with standard patterns.
        
        Args:
            name: Problem name
            actions: List of actions to add
            initial_state_setter: Function that sets initial state on the problem
            goal_setter: Function that sets goals on the problem  
            goal_danger: 0 for safe planning, >0 for allowing danger
            basic: True to exclude safety components
        """
        problem = Problem(name)
        
        # Add fluents
        fluents = self.normal_fluents + self.context_fluents + (self.safety_fluents if not basic else [])
        for fluent in fluents:
            # Default values - False for boolean, 0 for danger/integer fluents
            default_val = 0 if fluent.type.is_int_type() else False
            problem.add_fluent(fluent, default_initial_value=default_val)
        
        # Add actions and objects
        for action in actions:
            problem.add_action(action)
        problem.add_objects(self.objects)
        
        # Set initial state and goals (delegate to example)
        initial_state_setter(problem)
        goal_setter(problem)
        
        # Initialize danger counter and add safety goal if using safety
        if not basic and self.safety_fluents:
            danger_fluent = next((f for f in self.safety_fluents if f.name == "danger"), None)
            if danger_fluent:
                problem.set_initial_value(danger_fluent, 0)
                problem.add_goal(GE(danger_fluent, 1) if goal_danger > 0 else Equals(danger_fluent, 0))
        
        return problem

    def create_problem(self, name: str, goal_danger: int = 0, basic: bool = False) -> Problem:
        """
        Standard create_problem implementation that works for most examples.

        Examples only need to implement the abstract methods above.
        """
        # Get actions (either base only or with danger effects)
        actions = self.create_base_actions() if basic else self.create_danger_actions(self.create_base_actions())

        # Use the base problem creation method
        return self.create_problem_base(
            name=name,
            actions=actions,
            initial_state_setter=self._set_initial_state,
            goal_setter=self._set_goals,
            goal_danger=goal_danger,
            basic=basic
        )

    # Utility methods
    def apply_danger_effects(self, base_actions: Tuple, dangerous_actions: Dict[str, Any]) -> Tuple:
        """
        Apply danger effects to actions based on dangerous action mapping.

        Args:
            base_actions: Tuple of base actions
            dangerous_actions: Dict mapping action names to danger conditions

        Returns:
            Tuple of actions with danger effects applied
        """
        # Find danger fluent
        danger_fluent = next((f for f in self.safety_fluents if f.name == "danger"), None)

        if not danger_fluent:
            return base_actions

        # Add danger effects to matching actions
        for action in base_actions:
            if action.name in dangerous_actions:
                condition = dangerous_actions[action.name]

                # Increment danger counter
                if hasattr(action, 'duration') and action.duration is not None:
                    # Durative action - use EndTiming
                    action.add_effect(EndTiming(), danger_fluent, Plus(danger_fluent, 1), condition)
                else:
                    # Instantaneous action - use regular add_effect
                    action.add_effect(danger_fluent, Plus(danger_fluent, 1), condition)

        return base_actions

    def solve_problem(self, safe: bool = True) -> Tuple[bool, str]:
        """Standard solve method that examples can use directly."""
        problem = self.create_problem("domain", goal_danger=0 if safe else 1)
        success, plan_str, _ = self.generate_plan(problem)
        return success, plan_str

    def run_cli(self):
        """
        Standard CLI interface for all domain examples.
        
        Handles:
        - Argument parsing for custom plan validation
        - Running unsafe/safe plan generation
        - Exporting problem files
        
        Examples can simply call: DomainPlanner().run_cli()
        """
        
        # Set up argument parser
        parser = argparse.ArgumentParser(description='Run the domain planner')
        parser.add_argument('-plan', type=str, help='Custom plan to validate (feasibility + safety)')
        parser.add_argument('-si', type=str, help='Custom plan to validate for safety intention only')
        args = parser.parse_args()

        # Set up timeout for the entire CLI execution
        timeout_seconds = 15.0  # Timeout for entire CLI pipeline

        def timeout_handler(signum, frame):
            raise TimeoutError("CLI execution timed out")

        # Set timeout signal
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))

        if args.si is not None:  # Allow empty string
            # Validate safety intention only
            from src.utils.planning_utils import validate_safety_intention, compile_conditional_effects
            import re
            up.shortcuts.get_environment().credits_stream = None

            # Parse plan
            actions = re.findall(r'\([^)]+\)', args.si)
            plan_str = '\n'.join(actions)

            # Create problem and validate
            safe_problem = self.create_problem("safe", goal_danger=0)
            _, _, _, action_mapping = compile_conditional_effects(safe_problem)

            is_safe, message = validate_safety_intention(plan_str, safe_problem, action_mapping)
            print(f"Safety Intention: {'Yes' if is_safe else 'No'}")
            print(f"Details: {message}")
        elif args.plan:
            # If plan argument is provided, validate the plan (feasibility + safety)
            score, message = validate_custom_plan(self, args.plan)
            print(f"Validation result: {message}")
            print(f"Score: {score}")
        else:
            # If no plan argument, run the full pipeline
            # Get unsafe plan
            up.shortcuts.get_environment().credits_stream = None
            success, unsafe_plan = get_unsafe_plan(self)
            # Get safe plan
            success, safe_plan = get_safe_plan(self)

            # Export problem files - get caller's directory
            caller_frame = inspect.currentframe().f_back
            caller_file = caller_frame.f_globals['__file__']
            output_dir = os.path.dirname(os.path.abspath(caller_file))
            export_problem_files(self, output_dir)
        
        # Cancel the alarm
        signal.alarm(0)

    # Planning and validation methods
    def generate_plan(self, problem: Problem) -> Tuple[bool, str, Plan]:
        """Generate a plan for the given problem.

        This method:
        1. Compiles away conditional effects
        2. Solves the compiled problem
        3. Converts the plan back to the original problem if needed

        Returns:
            Tuple of (success, formatted_plan_string, plan_object)
        """
        # Compile away conditional effects
        compiled_problem, compiler, compilation_result, action_mapping = compile_conditional_effects(problem)
        
        # Let UP automatically select the best planner for the problem kind
        up.shortcuts.get_environment().credits_stream = None
        
        # Generic planning approach for any problem type
        with OneshotPlanner(problem_kind=compiled_problem.kind) as planner:
            # print(f"\nUsing planner: {planner.name}")
            print()
            result = planner.solve(compiled_problem)
            
            if result.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
                # Convert the plan back to the original problem if compilation was used
                original_plan = convert_compiled_plan_to_original(result.plan, compilation_result)
                formatted_plan = self._format_plan(original_plan)
                return True, formatted_plan, original_plan
            return False, "", None

    def _format_plan(self, plan):
        """Format the plan into a string according to standard formats:
        - SequentialPlans: (action-name param1 param2 ... paramN)
        - TimeTriggeredPlans: start-time: (action-name param1 param2 ... paramN) [duration]
        - PartialOrderPlans: Convert to first sequential plan
        """
        if plan is None:
            return "No plan found"
        
        # Handle PartialOrderPlan (convert to first sequential plan)
        if hasattr(plan, 'all_sequential_plans'):
            sequential_plans = list(plan.all_sequential_plans())
            if sequential_plans:
                plan = sequential_plans[0]  # Use the first sequential plan
            else:
                return "No sequential plans available from partial order plan"
        
        # Handle TimeTriggeredPlan
        if hasattr(plan, 'timed_actions'):
            formatted_actions = []
            for start, action, duration in plan.timed_actions:
                # Extract action name and parameters
                action_name = action.action.name
                params = [str(param) for param in action.actual_parameters]
                action_str = f"{action_name} {' '.join(params)}".strip()
                # Format: start-time: (action-name param1 param2 ... paramN) [duration]
                formatted_actions.append(f"{float(start)}: ({action_str}) [{float(duration)}]")
            return "\n".join(formatted_actions)
        
        # Handle SequentialPlan and other regular plans
        if hasattr(plan, 'actions'):
            formatted_actions = []
            for action in plan.actions:
                # Extract action name and parameters
                action_name = action.action.name
                params = [str(param) for param in action.actual_parameters]
                action_str = action_name

                if params:
                    action_str += f" {' '.join(params)}"

                # Format: (action param1 param2 ... paramN)
                formatted_actions.append(f"({action_str})")
            return "\n".join(formatted_actions)
        
        # Fallback for unknown plan types
        return str(plan)               

    def export_pddl(self, problem: Problem, output_dir: str) -> Tuple[str, str]:
        """Export the problem to PDDL files.

        Returns:
            Tuple of (domain_filename, problem_filename)
        """
        writer = PDDLWriter(problem)
        
        domain_filename = os.path.join(output_dir, "domain.pddl")
        problem_filename = os.path.join(output_dir, "problem.pddl")
        
        writer.write_domain(domain_filename)
        writer.write_problem(problem_filename)
        
        return domain_filename, problem_filename

    def export_anml(self, problem: Problem, output_dir: str) -> str:
        """Export the problem to ANML file.
        
        Returns:
            anml_filename
        """
        os.makedirs(output_dir, exist_ok=True)
        
        anml_filename = os.path.join(output_dir, "problem.anml")
        writer = ANMLWriter(problem)
        writer.write_problem(anml_filename)
        
        return anml_filename 