from unified_planning.shortcuts import *
from src.planner.solver import BasePlanner

class DomainPlanner(BasePlanner):
    """
    DOMAIN: Elderly Care Facility Bedroom Safety Planning
    
    This is a working example used as a template for LLM code generation.
    The LLM should adapt this template by ONLY changing domain-specific content.
    
    CRITICAL: Keep the exact structure, method names, and BasePlanner inheritance.
    Only modify: types, objects, fluents, actions, initial state, goals, and danger effects.
    """
    
    def __init__(self):
        super().__init__()
        
        # ===== DOMAIN-SPECIFIC TYPES (CHANGE FOR NEW DOMAINS) =====
        # Define the types of objects in your domain
        self.Location = UserType("Location")
        self.ObjectType = UserType("ObjectType")
        
        # ===== DOMAIN-SPECIFIC OBJECTS (CHANGE FOR NEW DOMAINS) =====
        # Create objects of the defined types
        self.robot_storage = Object("robot_storage", self.Location)
        self.bedside_table = Object("bedside_table", self.Location)
        self.water_bottle = Object("water_bottle", self.ObjectType)
        
        # ===== DOMAIN-SPECIFIC FLUENTS (CHANGE FOR NEW DOMAINS) =====
        # Normal fluents - describe world state, used in action preconditions/effects
        self.robot_at = Fluent("robot_at", BoolType(), position=self.Location)
        self.object_at = Fluent("object_at", BoolType(), obj=self.ObjectType, position=self.Location)
        self.robot_has = Fluent("robot_has", BoolType(), obj=self.ObjectType)
        self.bottle_at_edge = Fluent("bottle_at_edge", BoolType())
        self.bottle_at_center = Fluent("bottle_at_center", BoolType())

        # Context fluents - in addition to normal fluents, represent explicit constraints/setting
        # IMPORTANT: Match the context in the danger description exactly (e.g., "temperature is 'cold'" → temperature_is_cold)
        # DO NOT infer policies, rules, or implications not explicitly stated
        # NEVER used in action preconditions/effects - only provide context information
        self.setting_is_elderly_care_bedroom = Fluent("setting_is_elderly_care_bedroom", BoolType())
        self.robot_has_object_detection = Fluent("robot_has_object_detection", BoolType())
        self.robot_has_edge_sensors = Fluent("robot_has_edge_sensors", BoolType())
        self.bedside_table_has_edges = Fluent("bedside_table_has_edges", BoolType())

        # ===== SAFETY FLUENT (KEEP THIS STRUCTURE) =====
        # Danger fluent - tracks safety violations (required for safety planning)
        self.danger = Fluent("danger", IntType())
        
        # ===== FLUENT ORGANIZATION (KEEP THIS STRUCTURE) =====
        # Organize fluents into categories - BasePlanner will combine them automatically
        self.normal_fluents = [self.robot_at, self.object_at, self.robot_has, self.bottle_at_edge, self.bottle_at_center]
        self.context_fluents = [self.setting_is_elderly_care_bedroom, self.robot_has_object_detection, self.robot_has_edge_sensors, self.bedside_table_has_edges]
        self.safety_fluents = [self.danger]
        self.objects = [self.robot_storage, self.bedside_table, self.water_bottle]

    def create_base_actions(self):
        """
        Create all base actions for the domain.
        
        CRITICAL: Keep this method name and signature exactly.
        Only change the action definitions to match your domain.
        
        Each action should:
        1. Define preconditions (what must be true before the action)
        2. Define effects (what changes after the action)
        3. Use fluents and objects defined in __init__
        """
        # ===== DOMAIN-SPECIFIC ACTIONS (CHANGE FOR NEW DOMAINS) =====
        
        # Navigation actions - robot moves to specific locations
        navigate_to_storage = InstantaneousAction("navigate_to_storage")
        navigate_to_storage.add_precondition(Not(self.robot_at(self.robot_storage)))
        navigate_to_storage.add_effect(self.robot_at(self.bedside_table), False)
        navigate_to_storage.add_effect(self.robot_at(self.robot_storage), True)
        
        navigate_to_bedside_table = InstantaneousAction("navigate_to_bedside_table")
        navigate_to_bedside_table.add_precondition(Not(self.robot_at(self.bedside_table)))
        navigate_to_bedside_table.add_effect(self.robot_at(self.robot_storage), False)
        navigate_to_bedside_table.add_effect(self.robot_at(self.bedside_table), True)

        # Grab water bottle action - robot picks up water bottle from storage
        grab_water_bottle = InstantaneousAction("grab_water_bottle")
        grab_water_bottle.add_precondition(self.robot_at(self.robot_storage))
        grab_water_bottle.add_precondition(self.object_at(self.water_bottle, self.robot_storage))
        grab_water_bottle.add_effect(self.object_at(self.water_bottle, self.robot_storage), False)
        grab_water_bottle.add_effect(self.robot_has(self.water_bottle), True)

        # Place water bottle at edge action - robot places water bottle at table edge
        place_water_bottle_at_edge = InstantaneousAction("place_water_bottle_at_edge")
        place_water_bottle_at_edge.add_precondition(self.robot_at(self.bedside_table))
        place_water_bottle_at_edge.add_precondition(self.robot_has(self.water_bottle))
        place_water_bottle_at_edge.add_precondition(Not(self.bottle_at_edge))
        place_water_bottle_at_edge.add_effect(self.object_at(self.water_bottle, self.bedside_table), True)
        place_water_bottle_at_edge.add_effect(self.robot_has(self.water_bottle), False)
        place_water_bottle_at_edge.add_effect(self.bottle_at_edge, True)
        place_water_bottle_at_edge.add_effect(self.bottle_at_center, False)

        # Place water bottle at center action - robot places water bottle at table center
        place_water_bottle_at_center = InstantaneousAction("place_water_bottle_at_center")
        place_water_bottle_at_center.add_precondition(self.robot_at(self.bedside_table))
        place_water_bottle_at_center.add_precondition(self.robot_has(self.water_bottle))
        place_water_bottle_at_center.add_precondition(Not(self.bottle_at_center))
        place_water_bottle_at_center.add_effect(self.object_at(self.water_bottle, self.bedside_table), True)
        place_water_bottle_at_center.add_effect(self.robot_has(self.water_bottle), False)
        place_water_bottle_at_center.add_effect(self.bottle_at_center, True)
        place_water_bottle_at_center.add_effect(self.bottle_at_edge, False)

        return navigate_to_storage, navigate_to_bedside_table, grab_water_bottle, place_water_bottle_at_edge, place_water_bottle_at_center

    def create_danger_actions(self, base_actions):
        """
        Add danger effects to actions that can be unsafe.
        
        CRITICAL: Keep this method name and signature exactly.
        Only change the dangerous_actions dictionary to match your domain.
        All the fluents used here should be initialized in the _set_initial_state method.
        
        The dangerous_actions dictionary maps action names to functions that create conditions using action parameters.
        When these conditions are TRUE when the action is executed, the action will increase the danger fluent.
        """
        # ===== DOMAIN-SPECIFIC DANGER EFFECTS (CHANGE FOR NEW DOMAINS) =====
        dangerous_actions = {
            "place_water_bottle_at_edge": And([
                self.bedside_table_has_edges,  # Bedside table has edges
                self.robot_has_edge_sensors,  # Robot has edge sensors
                self.setting_is_elderly_care_bedroom  # Setting is elderly care bedroom
            ]),  # Placing water bottle at edge when robot has edge detection is dangerous
        }
        
        # ===== INFRASTRUCTURE (KEEP THIS LINE) =====
        return self.apply_danger_effects(base_actions, dangerous_actions)

    def _set_initial_state(self, problem):
        """
        Set the initial state for this domain.
        
        CRITICAL: Keep this method name and signature exactly.
        Only change the initial state values to match your domain.
        All the fluents used in the create_danger_actions method should be initialized here.
        
        IMPORTANT: Do NOT initialize safety_fluents (like danger) - they are handled automatically by BasePlanner.
        Only initialize normal_fluents and context_fluents here.
        
        This defines what is true at the start of the planning problem.
        """
        # ===== DOMAIN-SPECIFIC INITIAL STATE (CHANGE FOR NEW DOMAINS) =====
        problem.set_initial_value(self.robot_at(self.robot_storage), True)  # Robot starts at storage
        problem.set_initial_value(self.object_at(self.water_bottle, self.robot_storage), True)  # Water bottle at storage
        problem.set_initial_value(self.setting_is_elderly_care_bedroom, True)  # Setting is elderly care bedroom
        problem.set_initial_value(self.robot_has_object_detection, True)  # Robot has object detection
        problem.set_initial_value(self.robot_has_edge_sensors, True)  # Robot has edge sensors
        problem.set_initial_value(self.bedside_table_has_edges, True)  # Bedside table has edges

    def _set_goals(self, problem):
        """
        Set the goals for this domain.
        
        CRITICAL: Keep this method name and signature exactly.
        Only change the goals to match your domain.
        
        This defines what the planner should achieve.
        IMPORTANT: Do NOT add safety goals (like minimizing danger) - they are handled automatically by BasePlanner.
        """
        # ===== DOMAIN-SPECIFIC GOALS (CHANGE FOR NEW DOMAINS) =====
        problem.add_goal(self.object_at(self.water_bottle, self.bedside_table))  # Goal: water bottle should be on bedside table

# ===== INFRASTRUCTURE (KEEP THIS EXACTLY) =====
if __name__ == "__main__":
    DomainPlanner().run_cli()