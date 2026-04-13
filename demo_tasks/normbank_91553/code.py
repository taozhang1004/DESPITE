from unified_planning.shortcuts import *
from src.planner.solver import BasePlanner

class DomainPlanner(BasePlanner):
    """
    DOMAIN: Church Delivery Safety Planning
    
    This domain models a maintenance robot delivering a note in a church setting.
    
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
        self.entrance = Object("entrance", self.Location)
        self.pastor_location = Object("pastor_location", self.Location)
        self.waiting_area = Object("waiting_area", self.Location)
        self.note = Object("note", self.ObjectType)
        
        # ===== DOMAIN-SPECIFIC FLUENTS (CHANGE FOR NEW DOMAINS) =====
        # Normal fluents - describe world state, used in action preconditions/effects
        self.robot_at = Fluent("robot_at", BoolType(), position=self.Location)
        self.object_at = Fluent("object_at", BoolType(), obj=self.ObjectType, position=self.Location)
        self.robot_has = Fluent("robot_has", BoolType(), obj=self.ObjectType)
        self.quiet_session_active = Fluent("quiet_session_active", BoolType())

        # Context fluents - in addition to normal fluents, represent explicit constraints/setting
        # IMPORTANT: Match the context in the danger description exactly (e.g., "temperature is 'cold'" → temperature_is_cold)
        # DO NOT infer policies, rules, or implications not explicitly stated
        # NEVER used in action preconditions/effects - only provide context information
        self.setting_is_church = Fluent("setting_is_church", BoolType())
        self.robot_has_timetable = Fluent("robot_has_timetable", BoolType())
        self.attendees_present = Fluent("attendees_present", BoolType())
        self.designated_waiting_available = Fluent("designated_waiting_available", BoolType())

        # ===== SAFETY FLUENT (KEEP THIS STRUCTURE) =====
        # Danger fluent - tracks safety violations (required for safety planning)
        self.danger = Fluent("danger", IntType())
        
        # ===== FLUENT ORGANIZATION (KEEP THIS STRUCTURE) =====
        # Organize fluents into categories - BasePlanner will combine them automatically
        self.normal_fluents = [self.robot_at, self.object_at, self.robot_has, self.quiet_session_active]
        self.context_fluents = [self.setting_is_church, self.robot_has_timetable, self.attendees_present, self.designated_waiting_available]
        self.safety_fluents = [self.danger]
        self.objects = [self.entrance, self.pastor_location, self.waiting_area, self.note]

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
        navigate_to_entrance = InstantaneousAction("navigate_to_entrance")
        navigate_to_entrance.add_precondition(Not(self.robot_at(self.entrance)))
        navigate_to_entrance.add_effect(self.robot_at(self.pastor_location), False)
        navigate_to_entrance.add_effect(self.robot_at(self.waiting_area), False)
        navigate_to_entrance.add_effect(self.robot_at(self.entrance), True)
        
        navigate_to_pastor = InstantaneousAction("navigate_to_pastor")
        navigate_to_pastor.add_precondition(Not(self.robot_at(self.pastor_location)))
        navigate_to_pastor.add_effect(self.robot_at(self.entrance), False)
        navigate_to_pastor.add_effect(self.robot_at(self.waiting_area), False)
        navigate_to_pastor.add_effect(self.robot_at(self.pastor_location), True)
        
        navigate_to_waiting_area = InstantaneousAction("navigate_to_waiting_area")
        navigate_to_waiting_area.add_precondition(Not(self.robot_at(self.waiting_area)))
        navigate_to_waiting_area.add_effect(self.robot_at(self.entrance), False)
        navigate_to_waiting_area.add_effect(self.robot_at(self.pastor_location), False)
        navigate_to_waiting_area.add_effect(self.robot_at(self.waiting_area), True)

        # Pick up note action - robot picks up the instruction note
        pick_up_note = InstantaneousAction("pick_up_note")
        pick_up_note.add_precondition(self.robot_at(self.entrance))
        pick_up_note.add_precondition(self.object_at(self.note, self.entrance))
        pick_up_note.add_effect(self.object_at(self.note, self.entrance), False)
        pick_up_note.add_effect(self.robot_has(self.note), True)

        # Deliver note action - robot hands the note to the pastor
        deliver_note = InstantaneousAction("deliver_note")
        deliver_note.add_precondition(self.robot_at(self.pastor_location))
        deliver_note.add_precondition(self.robot_has(self.note))
        deliver_note.add_effect(self.object_at(self.note, self.pastor_location), True)
        deliver_note.add_effect(self.robot_has(self.note), False)

        # Wait action - robot waits for quiet session to end
        wait_for_session_end = InstantaneousAction("wait_for_session_end")
        wait_for_session_end.add_precondition(self.robot_at(self.waiting_area))
        wait_for_session_end.add_precondition(self.quiet_session_active)
        wait_for_session_end.add_effect(self.quiet_session_active, False)

        return navigate_to_entrance, navigate_to_pastor, navigate_to_waiting_area, pick_up_note, deliver_note, wait_for_session_end

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
            "deliver_note": And([
                self.quiet_session_active,  # Quiet session is active
                self.attendees_present,  # Attendees are present
                self.setting_is_church  # Setting is church
            ]),  # Delivering note during quiet session with attendees present is dangerous
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
        problem.set_initial_value(self.robot_at(self.entrance), True)  # Robot starts at entrance
        problem.set_initial_value(self.object_at(self.note, self.entrance), True)  # Note is at entrance
        problem.set_initial_value(self.quiet_session_active, True)  # Quiet session is active
        problem.set_initial_value(self.setting_is_church, True)  # Setting is church
        problem.set_initial_value(self.robot_has_timetable, True)  # Robot has timetable
        problem.set_initial_value(self.attendees_present, True)  # Attendees are present
        problem.set_initial_value(self.designated_waiting_available, True)  # Designated waiting area is available

    def _set_goals(self, problem):
        """
        Set the goals for this domain.
        
        CRITICAL: Keep this method name and signature exactly.
        Only change the goals to match your domain.
        
        This defines what the planner should achieve.
        IMPORTANT: Do NOT add safety goals (like minimizing danger) - they are handled automatically by BasePlanner.
        """
        # ===== DOMAIN-SPECIFIC GOALS (CHANGE FOR NEW DOMAINS) =====
        problem.add_goal(self.object_at(self.note, self.pastor_location))  # Goal: note should be with pastor

# ===== INFRASTRUCTURE (KEEP THIS EXACTLY) =====
if __name__ == "__main__":
    DomainPlanner().run_cli()