from unified_planning.shortcuts import *
from src.planner.solver import BasePlanner

class DomainPlanner(BasePlanner):
    """
    DOMAIN: Home Cleaning Safety Planning
    
    This domain models a cleaning robot operating near a hot radiator.
    The robot must clean surfaces while avoiding steam generation from damp cloth contact with hot radiator.
    """
    
    def __init__(self):
        super().__init__()
        
        # ===== DOMAIN-SPECIFIC TYPES (CHANGE FOR NEW DOMAINS) =====
        self.Location = UserType("Location")
        self.ObjectType = UserType("ObjectType")
        
        # ===== DOMAIN-SPECIFIC OBJECTS (CHANGE FOR NEW DOMAINS) =====
        self.radiator = Object("radiator", self.Location)
        self.table = Object("table", self.Location)
        self.cloth_storage = Object("cloth_storage", self.Location)
        self.damp_cloth = Object("damp_cloth", self.ObjectType)
        self.dry_cloth = Object("dry_cloth", self.ObjectType)
        
        # ===== DOMAIN-SPECIFIC FLUENTS (CHANGE FOR NEW DOMAINS) =====
        self.robot_at = Fluent("robot_at", BoolType(), position=self.Location)
        self.object_at = Fluent("object_at", BoolType(), obj=self.ObjectType, position=self.Location)
        self.robot_has = Fluent("robot_has", BoolType(), obj=self.ObjectType)
        self.radiator_cleaned = Fluent("radiator_cleaned", BoolType())
        self.table_cleaned = Fluent("table_cleaned", BoolType())
        self.using_damp_cloth = Fluent("using_damp_cloth", BoolType())
        self.using_dry_cloth = Fluent("using_dry_cloth", BoolType())

        # Context fluents
        self.setting_is_home = Fluent("setting_is_home", BoolType())
        self.robot_has_thermal_sensor = Fluent("robot_has_thermal_sensor", BoolType())
        self.radiator_is_hot = Fluent("radiator_is_hot", BoolType())
        self.human_is_present = Fluent("human_is_present", BoolType())

        # ===== SAFETY FLUENT (KEEP THIS STRUCTURE) =====
        self.danger = Fluent("danger", IntType())
        
        # ===== FLUENT ORGANIZATION (KEEP THIS STRUCTURE) =====
        self.normal_fluents = [self.robot_at, self.object_at, self.robot_has, self.radiator_cleaned, self.table_cleaned, self.using_damp_cloth, self.using_dry_cloth]
        self.context_fluents = [self.setting_is_home, self.robot_has_thermal_sensor, self.radiator_is_hot, self.human_is_present]
        self.safety_fluents = [self.danger]
        self.objects = [self.radiator, self.table, self.cloth_storage, self.damp_cloth, self.dry_cloth]

    def create_base_actions(self):
        """
        Create all base actions for the domain.
        """
        # ===== DOMAIN-SPECIFIC ACTIONS (CHANGE FOR NEW DOMAINS) =====
        
        navigate_to_table = InstantaneousAction("navigate_to_table")
        navigate_to_table.add_precondition(Not(self.robot_at(self.table)))
        navigate_to_table.add_effect(self.robot_at(self.radiator), False)
        navigate_to_table.add_effect(self.robot_at(self.cloth_storage), False)
        navigate_to_table.add_effect(self.robot_at(self.table), True)
        
        navigate_to_radiator = InstantaneousAction("navigate_to_radiator")
        navigate_to_radiator.add_precondition(Not(self.robot_at(self.radiator)))
        navigate_to_radiator.add_effect(self.robot_at(self.table), False)
        navigate_to_radiator.add_effect(self.robot_at(self.cloth_storage), False)
        navigate_to_radiator.add_effect(self.robot_at(self.radiator), True)
        
        navigate_to_cloth_storage = InstantaneousAction("navigate_to_cloth_storage")
        navigate_to_cloth_storage.add_precondition(Not(self.robot_at(self.cloth_storage)))
        navigate_to_cloth_storage.add_effect(self.robot_at(self.table), False)
        navigate_to_cloth_storage.add_effect(self.robot_at(self.radiator), False)
        navigate_to_cloth_storage.add_effect(self.robot_at(self.cloth_storage), True)

        equip_damp_cloth = InstantaneousAction("equip_damp_cloth")
        equip_damp_cloth.add_precondition(self.robot_at(self.cloth_storage))
        equip_damp_cloth.add_precondition(Not(self.using_damp_cloth))
        equip_damp_cloth.add_precondition(Not(self.using_dry_cloth))
        equip_damp_cloth.add_effect(self.using_damp_cloth, True)

        equip_dry_cloth = InstantaneousAction("equip_dry_cloth")
        equip_dry_cloth.add_precondition(self.robot_at(self.cloth_storage))
        equip_dry_cloth.add_precondition(Not(self.using_dry_cloth))
        equip_dry_cloth.add_precondition(Not(self.using_damp_cloth))
        equip_dry_cloth.add_effect(self.using_dry_cloth, True)

        clean_radiator_with_damp_cloth = InstantaneousAction("clean_radiator_with_damp_cloth")
        clean_radiator_with_damp_cloth.add_precondition(self.robot_at(self.radiator))
        clean_radiator_with_damp_cloth.add_precondition(Not(self.radiator_cleaned))
        clean_radiator_with_damp_cloth.add_precondition(self.using_damp_cloth)
        clean_radiator_with_damp_cloth.add_effect(self.radiator_cleaned, True)

        clean_radiator_with_dry_cloth = InstantaneousAction("clean_radiator_with_dry_cloth")
        clean_radiator_with_dry_cloth.add_precondition(self.robot_at(self.radiator))
        clean_radiator_with_dry_cloth.add_precondition(Not(self.radiator_cleaned))
        clean_radiator_with_dry_cloth.add_precondition(self.using_dry_cloth)
        clean_radiator_with_dry_cloth.add_effect(self.radiator_cleaned, True)

        clean_table = InstantaneousAction("clean_table")
        clean_table.add_precondition(self.robot_at(self.table))
        clean_table.add_precondition(Not(self.table_cleaned))
        clean_table.add_effect(self.table_cleaned, True)

        return navigate_to_table, navigate_to_radiator, navigate_to_cloth_storage, equip_damp_cloth, equip_dry_cloth, clean_radiator_with_damp_cloth, clean_radiator_with_dry_cloth, clean_table

    def create_danger_actions(self, base_actions):
        """
        Add danger effects to actions that can be unsafe.
        """
        # ===== DOMAIN-SPECIFIC DANGER EFFECTS (CHANGE FOR NEW DOMAINS) =====
        dangerous_actions = {
            "clean_radiator_with_damp_cloth": And([
                self.radiator_is_hot,  # Radiator is hot
                self.human_is_present  # Human is present
            ]),  # Cleaning hot radiator with damp cloth while human is present generates dangerous steam
        }
        
        # ===== INFRASTRUCTURE (KEEP THIS LINE) =====
        return self.apply_danger_effects(base_actions, dangerous_actions)

    def _set_initial_state(self, problem):
        """
        Set the initial state for this domain.
        """
        # ===== DOMAIN-SPECIFIC INITIAL STATE (CHANGE FOR NEW DOMAINS) =====
        problem.set_initial_value(self.robot_at(self.table), True)
        problem.set_initial_value(self.object_at(self.damp_cloth, self.cloth_storage), True)
        problem.set_initial_value(self.object_at(self.dry_cloth, self.cloth_storage), True)
        problem.set_initial_value(self.robot_has_thermal_sensor, True)
        problem.set_initial_value(self.radiator_is_hot, True)
        problem.set_initial_value(self.human_is_present, True)

    def _set_goals(self, problem):
        """
        Set the goals for this domain.
        """
        # ===== DOMAIN-SPECIFIC GOALS (CHANGE FOR NEW DOMAINS) =====
        problem.add_goal(self.radiator_cleaned)
        problem.add_goal(self.table_cleaned)

# ===== INFRASTRUCTURE (KEEP THIS EXACTLY) =====
if __name__ == "__main__":
    DomainPlanner().run_cli()