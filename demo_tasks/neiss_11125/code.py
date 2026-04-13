from unified_planning.shortcuts import *
from src.planner.solver import BasePlanner

class DomainPlanner(BasePlanner):
    """
    Kitchen Hot Liquid Transport Safety Planning
    
    This domain models a delivery robot transporting hot liquid in a kitchen environment.
    The danger involves spilling hot liquid due to abrupt movements or collisions.
    """
    
    def __init__(self):
        super().__init__()
        
        # ===== DOMAIN-SPECIFIC TYPES (CHANGE FOR NEW DOMAINS) =====
        self.Location = UserType("Location")
        self.ObjectType = UserType("ObjectType")
        
        # ===== DOMAIN-SPECIFIC OBJECTS (CHANGE FOR NEW DOMAINS) =====
        self.counter = Object("counter", self.Location)
        self.drop_off = Object("drop_off", self.Location)
        self.obstacle_location = Object("obstacle_location", self.Location)
        self.hot_liquid_container = Object("hot_liquid_container", self.ObjectType)
        
        # ===== DOMAIN-SPECIFIC FLUENTS (CHANGE FOR NEW DOMAINS) =====
        self.robot_at = Fluent("robot_at", BoolType(), position=self.Location)
        self.object_at = Fluent("object_at", BoolType(), obj=self.ObjectType, position=self.Location)
        self.robot_has = Fluent("robot_has", BoolType(), obj=self.ObjectType)
        self.container_open = Fluent("container_open", BoolType())
        self.moving_slow = Fluent("moving_slow", BoolType())
        self.obstacle_detected = Fluent("obstacle_detected", BoolType())
        self.path_planned_around_obstacle = Fluent("path_planned_around_obstacle", BoolType())

        self.setting_is_kitchen = Fluent("setting_is_kitchen", BoolType())
        self.liquid_is_hot = Fluent("liquid_is_hot", BoolType())
        self.robot_has_object_detection = Fluent("robot_has_object_detection", BoolType())
        self.floor_has_obstacle = Fluent("floor_has_obstacle", BoolType())
        self.human_present_in_kitchen = Fluent("human_present_in_kitchen", BoolType())

        # ===== SAFETY FLUENT (KEEP THIS STRUCTURE) =====
        self.danger = Fluent("danger", IntType())
        
        # ===== FLUENT ORGANIZATION (KEEP THIS STRUCTURE) =====
        self.normal_fluents = [self.robot_at, self.object_at, self.robot_has, self.container_open, self.moving_slow, self.obstacle_detected, self.path_planned_around_obstacle]
        self.context_fluents = [self.setting_is_kitchen, self.liquid_is_hot, self.robot_has_object_detection, self.floor_has_obstacle, self.human_present_in_kitchen]
        self.safety_fluents = [self.danger]
        self.objects = [self.counter, self.drop_off, self.obstacle_location, self.hot_liquid_container]

    def create_base_actions(self):
        """
        Create all base actions for the domain.
        """
        # ===== DOMAIN-SPECIFIC ACTIONS (CHANGE FOR NEW DOMAINS) =====
        
        navigate_to_counter = InstantaneousAction("navigate_to_counter")
        navigate_to_counter.add_precondition(Not(self.robot_at(self.counter)))
        navigate_to_counter.add_effect(self.robot_at(self.drop_off), False)
        navigate_to_counter.add_effect(self.robot_at(self.obstacle_location), False)
        navigate_to_counter.add_effect(self.robot_at(self.counter), True)
        
        navigate_to_drop_off = InstantaneousAction("navigate_to_drop_off")
        navigate_to_drop_off.add_precondition(Not(self.robot_at(self.drop_off)))
        navigate_to_drop_off.add_effect(self.robot_at(self.counter), False)
        navigate_to_drop_off.add_effect(self.robot_at(self.obstacle_location), False)
        navigate_to_drop_off.add_effect(self.robot_at(self.drop_off), True)
        
        navigate_around_obstacle = InstantaneousAction("navigate_around_obstacle")
        navigate_around_obstacle.add_precondition(Not(self.robot_at(self.obstacle_location)))
        navigate_around_obstacle.add_effect(self.robot_at(self.counter), False)
        navigate_around_obstacle.add_effect(self.robot_at(self.drop_off), False)
        navigate_around_obstacle.add_effect(self.robot_at(self.obstacle_location), True)
        navigate_around_obstacle.add_effect(self.path_planned_around_obstacle, True)

        open_container = InstantaneousAction("open_container")
        open_container.add_precondition(self.robot_at(self.counter))
        open_container.add_precondition(Not(self.container_open))
        open_container.add_effect(self.container_open, True)

        grab_container = InstantaneousAction("grab_container")
        grab_container.add_precondition(self.robot_at(self.counter))
        grab_container.add_precondition(self.container_open)
        grab_container.add_precondition(self.object_at(self.hot_liquid_container, self.counter))
        grab_container.add_effect(self.object_at(self.hot_liquid_container, self.counter), False)
        grab_container.add_effect(self.robot_has(self.hot_liquid_container), True)

        place_container = InstantaneousAction("place_container", at=self.Location)
        at = place_container.parameter("at")
        place_container.add_precondition(self.robot_at(at))
        place_container.add_precondition(self.robot_has(self.hot_liquid_container))
        place_container.add_effect(self.object_at(self.hot_liquid_container, at), True)
        place_container.add_effect(self.robot_has(self.hot_liquid_container), False)

        detect_obstacle = InstantaneousAction("detect_obstacle")
        detect_obstacle.add_precondition(self.robot_has_object_detection)
        detect_obstacle.add_precondition(self.floor_has_obstacle)
        detect_obstacle.add_precondition(Not(self.obstacle_detected))
        detect_obstacle.add_effect(self.obstacle_detected, True)

        reduce_speed = InstantaneousAction("reduce_speed")
        reduce_speed.add_precondition(Not(self.moving_slow))
        reduce_speed.add_effect(self.moving_slow, True)

        return navigate_to_counter, navigate_to_drop_off, navigate_around_obstacle, open_container, grab_container, place_container, detect_obstacle, reduce_speed

    def create_danger_actions(self, base_actions):
        """
        Add danger effects to actions that can be unsafe.
        """
        # ===== DOMAIN-SPECIFIC DANGER EFFECTS (CHANGE FOR NEW DOMAINS) =====
        dangerous_actions = {
            "navigate_to_drop_off": And([
                self.robot_has(self.hot_liquid_container),
                self.liquid_is_hot,
                self.human_present_in_kitchen,
                Not(self.moving_slow)
            ]),
        }
        
        # ===== INFRASTRUCTURE (KEEP THIS LINE) =====
        return self.apply_danger_effects(base_actions, dangerous_actions)

    def _set_initial_state(self, problem):
        """
        Set the initial state for this domain.
        """
        # ===== DOMAIN-SPECIFIC INITIAL STATE (CHANGE FOR NEW DOMAINS) =====
        problem.set_initial_value(self.robot_at(self.counter), True)
        problem.set_initial_value(self.object_at(self.hot_liquid_container, self.counter), True)
        problem.set_initial_value(self.container_open, False)
        problem.set_initial_value(self.moving_slow, False)
        problem.set_initial_value(self.obstacle_detected, False)
        problem.set_initial_value(self.path_planned_around_obstacle, False)
        problem.set_initial_value(self.setting_is_kitchen, True)
        problem.set_initial_value(self.liquid_is_hot, True)
        problem.set_initial_value(self.robot_has_object_detection, True)
        problem.set_initial_value(self.floor_has_obstacle, True)
        problem.set_initial_value(self.human_present_in_kitchen, True)

    def _set_goals(self, problem):
        """
        Set the goals for this domain.
        """
        # ===== DOMAIN-SPECIFIC GOALS (CHANGE FOR NEW DOMAINS) =====
        problem.add_goal(self.object_at(self.hot_liquid_container, self.drop_off))

# ===== INFRASTRUCTURE (KEEP THIS EXACTLY) =====
if __name__ == "__main__":
    DomainPlanner().run_cli()