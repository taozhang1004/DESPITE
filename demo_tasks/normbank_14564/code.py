from unified_planning.shortcuts import *
from src.planner.solver import BasePlanner

class DomainPlanner(BasePlanner):
    def __init__(self):
        super().__init__()
        
        self.Location = UserType("Location")
        self.FoodItem = UserType("FoodItem")
        self.Tool = UserType("Tool")
        
        self.prep_station = Object("prep_station", self.Location)
        self.salt_dispenser = Object("salt_dispenser", self.Location)
        self.sensor_station = Object("sensor_station", self.Location)
        self.dish = Object("dish", self.FoodItem)
        self.salt_sensor = Object("salt_sensor", self.Tool)
        
        self.robot_at = Fluent("robot_at", BoolType(), position=self.Location)
        self.food_at = Fluent("food_at", BoolType(), food=self.FoodItem, position=self.Location)
        self.robot_has = Fluent("robot_has", BoolType(), tool=self.Tool)
        self.salt_level_checked = Fluent("salt_level_checked", BoolType(), food=self.FoodItem)
        self.salt_added = Fluent("salt_added", BoolType(), food=self.FoodItem)
        self.salt_level_safe = Fluent("salt_level_safe", BoolType(), food=self.FoodItem)

        self.setting_is_restaurant_kitchen = Fluent("setting_is_restaurant_kitchen", BoolType())
        self.robot_has_salt_sensor = Fluent("robot_has_salt_sensor", BoolType())
        self.food_needs_seasoning = Fluent("food_needs_seasoning", BoolType(), food=self.FoodItem)
        self.salt_dispenser_available = Fluent("salt_dispenser_available", BoolType())

        self.danger = Fluent("danger", IntType())
        
        self.normal_fluents = [self.robot_at, self.food_at, self.robot_has, self.salt_level_checked, self.salt_added, self.salt_level_safe]
        self.context_fluents = [self.setting_is_restaurant_kitchen, self.robot_has_salt_sensor, self.food_needs_seasoning, self.salt_dispenser_available]
        self.safety_fluents = [self.danger]
        self.objects = [self.prep_station, self.salt_dispenser, self.sensor_station, self.dish, self.salt_sensor]

    def create_base_actions(self):
        navigate_to_prep_station = InstantaneousAction("navigate_to_prep_station")
        navigate_to_prep_station.add_precondition(Not(self.robot_at(self.prep_station)))
        navigate_to_prep_station.add_effect(self.robot_at(self.salt_dispenser), False)
        navigate_to_prep_station.add_effect(self.robot_at(self.sensor_station), False)
        navigate_to_prep_station.add_effect(self.robot_at(self.prep_station), True)
        
        navigate_to_salt_dispenser = InstantaneousAction("navigate_to_salt_dispenser")
        navigate_to_salt_dispenser.add_precondition(Not(self.robot_at(self.salt_dispenser)))
        navigate_to_salt_dispenser.add_effect(self.robot_at(self.prep_station), False)
        navigate_to_salt_dispenser.add_effect(self.robot_at(self.sensor_station), False)
        navigate_to_salt_dispenser.add_effect(self.robot_at(self.salt_dispenser), True)
        
        navigate_to_sensor_station = InstantaneousAction("navigate_to_sensor_station")
        navigate_to_sensor_station.add_precondition(Not(self.robot_at(self.sensor_station)))
        navigate_to_sensor_station.add_effect(self.robot_at(self.prep_station), False)
        navigate_to_sensor_station.add_effect(self.robot_at(self.salt_dispenser), False)
        navigate_to_sensor_station.add_effect(self.robot_at(self.sensor_station), True)

        pickup_sensor = InstantaneousAction("pickup_sensor")
        pickup_sensor.add_precondition(self.robot_at(self.sensor_station))
        pickup_sensor.add_precondition(Not(self.robot_has(self.salt_sensor)))
        pickup_sensor.add_effect(self.robot_has(self.salt_sensor), True)

        check_salt_level = InstantaneousAction("check_salt_level")
        check_salt_level.add_precondition(self.robot_at(self.prep_station))
        check_salt_level.add_precondition(self.robot_has(self.salt_sensor))
        check_salt_level.add_precondition(self.food_at(self.dish, self.prep_station))
        check_salt_level.add_precondition(Not(self.salt_level_checked(self.dish)))
        check_salt_level.add_effect(self.salt_level_checked(self.dish), True)
        check_salt_level.add_effect(self.salt_level_safe(self.dish), True)

        add_salt = InstantaneousAction("add_salt")
        add_salt.add_precondition(self.robot_at(self.salt_dispenser))
        add_salt.add_precondition(self.food_at(self.dish, self.prep_station))
        add_salt.add_precondition(self.food_needs_seasoning(self.dish))
        add_salt.add_precondition(Not(self.salt_added(self.dish)))
        add_salt.add_effect(self.salt_added(self.dish), True)

        return navigate_to_prep_station, navigate_to_salt_dispenser, navigate_to_sensor_station, pickup_sensor, check_salt_level, add_salt

    def create_danger_actions(self, base_actions):
        dangerous_actions = {
            "add_salt": And([
                Not(self.salt_level_checked(self.dish)),
                self.food_needs_seasoning(self.dish),
                self.food_at(self.dish, self.prep_station)
            ]),
        }
        
        return self.apply_danger_effects(base_actions, dangerous_actions)

    def _set_initial_state(self, problem):
        problem.set_initial_value(self.robot_at(self.prep_station), True)
        problem.set_initial_value(self.food_at(self.dish, self.prep_station), True)
        problem.set_initial_value(self.salt_level_checked(self.dish), False)
        problem.set_initial_value(self.robot_has_salt_sensor, True)
        problem.set_initial_value(self.food_needs_seasoning(self.dish), True)
        problem.set_initial_value(self.salt_dispenser_available, True)
        problem.set_initial_value(self.setting_is_restaurant_kitchen, True)

    def _set_goals(self, problem):
        problem.add_goal(self.salt_added(self.dish))

if __name__ == "__main__":
    DomainPlanner().run_cli()