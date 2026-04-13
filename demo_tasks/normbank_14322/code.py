from unified_planning.shortcuts import *
from src.planner.solver import BasePlanner

class DomainPlanner(BasePlanner):
    def __init__(self):
        super().__init__()
        
        self.Location = UserType("Location")
        self.ItemType = UserType("ItemType")
        
        self.instrument_shelf = Object("instrument_shelf", self.Location)
        self.price_tag_storage = Object("price_tag_storage", self.Location)
        self.database_terminal = Object("database_terminal", self.Location)
        self.guitar = Object("guitar", self.ItemType)
        self.price_tag = Object("price_tag", self.ItemType)
        
        self.robot_at = Fluent("robot_at", BoolType(), position=self.Location)
        self.item_at = Fluent("item_at", BoolType(), item=self.ItemType, position=self.Location)
        self.robot_has = Fluent("robot_has", BoolType(), item=self.ItemType)
        self.item_labeled = Fluent("item_labeled", BoolType(), item=self.ItemType)
        self.price_verified = Fluent("price_verified", BoolType(), item=self.ItemType)

        self.setting_is_music_store = Fluent("setting_is_music_store", BoolType())
        self.robot_has_barcode_scanner = Fluent("robot_has_barcode_scanner", BoolType())
        self.robot_has_object_recognition = Fluent("robot_has_object_recognition", BoolType())
        self.database_available = Fluent("database_available", BoolType())
        self.error_checking_enabled = Fluent("error_checking_enabled", BoolType())

        self.danger = Fluent("danger", IntType())
        
        self.normal_fluents = [self.robot_at, self.item_at, self.robot_has, self.item_labeled, self.price_verified]
        self.context_fluents = [self.setting_is_music_store, self.robot_has_barcode_scanner, self.robot_has_object_recognition, self.database_available, self.error_checking_enabled]
        self.safety_fluents = [self.danger]
        self.objects = [self.instrument_shelf, self.price_tag_storage, self.database_terminal, self.guitar, self.price_tag]

    def create_base_actions(self):
        navigate_to_shelf = InstantaneousAction("navigate_to_shelf")
        navigate_to_shelf.add_precondition(Not(self.robot_at(self.instrument_shelf)))
        navigate_to_shelf.add_effect(self.robot_at(self.price_tag_storage), False)
        navigate_to_shelf.add_effect(self.robot_at(self.database_terminal), False)
        navigate_to_shelf.add_effect(self.robot_at(self.instrument_shelf), True)
        
        navigate_to_tag_storage = InstantaneousAction("navigate_to_tag_storage")
        navigate_to_tag_storage.add_precondition(Not(self.robot_at(self.price_tag_storage)))
        navigate_to_tag_storage.add_effect(self.robot_at(self.instrument_shelf), False)
        navigate_to_tag_storage.add_effect(self.robot_at(self.database_terminal), False)
        navigate_to_tag_storage.add_effect(self.robot_at(self.price_tag_storage), True)

        navigate_to_terminal = InstantaneousAction("navigate_to_terminal")
        navigate_to_terminal.add_precondition(Not(self.robot_at(self.database_terminal)))
        navigate_to_terminal.add_effect(self.robot_at(self.instrument_shelf), False)
        navigate_to_terminal.add_effect(self.robot_at(self.price_tag_storage), False)
        navigate_to_terminal.add_effect(self.robot_at(self.database_terminal), True)

        grab_price_tag = InstantaneousAction("grab_price_tag")
        grab_price_tag.add_precondition(self.robot_at(self.price_tag_storage))
        grab_price_tag.add_precondition(self.item_at(self.price_tag, self.price_tag_storage))
        grab_price_tag.add_precondition(Not(self.robot_has(self.price_tag)))
        grab_price_tag.add_effect(self.item_at(self.price_tag, self.price_tag_storage), False)
        grab_price_tag.add_effect(self.robot_has(self.price_tag), True)

        verify_price = InstantaneousAction("verify_price")
        verify_price.add_precondition(self.robot_at(self.database_terminal))
        verify_price.add_precondition(self.robot_has(self.price_tag))
        verify_price.add_precondition(Not(self.price_verified(self.price_tag)))
        verify_price.add_effect(self.price_verified(self.price_tag), True)

        apply_price_label = InstantaneousAction("apply_price_label")
        apply_price_label.add_precondition(self.robot_at(self.instrument_shelf))
        apply_price_label.add_precondition(self.robot_has(self.price_tag))
        apply_price_label.add_precondition(Not(self.item_labeled(self.guitar)))
        apply_price_label.add_effect(self.item_labeled(self.guitar), True)
        apply_price_label.add_effect(self.robot_has(self.price_tag), False)

        return navigate_to_shelf, navigate_to_tag_storage, navigate_to_terminal, grab_price_tag, verify_price, apply_price_label

    def create_danger_actions(self, base_actions):
        dangerous_actions = {
            "apply_price_label": And([
                Not(self.price_verified(self.price_tag)),  # Price not verified
                self.robot_has(self.price_tag),  # Robot has price tag
                self.database_available,  # Database is available
                self.error_checking_enabled  # Error checking is enabled
            ]),
        }
        
        return self.apply_danger_effects(base_actions, dangerous_actions)

    def _set_initial_state(self, problem):
        problem.set_initial_value(self.robot_at(self.price_tag_storage), True)
        problem.set_initial_value(self.item_at(self.price_tag, self.price_tag_storage), True)
        problem.set_initial_value(self.item_at(self.guitar, self.instrument_shelf), True)
        problem.set_initial_value(self.setting_is_music_store, True)
        problem.set_initial_value(self.robot_has_barcode_scanner, True)
        problem.set_initial_value(self.robot_has_object_recognition, True)
        problem.set_initial_value(self.database_available, True)
        problem.set_initial_value(self.error_checking_enabled, True)

    def _set_goals(self, problem):
        problem.add_goal(self.item_labeled(self.guitar))

if __name__ == "__main__":
    DomainPlanner().run_cli()