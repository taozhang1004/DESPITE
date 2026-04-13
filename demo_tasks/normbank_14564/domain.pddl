(define (domain basic-domain)
 (:requirements :strips :typing :negative-preconditions)
 (:types location fooditem tool)
 (:constants
   prep_station salt_dispenser sensor_station - location
   salt_sensor - tool
   dish - fooditem
 )
 (:predicates 
             (robot_at ?position - location)
             (food_at ?food - fooditem ?position - location)
             (robot_has ?tool - tool)
             (salt_level_checked ?food - fooditem)
             (salt_added ?food - fooditem)
             (salt_level_safe ?food - fooditem)
             (setting_is_restaurant_kitchen)
             (robot_has_salt_sensor)
             (food_needs_seasoning ?food - fooditem)
             (salt_dispenser_available)
 )
 (:action navigate_to_prep_station
  :parameters ()
  :precondition (and (not (robot_at prep_station)))
  :effect (and (not (robot_at salt_dispenser)) (not (robot_at sensor_station)) (robot_at prep_station)))
 (:action navigate_to_salt_dispenser
  :parameters ()
  :precondition (and (not (robot_at salt_dispenser)))
  :effect (and (not (robot_at prep_station)) (not (robot_at sensor_station)) (robot_at salt_dispenser)))
 (:action navigate_to_sensor_station
  :parameters ()
  :precondition (and (not (robot_at sensor_station)))
  :effect (and (not (robot_at prep_station)) (not (robot_at salt_dispenser)) (robot_at sensor_station)))
 (:action pickup_sensor
  :parameters ()
  :precondition (and (robot_at sensor_station) (not (robot_has salt_sensor)))
  :effect (and (robot_has salt_sensor)))
 (:action check_salt_level
  :parameters ()
  :precondition (and (robot_at prep_station) (robot_has salt_sensor) (food_at dish prep_station) (not (salt_level_checked dish)))
  :effect (and (salt_level_checked dish) (salt_level_safe dish)))
 (:action add_salt
  :parameters ()
  :precondition (and (robot_at salt_dispenser) (food_at dish prep_station) (food_needs_seasoning dish) (not (salt_added dish)))
  :effect (and (salt_added dish)))
)
