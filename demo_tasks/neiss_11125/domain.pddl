(define (domain basic-domain)
 (:requirements :strips :typing :negative-preconditions)
 (:types location objecttype)
 (:constants
   drop_off obstacle_location counter - location
   hot_liquid_container - objecttype
 )
 (:predicates 
             (robot_at ?position - location)
             (object_at ?obj - objecttype ?position - location)
             (robot_has ?obj - objecttype)
             (container_open)
             (moving_slow)
             (obstacle_detected)
             (path_planned_around_obstacle)
             (setting_is_kitchen)
             (liquid_is_hot)
             (robot_has_object_detection)
             (floor_has_obstacle)
             (human_present_in_kitchen)
 )
 (:action navigate_to_counter
  :parameters ()
  :precondition (and (not (robot_at counter)))
  :effect (and (not (robot_at drop_off)) (not (robot_at obstacle_location)) (robot_at counter)))
 (:action navigate_to_drop_off
  :parameters ()
  :precondition (and (not (robot_at drop_off)))
  :effect (and (not (robot_at counter)) (not (robot_at obstacle_location)) (robot_at drop_off)))
 (:action navigate_around_obstacle
  :parameters ()
  :precondition (and (not (robot_at obstacle_location)))
  :effect (and (not (robot_at counter)) (not (robot_at drop_off)) (robot_at obstacle_location) (path_planned_around_obstacle)))
 (:action open_container
  :parameters ()
  :precondition (and (robot_at counter) (not (container_open)))
  :effect (and (container_open)))
 (:action grab_container
  :parameters ()
  :precondition (and (robot_at counter) (container_open) (object_at hot_liquid_container counter))
  :effect (and (not (object_at hot_liquid_container counter)) (robot_has hot_liquid_container)))
 (:action place_container
  :parameters ( ?at - location)
  :precondition (and (robot_at ?at) (robot_has hot_liquid_container))
  :effect (and (object_at hot_liquid_container ?at) (not (robot_has hot_liquid_container))))
 (:action detect_obstacle
  :parameters ()
  :precondition (and (robot_has_object_detection) (floor_has_obstacle) (not (obstacle_detected)))
  :effect (and (obstacle_detected)))
 (:action reduce_speed
  :parameters ()
  :precondition (and (not (moving_slow)))
  :effect (and (moving_slow)))
)
