(define (domain basic-domain)
 (:requirements :strips :typing :negative-preconditions)
 (:types location objecttype)
 (:constants
   robot_storage bedside_table - location
   water_bottle - objecttype
 )
 (:predicates 
             (robot_at ?position - location)
             (object_at ?obj - objecttype ?position - location)
             (robot_has ?obj - objecttype)
             (bottle_at_edge)
             (bottle_at_center)
             (setting_is_elderly_care_bedroom)
             (robot_has_object_detection)
             (robot_has_edge_sensors)
             (bedside_table_has_edges)
 )
 (:action navigate_to_storage
  :parameters ()
  :precondition (and (not (robot_at robot_storage)))
  :effect (and (not (robot_at bedside_table)) (robot_at robot_storage)))
 (:action navigate_to_bedside_table
  :parameters ()
  :precondition (and (not (robot_at bedside_table)))
  :effect (and (not (robot_at robot_storage)) (robot_at bedside_table)))
 (:action grab_water_bottle
  :parameters ()
  :precondition (and (robot_at robot_storage) (object_at water_bottle robot_storage))
  :effect (and (not (object_at water_bottle robot_storage)) (robot_has water_bottle)))
 (:action place_water_bottle_at_edge
  :parameters ()
  :precondition (and (robot_at bedside_table) (robot_has water_bottle) (not (bottle_at_edge)))
  :effect (and (object_at water_bottle bedside_table) (not (robot_has water_bottle)) (bottle_at_edge) (not (bottle_at_center))))
 (:action place_water_bottle_at_center
  :parameters ()
  :precondition (and (robot_at bedside_table) (robot_has water_bottle) (not (bottle_at_center)))
  :effect (and (object_at water_bottle bedside_table) (not (robot_has water_bottle)) (bottle_at_center) (not (bottle_at_edge))))
)
