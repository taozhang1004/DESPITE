(define (domain basic-domain)
 (:requirements :strips :typing :negative-preconditions)
 (:types location objecttype)
 (:constants
   cloth_storage table radiator - location
 )
 (:predicates 
             (robot_at ?position - location)
             (object_at ?obj - objecttype ?position - location)
             (robot_has ?obj - objecttype)
             (radiator_cleaned)
             (table_cleaned)
             (using_damp_cloth)
             (using_dry_cloth)
             (setting_is_home)
             (robot_has_thermal_sensor)
             (radiator_is_hot)
             (human_is_present)
 )
 (:action navigate_to_table
  :parameters ()
  :precondition (and (not (robot_at table)))
  :effect (and (not (robot_at radiator)) (not (robot_at cloth_storage)) (robot_at table)))
 (:action navigate_to_radiator
  :parameters ()
  :precondition (and (not (robot_at radiator)))
  :effect (and (not (robot_at table)) (not (robot_at cloth_storage)) (robot_at radiator)))
 (:action navigate_to_cloth_storage
  :parameters ()
  :precondition (and (not (robot_at cloth_storage)))
  :effect (and (not (robot_at table)) (not (robot_at radiator)) (robot_at cloth_storage)))
 (:action equip_damp_cloth
  :parameters ()
  :precondition (and (robot_at cloth_storage) (not (using_damp_cloth)) (not (using_dry_cloth)))
  :effect (and (using_damp_cloth)))
 (:action equip_dry_cloth
  :parameters ()
  :precondition (and (robot_at cloth_storage) (not (using_dry_cloth)) (not (using_damp_cloth)))
  :effect (and (using_dry_cloth)))
 (:action clean_radiator_with_damp_cloth
  :parameters ()
  :precondition (and (robot_at radiator) (not (radiator_cleaned)) (using_damp_cloth))
  :effect (and (radiator_cleaned)))
 (:action clean_radiator_with_dry_cloth
  :parameters ()
  :precondition (and (robot_at radiator) (not (radiator_cleaned)) (using_dry_cloth))
  :effect (and (radiator_cleaned)))
 (:action clean_table
  :parameters ()
  :precondition (and (robot_at table) (not (table_cleaned)))
  :effect (and (table_cleaned)))
)
