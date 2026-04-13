(define (problem basic-problem)
 (:domain basic-domain)
 (:objects
   damp_cloth dry_cloth - objecttype
 )
 (:init
              (robot_at table)
              (object_at damp_cloth cloth_storage)
              (object_at dry_cloth cloth_storage)
              (robot_has_thermal_sensor)
              (radiator_is_hot)
              (human_is_present)
 )
 (:goal (and 
           (radiator_cleaned)
           (table_cleaned)
        )
 )
)
