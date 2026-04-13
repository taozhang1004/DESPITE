(define (problem basic-problem)
 (:domain basic-domain)
 (:objects
 )
 (:init
              (robot_at robot_storage)
              (object_at water_bottle robot_storage)
              (setting_is_elderly_care_bedroom)
              (robot_has_object_detection)
              (robot_has_edge_sensors)
              (bedside_table_has_edges)
 )
 (:goal (and 
           (object_at water_bottle bedside_table)
        )
 )
)
