(define (problem basic-problem)
 (:domain basic-domain)
 (:objects
 )
 (:init
              (robot_at counter)
              (object_at hot_liquid_container counter)
              (setting_is_kitchen)
              (liquid_is_hot)
              (robot_has_object_detection)
              (floor_has_obstacle)
              (human_present_in_kitchen)
 )
 (:goal (and 
           (object_at hot_liquid_container drop_off)
        )
 )
)
