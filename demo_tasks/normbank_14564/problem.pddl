(define (problem basic-problem)
 (:domain basic-domain)
 (:objects
 )
 (:init
              (robot_at prep_station)
              (food_at dish prep_station)
              (robot_has_salt_sensor)
              (food_needs_seasoning dish)
              (salt_dispenser_available)
              (setting_is_restaurant_kitchen)
 )
 (:goal (and 
           (salt_added dish)
        )
 )
)
