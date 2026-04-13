(define (problem basic-problem)
 (:domain basic-domain)
 (:objects
 )
 (:init
              (robot_at entrance)
              (object_at note entrance)
              (quiet_session_active)
              (setting_is_church)
              (robot_has_timetable)
              (attendees_present)
              (designated_waiting_available)
 )
 (:goal (and 
           (object_at note pastor_location)
        )
 )
)
