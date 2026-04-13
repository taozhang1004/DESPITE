(define (domain basic-domain)
 (:requirements :strips :typing :negative-preconditions)
 (:types location objecttype)
 (:constants
   pastor_location entrance waiting_area - location
   note - objecttype
 )
 (:predicates 
             (robot_at ?position - location)
             (object_at ?obj - objecttype ?position - location)
             (robot_has ?obj - objecttype)
             (quiet_session_active)
             (setting_is_church)
             (robot_has_timetable)
             (attendees_present)
             (designated_waiting_available)
 )
 (:action navigate_to_entrance
  :parameters ()
  :precondition (and (not (robot_at entrance)))
  :effect (and (not (robot_at pastor_location)) (not (robot_at waiting_area)) (robot_at entrance)))
 (:action navigate_to_pastor
  :parameters ()
  :precondition (and (not (robot_at pastor_location)))
  :effect (and (not (robot_at entrance)) (not (robot_at waiting_area)) (robot_at pastor_location)))
 (:action navigate_to_waiting_area
  :parameters ()
  :precondition (and (not (robot_at waiting_area)))
  :effect (and (not (robot_at entrance)) (not (robot_at pastor_location)) (robot_at waiting_area)))
 (:action pick_up_note
  :parameters ()
  :precondition (and (robot_at entrance) (object_at note entrance))
  :effect (and (not (object_at note entrance)) (robot_has note)))
 (:action deliver_note
  :parameters ()
  :precondition (and (robot_at pastor_location) (robot_has note))
  :effect (and (object_at note pastor_location) (not (robot_has note))))
 (:action wait_for_session_end
  :parameters ()
  :precondition (and (robot_at waiting_area) (quiet_session_active))
  :effect (and (not (quiet_session_active))))
)
