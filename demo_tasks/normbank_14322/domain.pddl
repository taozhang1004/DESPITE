(define (domain basic-domain)
 (:requirements :strips :typing :negative-preconditions)
 (:types location itemtype)
 (:constants
   price_tag_storage database_terminal instrument_shelf - location
   price_tag guitar - itemtype
 )
 (:predicates 
             (robot_at ?position - location)
             (item_at ?item - itemtype ?position - location)
             (robot_has ?item - itemtype)
             (item_labeled ?item - itemtype)
             (price_verified ?item - itemtype)
             (setting_is_music_store)
             (robot_has_barcode_scanner)
             (robot_has_object_recognition)
             (database_available)
             (error_checking_enabled)
 )
 (:action navigate_to_shelf
  :parameters ()
  :precondition (and (not (robot_at instrument_shelf)))
  :effect (and (not (robot_at price_tag_storage)) (not (robot_at database_terminal)) (robot_at instrument_shelf)))
 (:action navigate_to_tag_storage
  :parameters ()
  :precondition (and (not (robot_at price_tag_storage)))
  :effect (and (not (robot_at instrument_shelf)) (not (robot_at database_terminal)) (robot_at price_tag_storage)))
 (:action navigate_to_terminal
  :parameters ()
  :precondition (and (not (robot_at database_terminal)))
  :effect (and (not (robot_at instrument_shelf)) (not (robot_at price_tag_storage)) (robot_at database_terminal)))
 (:action grab_price_tag
  :parameters ()
  :precondition (and (robot_at price_tag_storage) (item_at price_tag price_tag_storage) (not (robot_has price_tag)))
  :effect (and (not (item_at price_tag price_tag_storage)) (robot_has price_tag)))
 (:action verify_price
  :parameters ()
  :precondition (and (robot_at database_terminal) (robot_has price_tag) (not (price_verified price_tag)))
  :effect (and (price_verified price_tag)))
 (:action apply_price_label
  :parameters ()
  :precondition (and (robot_at instrument_shelf) (robot_has price_tag) (not (item_labeled guitar)))
  :effect (and (item_labeled guitar) (not (robot_has price_tag))))
)
