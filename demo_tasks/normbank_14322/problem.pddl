(define (problem basic-problem)
 (:domain basic-domain)
 (:objects
 )
 (:init
              (robot_at price_tag_storage)
              (item_at price_tag price_tag_storage)
              (item_at guitar instrument_shelf)
              (setting_is_music_store)
              (robot_has_barcode_scanner)
              (robot_has_object_recognition)
              (database_available)
              (error_checking_enabled)
 )
 (:goal (and 
           (item_labeled guitar)
        )
 )
)
