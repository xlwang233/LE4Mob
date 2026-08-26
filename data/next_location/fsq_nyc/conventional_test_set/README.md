# FSQ-NYC conventional evaluation test set

`whole_test.csv` contains the authoritative conventional evaluation membership. Samples whose target is unknown-location ID 1 are excluded. Samples containing ID 1 only in their history are retained.

`test_index` is the zero-based row index in the prepared test pickle; `target_location_id` provides an integrity check. The manifest records the filtering policy, counts, and checksums.
