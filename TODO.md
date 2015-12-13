- should I remove iteration no from MCMCRun.record_iteration? Maybe I should keep the current 
iteration no internal and hence not allow overwriting previous log record?
- move_type in samples and best_samples lists are not correct; if the current move is not accepted, the move that led
to current sample is not the current_move. I am thinking about removing move_type from that list; you can get the same
info from run_log if needed. I didn't do this yet because I didn't want to break pickling of MCMCRun class with earlier
versions of the code.