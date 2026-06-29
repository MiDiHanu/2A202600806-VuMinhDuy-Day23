# Crash-resume / time-travel evidence

Procedure: run a scenario with a SQLite checkpointer, discard the graph
and saver, rebuild both from the same on-disk DB, then read state back.

- **resume_success**: True
- **thread_id**: resume-probe-thread
- **database**: checkpoints_resume.db
- **checkpoints_in_history**: 6
- **final_answer_present**: True
- **recovered_route**: simple
- **events_recovered**: 4
