# Six-client Gossip experiment

Run `run_gossip` from this directory. To obtain the result structure, use `results = run_gossip();`.

This directory runs Gossip only, using centralized least squares as an accuracy reference. The paper checkpoint is the first relative objective gap at most `1e-15`: 23 iterations, 230 messages, and 230 Float64 scalars (1,840 bytes).

`results/communication_to_1e-15.csv` contains one Gossip row. The full 500-round convergence history is retained in `gossip_convergence.csv` and the MAT file. Run `run_all_experiments` from the parent directory for the combined comparison.
