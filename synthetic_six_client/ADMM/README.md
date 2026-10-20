# Six-client ADMM experiment

Run `run_admm` from this directory. To obtain the result structure, use `results = run_admm();`.

This directory runs edge-consensus ADMM only, using centralized least squares as an accuracy reference. The original data and settings are retained: `rho=100`, absolute and relative residual tolerances both `1e-8`. The run takes 30 iterations, 300 messages, and 300 Float64 scalars (2,400 bytes).

Outputs in `results/` include a single-row `communication_summary.csv`, `admm_convergence.csv`, `admm_results.mat`, and ADMM-only convergence figures `admm_convergence.png/.pdf`. Run `run_all_experiments` from the parent directory for the combined comparison.
