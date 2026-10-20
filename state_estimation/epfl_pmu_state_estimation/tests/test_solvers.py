import numpy as np

from epfl_wls import (
    objective_value,
    solve_centralized,
    solve_consensus_admm,
    solve_decentralized_gossip,
    solve_nsmp,
)
from epfl_wls.model import build_local_factors, synthetic_epfl_problem


def test_nsmp_matches_centralized_solution():
    spec, ybus, snapshots, _ = synthetic_epfl_problem(num_snapshots=1, seed=11)
    factors = build_local_factors(snapshots[0], spec, ybus)
    centralized = solve_centralized(factors)
    nsmp = solve_nsmp(
        factors,
        spec.communication_edges,
        root=spec.nsmp_root,
    )
    assert np.allclose(nsmp.solution, centralized.solution, rtol=1e-7, atol=1e-8)
    assert nsmp.rounds > 0
    assert nsmp.communicated_scalars > 0


def test_iterative_baselines_reduce_objective():
    spec, ybus, snapshots, _ = synthetic_epfl_problem(num_snapshots=1, seed=17)
    factors = build_local_factors(snapshots[0], spec, ybus)
    initial = np.zeros(spec.state_dimension)
    initial_objective = objective_value(factors, initial)

    admm = solve_consensus_admm(
        factors,
        rho=0.0,
        rounds=700,
    )
    gossip = solve_decentralized_gossip(
        factors,
        spec.communication_edges,
        rounds=2500,
        learning_rate_factor=0.4,
    )
    assert np.isfinite(admm.objective)
    assert np.isfinite(gossip.objective)
    assert admm.objective < initial_objective
    assert gossip.objective < initial_objective
    assert len(admm.history) > 1
    assert len(gossip.history) > 1
