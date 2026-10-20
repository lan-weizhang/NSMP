from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
EPFL_PROJECT = PROJECT.parent / "epfl_pmu_state_estimation"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(EPFL_PROJECT))

from epfl_wls.quadratic import combine_factors
from epfl_wls.solvers import solve_centralized, solve_nsmp
from socal_wls import build_factors, load_socal_case


def test_real_a_side_case_is_observable_and_nsmp_is_exact():
    case = load_socal_case(PROJECT / "official_repo", max_snapshots=1)
    assert case.num_nodes == 12
    assert case.state_dimension == 72
    assert len(case.tree_edges) == 11
    assert len(case.measured_nodes) == 6

    factors = build_factors(case, case.snapshots[0])
    combined = combine_factors(factors)
    assert len(factors) == 12
    assert np.linalg.matrix_rank(combined.hessian) == 72

    centralized = solve_centralized(factors)
    nsmp = solve_nsmp(factors, case.tree_edges, root=0)
    np.testing.assert_allclose(nsmp.solution, centralized.solution, atol=1e-6, rtol=1e-6)
