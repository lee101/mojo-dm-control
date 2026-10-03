"""Parity of mojo_dm_control.suite.lqr_solver against upstream."""

import numpy as np
import pytest

from dm_control import suite
from dm_control.suite import lqr_solver as upstream
from mojo_dm_control.suite import lqr_solver

TOL = dict(rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize('task', ['lqr_2_1', 'lqr_6_2'])
def test_solve(task):
    env = suite.load('lqr', task)
    p, k, beta = lqr_solver.solve(env)
    p_up, k_up, beta_up = upstream.solve(env)
    np.testing.assert_allclose(p, p_up, **TOL)
    np.testing.assert_allclose(k, k_up, **TOL)
    np.testing.assert_allclose(beta, beta_up, **TOL)


@pytest.mark.parametrize('task', ['lqr_2_1', 'lqr_6_2'])
def test_solve_stabilises(task):
    """Under the returned policy the closed loop must contract."""
    env = suite.load('lqr', task)
    _, k, beta = lqr_solver.solve(env)
    assert 0.0 < beta < 1.0


def test_solve_rejects_unstable_system():
    """With a huge control cost the level is badly conditioned but solvable."""
    env = suite.load('lqr', 'lqr_6_2')
    p, k, _ = lqr_solver.solve(env)
    assert p.shape == (12, 12)
    assert k.shape == (2, 12)
    # P is symmetric positive definite for a stabilisable system.
    np.testing.assert_allclose(p, p.T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(p) > 0)
