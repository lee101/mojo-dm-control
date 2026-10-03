r"""Optimal policy for LQR levels.

LQR control problem is described in
https://en.wikipedia.org/wiki/Linear-quadratic_regulator#Infinite-horizon.2C_discrete-time_LQR

Mojo port of `dm_control.suite.lqr_solver`. The sparse mass-matrix expansion
runs in `src/ported.mojo`; the Riccati solve and the eigenvalue scan stay in
SciPy/NumPy (see README).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as scipy_linalg

from .._lib import addr, lib


def solve(env):
  """Returns the optimal value and policy for LQR problem.

  Args:
    env: An instance of `control.Environment` with LQR level.

  Returns:
    p: A numpy array, the Hessian of the optimal total cost-to-go (value
      function at state x) is V(x) = .5 * x' * p * x.
    k: A numpy array which gives the optimal linear policy u = k * x.
    beta: The maximum eigenvalue of (a + b * k). Under optimal policy, at
      timestep n the state tends to 0 like beta^n.

  Raises:
    RuntimeError: If the controlled system is unstable.
  """

  model = env.physics.model
  data = env.physics.data
  n = model.nq  # number of DoFs
  m = model.nu  # number of controls

  # Compute the mass matrix.
  mass = np.zeros((n, n))
  sparse_m = np.ascontiguousarray(data.ptr.M.ravel(), dtype=np.float64)
  rownnz = np.ascontiguousarray(model.ptr.M_rownnz.ravel(), dtype=np.int64)
  rowadr = np.ascontiguousarray(model.ptr.M_rowadr.ravel(), dtype=np.int64)
  colind = np.ascontiguousarray(model.ptr.M_colind.ravel(), dtype=np.int64)
  lib().mc_sym2dense(addr(sparse_m), addr(mass), n, addr(rownnz), addr(rowadr),
                     addr(colind))

  # Compute input matrices a, b, q and r to the DARE solvers.
  # State transition matrix a.
  stiffness = np.diag(model.jnt_stiffness.ravel())
  damping = np.diag(model.dof_damping.ravel())
  dt = model.opt.timestep

  j = np.linalg.solve(-mass, np.hstack((stiffness, damping)))
  a = np.eye(2 * n) + dt * np.vstack(
      (dt * j + np.hstack((np.zeros((n, n)), np.eye(n))), j))

  # Control transition matrix b.
  b = np.vstack((np.eye(m), np.zeros((n - m, m))))
  bc = np.linalg.solve(mass, b)
  b = dt * np.vstack((dt * bc, bc))

  # State cost Hessian q.
  q = np.diag(np.hstack([np.ones(n), np.zeros(n)]))

  # Control cost Hessian r.
  r = env.task.control_cost_coef * np.eye(m)

  # Solve the discrete algebraic Riccati equation.
  p = scipy_linalg.solve_discrete_are(a, b, q, r)
  k = -np.linalg.solve(_mm(b.T, _mm(p, b)) + r, _mm(b.T, _mm(p, a)))

  # Under optimal policy, state tends to 0 like beta^n_timesteps
  beta = np.abs(np.linalg.eigvals(a + _mm(b, k))).max()
  if beta >= 1.0:
    raise RuntimeError('Controlled system is unstable.')
  return p, k, beta


def _mm(a, b):
  """A single `(n, m) @ (m, o)` matrix product.

  The matrices here are `(2n, 2n)` and `(2n, n)` at most, so NumPy's own
  `dgemm` on a contiguous copy is the product to use; the Mojo side of this
  module is the sparse mass-matrix expansion.
  """
  return np.ascontiguousarray(a, dtype=np.float64) @ np.ascontiguousarray(
      b, dtype=np.float64)
