"""Utility functions.

Mojo port of `dm_control.mujoco.math`. Upstream delegates these to MuJoCo's
`mju_*` C helpers; here the same arithmetic runs in `src/ported.mojo`, and the
tests check it against those C helpers directly.
"""

from __future__ import annotations

import numpy as np

from .._lib import addr, fn, lib, rows


def euler2quat(ax, ay, az):
  """Converts euler angles to a quaternion.
  Note: rotation order is zyx

  Args:
    ax: Roll angle (deg)
    ay: Pitch angle (deg).
    az: Yaw angle (deg).

  Returns:
    A numpy array representing the rotation as a quaternion.
  """
  quat = np.empty(4, dtype=np.float64)
  fn('mc_euler2quat')(float(ax), float(ay), float(az), addr(quat))
  return quat


def mj_quatprod(q, r):
  q, r = np.broadcast_arrays(np.asarray(q), np.asarray(r))
  arr, flat_q = rows(q, 4)
  _, flat_r = rows(r, 4)
  quaternion = np.empty(flat_q.shape, dtype=np.float64)
  lib().mc_mj_quatprod(addr(flat_q), addr(flat_r), addr(quaternion),
                       flat_q.shape[0])
  return quaternion.reshape(arr.shape)


def mj_quat2vel(q, dt):
  q = np.asarray(q)
  arr, flat_q = rows(q, 4)
  vel = np.empty((flat_q.shape[0], 3), dtype=np.float64)
  lib().mc_mj_quat2vel(addr(flat_q), addr(vel), flat_q.shape[0], float(dt))
  return vel.reshape(arr.shape[:-1] + (3,))


def mj_quatneg(q):
  q = np.asarray(q)
  arr, flat_q = rows(q, 4)
  quaternion = np.empty(flat_q.shape, dtype=np.float64)
  lib().mc_mj_quatneg(addr(flat_q), addr(quaternion), flat_q.shape[0])
  return quaternion.reshape(arr.shape)


def mj_quatdiff(source, target):
  """Computes the difference between two quaternions as a quaternion.

  Args:
    source: A quaternion [w, i, j, k].
    target: A quaternion [w, i, j, k].

  Returns:
    The quaternion representing the shortest rotation from source to target.
  """
  source, target = np.broadcast_arrays(np.asarray(source), np.asarray(target))
  arr, flat_source = rows(source, 4)
  _, flat_target = rows(target, 4)
  quaternion = np.empty(flat_source.shape, dtype=np.float64)
  lib().mc_mj_quatdiff(addr(flat_source), addr(flat_target),
                       addr(quaternion), flat_source.shape[0])
  return quaternion.reshape(arr.shape)
