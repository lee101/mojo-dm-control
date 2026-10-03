"""Rigid-body transformations including velocities and static forces.

Mojo port of `dm_control.utils.transformations`. Names, argument order, defaults
and branch structure follow upstream; only the array plumbing differs (batched
NumPy arguments are handed to the kernels as flat C-contiguous rows).
"""

from __future__ import annotations

import logging

import numpy as np

from .._lib import addr, fn, lib, rows

# Constants used to determine when a rotation is close to a pole.
_POLE_LIMIT = 1.0 - 1e-6
_TOL = 1e-10

_EULER_ORDERINGS = ('XYZ', 'XYX', 'ZYX', 'XZY', 'YZX', 'ZXY', 'YXZ')


def _clip_within_precision(number, low, high, precision=_TOL):
  """Clips input to provided range, checking precision."""
  if (number < low - precision).any() or (number > high + precision).any():
    raise ValueError(
        'Input {:.12f} not inside range [{:.12f}, {:.12f}] with precision {}'.
        format(number, low, high, precision))

  return np.clip(number, low, high)


def _run_rmat_to_euler(rmat, ordering):
  """Runs the Mojo kernel for one 3x3 matrix."""
  rmat = np.ascontiguousarray(rmat, dtype=np.float64)
  euler = np.empty(3, dtype=np.float64)
  lib().mc_rmat_to_euler(addr(rmat), addr(euler),
                         _EULER_ORDERINGS.index(ordering))
  return euler


def _rmat_to_euler_xyz(rmat):
  """Converts a 3x3 rotation matrix to XYZ euler angles."""
  # | r00 r01 r02 |   |  cy*cz           -cy*sz            sy    |
  # | r10 r11 r12 | = |  cz*sx*sy+cx*sz   cx*cz-sx*sy*sz  -cy*sx |
  # | r20 r21 r22 |   | -cx*cz*sy+sx*sz   cz*sx+cx*sy*sz   cx*cy |
  if rmat[0, 2] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
  elif rmat[0, 2] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
  return _run_rmat_to_euler(rmat, 'XYZ')


def _rmat_to_euler_xyx(rmat):
  """Converts a 3x3 rotation matrix to XYX euler angles."""
  # | r00 r01 r02 |   |  cy      sy*sx1               sy*cx1             |
  # | r10 r11 r12 | = |  sy*sx0  cx0*cx1-cy*sx0*sx1  -cy*cx1*sx0-cx0*sx1 |
  # | r20 r21 r22 |   | -sy*cx0  cx1*sx0+cy*cx0*sx1   cy*cx0*cx1-sx0*sx1 |
  if rmat[0, 0] < 1.0:
    if rmat[0, 0] > -1.0:
      _clip_within_precision(rmat[0, 0], -1., 1.)
  return _run_rmat_to_euler(rmat, 'XYX')


def _rmat_to_euler_zyx(rmat):
  """Converts a 3x3 rotation matrix to ZYX euler angles."""
  if rmat[2, 0] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
  elif rmat[2, 0] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
  return _run_rmat_to_euler(rmat, 'ZYX')


def _rmat_to_euler_xzy(rmat):
  """Converts a 3x3 rotation matrix to XZY euler angles."""
  if rmat[0, 1] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
  elif rmat[0, 1] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
  return _run_rmat_to_euler(rmat, 'XZY')


def _rmat_to_euler_yzx(rmat):
  """Converts a 3x3 rotation matrix to YZX euler angles."""
  if rmat[1, 0] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
  elif rmat[1, 0] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
  return _run_rmat_to_euler(rmat, 'YZX')


def _rmat_to_euler_zxy(rmat):
  """Converts a 3x3 rotation matrix to ZXY euler angles."""
  if rmat[2, 1] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
  elif rmat[2, 1] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
  return _run_rmat_to_euler(rmat, 'ZXY')


def _rmat_to_euler_yxz(rmat):
  """Converts a 3x3 rotation matrix to YXZ euler angles."""
  if rmat[1, 2] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
  elif rmat[1, 2] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
  return _run_rmat_to_euler(rmat, 'YXZ')


def _axis_rmat(theta, full, axis):
  """Allocates the batch uninitialised and lets the kernel fill every entry."""
  n = 1 if np.isscalar(theta) else len(theta)
  rmat = np.empty((n, 4, 4) if full else (n, 3, 3))
  theta = np.ascontiguousarray(np.atleast_1d(theta), dtype=np.float64)
  fn('mc_axis_rotation')(addr(theta), addr(rmat), n, int(bool(full)), axis)
  return rmat.squeeze()


# Axis code per character, and the ordering string each code triple matches. The
# kernel takes the three axis codes as arguments so the Euler orderings resolve
# here, once per call, rather than in a branch inside the row loop.
_AXIS_CODE = {'X': 0, 'Y': 1, 'Z': 2}
# `euler_to_rmat`'s default is 'ZXZ', which is not one of the seven orderings
# `_eulermap` (and therefore `rmat_to_euler`) can invert. Upstream still builds
# it as the product of its three axis rotations, so it is composed here the same
# way; only `rmat_to_euler` rejects it, as upstream does.
_EULER_AXES = dict(
    [(order, tuple(_AXIS_CODE[c] for c in order)) for order in _EULER_ORDERINGS]
    + [('ZXZ', (2, 0, 2))])


def _euler_cols(theta, order):
  """`(n, 3)` holding `theta[:, 0]`, `theta[:, 1]`, `theta[:, 2]` concatenated.

  The composition kernel gathers one width-`W` vector of each angle column, so
  the columns it needs must be adjacent in memory. This copy is the whole price
  of that layout change, and the output's (n, 3, 3) or (n, 4, 4) allocation
  dwarfs it: it is one pass over 3n doubles against s * s * n doubles written.
  """
  euler = np.ascontiguousarray(np.atleast_2d(theta))
  if euler.shape[1] != 3:
    raise ValueError(
        'Euler angles must have three columns, got {}.'.format(euler.shape))
  del order  # the axis codes travel separately, to the kernel call
  n = euler.shape[0]
  cols = np.empty(3 * n, dtype=np.float64)
  cols.reshape(3, n)[:] = euler.T
  return cols


def _euler_planes(rmat, n, full):
  """A contiguous `(n, s, s)` batch of the same matrices `rmat` already holds.

  `euler_to_rmat` must return the same array regardless of whether the caller
  keeps the argument's memory, so the kernel's contiguous result is copied back
  rather than returned in place. `np.copyto` does it as a strided memcpy with
  no intermediate; the cost is one extra pass over the output, which is why it
  is paid here and not on every call inside the kernel.
  """
  shape = (n, 4, 4) if full else (n, 3, 3)
  out = np.empty(shape, dtype=np.float64)
  if n:
    np.copyto(out.reshape(n, -1), rmat.reshape(n, -1))
  return out


# map from full rotation orderings to euler conversion functions
_eulermap = {
    'XYZ': _rmat_to_euler_xyz,
    'XYX': _rmat_to_euler_xyx,
    'ZYX': _rmat_to_euler_zyx,
    'XZY': _rmat_to_euler_xzy,
    'YZX': _rmat_to_euler_yzx,
    'ZXY': _rmat_to_euler_zxy,
    'YXZ': _rmat_to_euler_yxz
}


def euler_to_quat(euler_vec, ordering='XYZ'):
  """Returns the quaternion corresponding to the provided euler angles.

  Args:
    euler_vec: The euler angle rotations.
    ordering: (str) Desired euler angle ordering.

  Returns:
    quat: A quaternion [w, i, j, k]
  """
  mat = euler_to_rmat(euler_vec, ordering=ordering)
  return mat_to_quat(mat)


def euler_to_rmat(euler_vec, ordering='ZXZ', full=False):
  """Returns rotation matrix (or transform) for the given Euler rotations.

  Euler*** methods compose a Rotation matrix corresponding to the given
  rotations r1, r2, r3 following the given rotation ordering. Ordering
  specifies the order of rotation matrices in matrix multiplication order.
  E.g. for XYZ we return rotX(r1) * rotY(r2) * rotZ(r3).

  Args:
    euler_vec: The euler angle rotations.
    ordering: euler angle ordering string (see _eulermap).
    full: If true, returns a full 4x4 transfom.

  Returns:
    The rotation matrix or homogenous transform corresponding to the given
    Euler rotation.
  """
  # Upstream composes three axis rotations with two batched matrix products.
  # The kernel evaluates the same six cosines and six sines, contracts the
  # three rotations and writes the result in a single pass, so none of the
  # five `(n, s, s)` arrays that route through the products is built.
  full = int(bool(full))
  cols = _euler_cols(euler_vec, ordering)
  n = cols.size // 3
  a0, a1, a2 = _EULER_AXES[ordering]
  packed = np.empty((n, 4, 4) if full else (n, 3, 3), dtype=np.float64)
  fn('mc_euler_to_rmat')(addr(cols), addr(packed), n, full, a0, a1, a2)
  result = _euler_planes(packed, n, full)
  return result.squeeze()


def quat_conj(quat):
  """Return conjugate of quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    A quaternion [w, -i, -j, -k] representing the inverse of the rotation
    defined by `quat` (not assuming normalization).
  """
  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = np.asarray(quat)
  arr, flat = rows(quat, 4)
  dst = np.empty(flat.shape, dtype=np.float64)
  lib().mc_quat_conj(addr(flat), addr(dst), flat.shape[0])
  return dst.reshape(arr.shape)


def quat_inv(quat):
  """Return inverse of quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    A quaternion representing the inverse of the original rotation.
  """
  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = np.asarray(quat)
  arr, flat = rows(quat, 4)
  dst = np.empty(flat.shape, dtype=np.float64)
  lib().mc_quat_inv(addr(flat), addr(dst), flat.shape[0])
  return dst.reshape(arr.shape)


def quat_mul(quat1, quat2):
  """Computes the Hamilton product of two quaternions.

  Any number of leading batch dimensions is supported.

  Args:
    quat1: A quaternion [w, i, j, k].
    quat2: A quaternion [w, i, j, k].

  Returns:
    The quaternion product quat1 * quat2.
  """
  # `qmat @ quat2[..., None]` broadcasts the batch dimensions upstream.
  quat1, quat2 = np.broadcast_arrays(np.asarray(quat1), np.asarray(quat2))
  arr, flat1 = rows(quat1, 4)
  _, flat2 = rows(quat2, 4)
  dst = np.empty(flat1.shape, dtype=np.float64)
  lib().mc_quat_mul(addr(flat1), addr(flat2), addr(dst), flat1.shape[0])
  return dst.reshape(arr.shape)


def quat_diff(source, target):
  """Computes quaternion difference between source and target quaternions.

  This function supports inputs with or without leading batch dimensions.

  Args:
    source: A quaternion [w, i, j, k].
    target: A quaternion [w, i, j, k].

  Returns:
    A quaternion representing the rotation from source to target.
  """
  return quat_mul(quat_conj(source), target)


def quat_log(quat, tol=_TOL):
  """Log of a quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A quaternion [w, i, j, k].
    tol: numerical tolerance to prevent nan.

  Returns:
    4D array representing the log of `quat`. This is analogous to
    `rmat_to_axisangle`.
  """
  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = np.asarray(quat)
  # Clip to 2*tol because we subtract it here.
  _clip_within_precision(quat[..., 0:1] - tol, -1., 1., precision=2. * tol)
  arr, flat = rows(quat, 4)
  dst = np.empty(flat.shape, dtype=np.float64)
  lib().mc_quat_log(addr(flat), addr(dst), flat.shape[0], float(tol))
  return dst.reshape(arr.shape)


def quat_dist(source, target):
  """Computes distance between source and target quaternions.

  This function assumes that both input arguments are unit quaternions.

  This function supports inputs with or without leading batch dimensions.

  Args:
    source: A quaternion [w, i, j, k].
    target: A quaternion [w, i, j, k].

  Returns:
    Scalar representing the rotational distance from source to target.
  """
  source, target = np.broadcast_arrays(np.asarray(source), np.asarray(target))
  arr, flat_source = rows(source, 4)
  _, flat_target = rows(target, 4)
  dst = np.empty(flat_source.shape[0], dtype=np.float64)
  lib().mc_quat_dist(addr(flat_source), addr(flat_target), addr(dst),
                     flat_source.shape[0], _TOL)
  return dst.reshape(arr.shape[:-1] + (1,))


def quat_rotate(quat, vec):
  """Rotate a vector by a quaternion.

  Args:
    quat: A quaternion [w, i, j, k].
    vec: A 3-vector representing a position.

  Returns:
    The rotated vector.
  """
  quat = np.asarray(quat, dtype=np.float64)
  vec = np.asarray(vec, dtype=np.float64)
  if quat.shape == (4,) and vec.shape == (3,):
    # The unbatched call is the one dm_control's task code makes; it needs no
    # reshape, no broadcast and no copy, only one crossing of the FFI.
    dst = np.empty(3, dtype=np.float64)
    fn('mc_quat_rotate_scalar')(*quat.tolist(), *vec.tolist(), addr(dst))
    return dst
  if quat.shape == (4,) and vec.ndim > 1:
    # One quaternion against a batch of vectors: the batch dimension lives on
    # the vector, so the result is `(vec.shape[:-1] + (3,))`.
    batch = vec.shape[0]
    out_shape = vec.shape[:-1] + (3,)
    flat_quat = np.broadcast_to(quat, (batch, 4))
    flat_vec = rows(vec, 3)[1]
  elif vec.shape == (3,) and quat.ndim > 1:
    # `np.hstack([[0], vec])` broadcasts a single 3-vector across the
    # quaternion batch upstream; the kernel input is broadcast the same way.
    arr, flat_quat = rows(quat, 4)
    out_shape = arr.shape[:-1] + (3,)
    flat_vec = np.broadcast_to(vec, (flat_quat.shape[0], 3))
  else:
    arr, flat_quat = rows(quat, 4)
    _, flat_vec = rows(vec, 3)
    if flat_quat.shape != flat_vec.shape:
      raise ValueError(
          'quat and vec must broadcast, got {} and {}.'.format(
              np.shape(quat), np.shape(vec)))
    out_shape = arr.shape[:-1] + (3,)
  n = flat_quat.shape[0]
  flat_quat = np.ascontiguousarray(flat_quat)
  flat_vec = np.ascontiguousarray(flat_vec)
  dst = np.empty((n, 3), dtype=np.float64)
  lib().mc_quat_rotate(addr(flat_quat), addr(flat_vec), addr(dst), n)
  return dst.reshape(out_shape)


def quat_to_axisangle(quat):
  """Returns the axis-angle corresponding to the provided quaternion.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    axisangle: A 3x1 numpy array describing the axis of rotation, with angle
        encoded by its length.
  """
  quat = np.ascontiguousarray(quat, dtype=np.float64)
  _clip_within_precision(quat[0], -1., 1.)
  axisangle = np.empty(3, dtype=np.float64)
  lib().mc_quat_to_axisangle(addr(quat), addr(axisangle))
  return axisangle


def quat_to_euler(quat, ordering='XYZ'):
  """Returns the euler angles corresponding to the provided quaternion.

  Args:
    quat: A quaternion [w, i, j, k].
    ordering: (str) Desired euler angle ordering.

  Returns:
    euler_vec: The euler angle rotations.
  """
  mat = quat_to_mat(quat)
  return rmat_to_euler(mat[0:3, 0:3], ordering=ordering)


def quat_to_mat(quat):
  """Return homogeneous rotation matrix from quaternion.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    A 4x4 homogeneous matrix with the rotation corresponding to `quat`.
  """
  quat = np.ascontiguousarray(quat, dtype=np.float64)
  mat = np.empty((4, 4), dtype=np.float64)
  lib().mc_quat_to_mat(addr(quat), addr(mat))
  return mat


def rotation_x_axis(theta, full=False):
  """Returns a rotation matrix of a rotation about the X-axis.

  Supports vector-valued theta, in which case the returned array is of shape
  (len(t), 3, 3), or (len(t), 4, 4) if full=True. If theta is scalar the batch
  dimension is squeezed out.

  Args:
    theta: The rotation amount.
    full: (bool) If true, returns a full 4x4 transform.
  """
  return _axis_rmat(theta, full, 0)


def rotation_y_axis(theta, full=False):
  """Returns a rotation matrix of a rotation about the Y-axis.

  Supports vector-valued theta, in which case the returned array is of shape
  (len(t), 3, 3), or (len(t), 4, 4) if full=True. If theta is scalar the batch
  dimension is squeezed out.

  Args:
    theta: The rotation amount.
    full: (bool) If true, returns a full 4x4 transfom.
  """
  return _axis_rmat(theta, full, 1)



def rotation_z_axis(theta, full=False):
  """Returns a rotation matrix of a rotation about the z-axis.

  Supports vector-valued theta, in which case the returned array is of shape
  (len(t), 3, 3), or (len(t), 4, 4) if full=True. If theta is scalar the batch
  dimension is squeezed out.

  Args:
    theta: The rotation amount.
    full: (bool) If true, returns a full 4x4 transfom.
  """
  return _axis_rmat(theta, full, 2)


def rmat_to_euler(rmat, ordering='ZXZ'):
  """Returns the euler angles corresponding to the provided rotation matrix.

  Args:
    rmat: The rotation matrix.
    ordering: (str) Desired euler angle ordering.

  Returns:
    Euler angles corresponding to the provided rotation matrix.
  """
  return _eulermap[ordering](rmat)


def mat_to_quat(mat):
  """Return quaternion from homogeneous or rotation matrix.

  Args:
    mat: A homogeneous transform or rotation matrix

  Returns:
    A quaternion [w, i, j, k].
  """
  mat = np.asarray(mat, dtype=np.float64)
  if mat.shape == (3, 3):
    tmp = np.eye(4)
    tmp[0:3, 0:3] = mat
    mat = tmp
  mat = np.ascontiguousarray(mat)
  q = np.empty((4,), dtype=np.float64)
  lib().mc_mat_to_quat(addr(mat), addr(q))
  return q


def axisangle_to_quat(axisangle, tol=0.0):
  """Returns the quaternion corresponding to the provided axis-angle.

  Args:
    axisangle: A 3x1 numpy array describing the axis of rotation, with angle
      encoded by its length.
    tol: Tolerance for the angle magnitude below which the identity quaternion
      is returned.

  Returns:
    A quaternion [w, i, j, k].
  """
  axisangle = np.asarray(axisangle)
  arr, flat = rows(axisangle, 3)
  quat = np.empty((flat.shape[0], 4), dtype=np.float64)
  lib().mc_axisangle_to_quat(addr(flat), addr(quat), flat.shape[0], float(tol))
  return quat.reshape(arr.shape[:-1] + (4,))


# ################
# # 2D Functions #
# ################


def rotation_matrix_2d(theta):
  theta = np.asarray(theta, dtype=np.float64)
  arr, flat = rows(theta, 1)
  res = np.empty((flat.shape[0], 4), dtype=np.float64)
  lib().mc_rotation_matrix_2d(addr(flat), addr(res), flat.shape[0])
  # Upstream builds `np.array([[ct, -st], [st, ct]])`, so a vector-valued
  # `theta` gives shape (2, 2, n) and a scalar gives (2, 2).
  out = res.reshape(flat.shape[0], 2, 2).transpose(1, 2, 0)
  return out[..., 0] if out.ndim == 3 and flat.shape[0] == 1 and arr.ndim == 0 else out
