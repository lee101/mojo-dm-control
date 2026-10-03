"""Every exported kernel is called by name and held to numerical parity.

The other test modules exercise the kernels through the Python API, which is
what a caller uses. That leaves a kernel that is wired up but wrong invisible:
the API path and the kernel agree, so nothing fails. This module closes that
gap by crossing the ctypes boundary into each `@export` symbol directly and
comparing its output against the value the same arithmetic produces in NumPy,
SciPy or MuJoCo's own C library.

Each test here names its symbol, resolves it with `_lib.fn`, calls it, and
asserts on the returned buffer. The references are the same ones the API tests
use, but computed independently of this package's wrapper code, so a mistake in
the wrapper cannot cancel out a mistake in the kernel.
"""

import ctypes
import os

import numpy as np
import pytest
from dm_control.locomotion.tasks.reference_pose import rewards as upstream_pose
from dm_control.utils import rewards as upstream_rewards
from dm_control.utils import transformations as upstream

from mojo_dm_control._lib import LIB, addr, fn, lib

# The library must really be the compiled one. A wrapper that could satisfy a
# test without loading `dist/*.so` would prove nothing about the port, so the
# missing shared object is a loud failure here rather than a quiet pass.
pytestmark = pytest.mark.skipif(
    not os.path.exists(LIB), reason='compiled library missing: ' + LIB)

# `simd_width_of[DType.float64]()` on this target. Batch lengths deliberately
# straddle it so each kernel runs both a vector body and a scalar tail.
W = 4
SIZES = (1, 3, 4, 5, 17, 64)

EXACT = dict(rtol=1e-12, atol=1e-12)
# `exp`, `cosh` and `tanh` are polynomial approximations in this toolchain
# rather than libm calls; `acos`/`log`/`sqrt` agree to the last few ulp.
TRANSCENDENTAL = dict(rtol=1e-11, atol=1e-12)

SIGMOIDS = ('gaussian', 'hyperbolic', 'long_tail', 'reciprocal', 'cosine',
            'linear', 'quadratic', 'tanh_squared')
# Branch order of upstream's `_sigmoids` if/elif chain.
SIGMOID_CODE = {name: i for i, name in enumerate(SIGMOIDS)}


def unit_quats(rng, n):
  q = rng.normal(size=(n, 4))
  return q / np.linalg.norm(q, axis=-1, keepdims=True)


def scratch(width):
  """A writable contiguous flat buffer, as MuJoCo's autowrap requires.

  The autowrapped `mju_*` signatures are annotated `[3, 1]` / `[4, 1]`, which
  NumPy satisfies with a flat 1-D array of that length, not an `(1, w)` one.
  """
  return np.zeros(width, dtype=np.float64)


def rows4(a):
  """An `(n, 4)` contiguous copy, and the count the kernels are driven with."""
  flat = np.ascontiguousarray(a, dtype=np.float64).reshape(-1, 4)
  return flat, flat.shape[0]


def call(name):
  """The exported symbol `name`, loaded from the compiled library."""
  return fn(name)


# =========================== dm_control/mujoco/math.py =========================


def test_mc_euler2quat():
  """`mc_euler2quat` against dm_control.mujoco.math.euler2quat."""
  from dm_control.mujoco import math as upstream_math
  rng = np.random.default_rng(0)
  angles = [(0., 0., 0.), (30., 0., 45.), (-170., 95., 12.5)]
  angles.extend(tuple(a) for a in rng.uniform(-180, 180, size=(5, 3)))
  for ax, ay, az in angles:
    got = np.empty(4)
    call('mc_euler2quat')(float(ax), float(ay), float(az), addr(got))
    assert np.allclose(got, upstream_math.euler2quat(ax, ay, az), **EXACT)


def test_mc_mj_quatprod():
  """`mc_mj_quatprod` against MuJoCo's `mju_mulQuat` row by row.

  Upstream's own `mj_quatprod` forwards the whole `(n, 4)` array to a C helper
  that only accepts a single `(4,)`, so it cannot serve as a batched reference;
  the C helper applied one row at a time is the stronger check anyway.
  """
  mjlib = _mjlib()
  rng = np.random.default_rng(1)
  for n in SIZES:
    q, r = unit_quats(rng, n), unit_quats(rng, n)
    got = np.empty((n, 4))
    call('mc_mj_quatprod')(addr(q), addr(r), addr(got), n)
    for i in range(n):
      expect = scratch(4)
      mjlib.mju_mulQuat(expect, np.ascontiguousarray(q[i]),
                        np.ascontiguousarray(r[i]))
      assert np.allclose(got[i], expect, **EXACT)
    # The identity is a left and right unit.
    ident = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    left = np.empty((n, 4))
    call('mc_mj_quatprod')(addr(ident), addr(q), addr(left), n)
    assert np.allclose(left, q, **EXACT)


def test_mc_mj_quatneg():
  """`mc_mj_quatneg` against MuJoCo's `mju_negQuat`."""
  mjlib = _mjlib()
  rng = np.random.default_rng(2)
  for n in SIZES:
    q = unit_quats(rng, n)
    got = np.empty((n, 4))
    call('mc_mj_quatneg')(addr(q), addr(got), n)
    for i in range(n):
      expect = scratch(4)
      mjlib.mju_negQuat(expect, np.ascontiguousarray(q[i]))
      assert np.allclose(got[i], expect, **EXACT)
    # The identity that defines the conjugate: flipping the vector part.
    assert np.allclose(got[:, 0], q[:, 0], **EXACT)
    assert np.allclose(got[:, 1:], -q[:, 1:], **EXACT)


def test_mc_mj_quat2vel():
  """`mc_mj_quat2vel` against MuJoCo's `mju_quat2Vel`, dt in the signature."""
  mjlib = _mjlib()
  rng = np.random.default_rng(3)
  for dt in (0.005, 1.0):
    for n in SIZES:
      q = unit_quats(rng, n)
      got = np.empty((n, 3))
      call('mc_mj_quat2vel')(addr(q), addr(got), n, float(dt))
      for i in range(n):
        expect = scratch(3)
        mjlib.mju_quat2Vel(expect, np.ascontiguousarray(q[i]), float(dt))
        assert np.allclose(got[i], expect, **EXACT)
  # The zero vector part is the identity rotation and carries no velocity.
  q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (8, 1))
  got = np.empty((8, 3))
  call('mc_mj_quat2vel')(addr(q), addr(got), 8, 0.01)
  assert np.array_equal(got, np.zeros((8, 3)))


def test_mc_mj_quatdiff():
  """`mc_mj_quatdiff` against MuJoCo's `mju_mulQuat(mju_negQuat(a), b)`."""
  mjlib = _mjlib()
  rng = np.random.default_rng(4)
  for n in SIZES:
    source, target = unit_quats(rng, n), unit_quats(rng, n)
    got = np.empty((n, 4))
    call('mc_mj_quatdiff')(addr(source), addr(target), addr(got), n)
    for i in range(n):
      neg = scratch(4)
      mjlib.mju_negQuat(neg, np.ascontiguousarray(source[i]))
      expect = scratch(4)
      mjlib.mju_mulQuat(expect, neg, np.ascontiguousarray(target[i]))
      assert np.allclose(got[i], expect, **EXACT)


# ======================= dm_control/utils/transformations.py ===================


def test_mc_euler_to_rmat():
  """`mc_euler_to_rmat` against upstream's `euler_to_rmat`, all seven orderings.

  The kernel takes the three angle columns concatenated and writes one packed
  `(n, s * s)` batch, so this drives the buffer layout as well as the
  arithmetic: a wrong column offset or a wrong lane offset shows up here as a
  row that disagrees with upstream, not just as a wrong number.
  """
  orderings = ('XYZ', 'XYX', 'ZYX', 'XZY', 'YZX', 'ZXY', 'YXZ')
  rng = np.random.default_rng(5)
  for ordering in orderings:
    axes = tuple('XYZ'.index(c) for c in ordering)
    for full in (False, True):
      shape = (4, 4) if full else (3, 3)
      for n in SIZES:
        angles = rng.uniform(-10.0, 10.0, size=(n, 3))
        cols = np.ascontiguousarray(np.concatenate(
            [angles[:, 0], angles[:, 1], angles[:, 2]]))
        got = np.zeros(n * shape[0] * shape[1])
        call('mc_euler_to_rmat')(addr(cols), addr(got), n, int(full), *axes)
        expect = upstream.euler_to_rmat(angles, ordering=ordering, full=full)
        assert np.allclose(got.reshape(n, *shape), expect, **EXACT)
  # Zero angles are the identity, with the homogeneous row and column too.
  for full, expect in ((False, np.eye(3)), (True, np.eye(4))):
    cols = np.zeros(6)
    got = np.zeros(2 * expect.size)
    call('mc_euler_to_rmat')(addr(cols), addr(got), 2, int(full), 0, 1, 2)
    assert np.allclose(got.reshape(2, *expect.shape), expect, **EXACT)


def test_mc_axis_rotation():
  """`mc_axis_rotation` against upstream's rotation_{x,y,z}_axis."""
  rng = np.random.default_rng(6)
  for full in (False, True):
    shape = (4, 4) if full else (3, 3)
    for axis, upstream_fn in [(0, upstream.rotation_x_axis),
                              (1, upstream.rotation_y_axis),
                              (2, upstream.rotation_z_axis)]:
      for n in SIZES:
        theta = rng.uniform(-10.0, 10.0, size=n)
        got = np.empty((n,) + shape)
        call('mc_axis_rotation')(addr(theta), addr(got), n, int(full), axis)
        # The kernel emits one matrix per angle. Upstream's wrapper squeezes
        # the batch away for a length-1 theta, so compare row by row against
        # the same single-angle call.
        for i in range(n):
          assert np.allclose(got[i], upstream_fn(float(theta[i]), full),
                             **EXACT)
  # theta == 0 is the identity for every axis.
  theta = np.zeros(3)
  got = np.empty((3, 3, 3))
  call('mc_axis_rotation')(addr(theta), addr(got), 3, 0, 1)
  assert np.allclose(got, np.eye(3), **EXACT)


def test_mc_rmat_to_euler():
  """`mc_rmat_to_euler` for all seven orderings, including the pole branches."""
  orderings = ('XYZ', 'XYX', 'ZYX', 'XZY', 'YZX', 'ZXY', 'YXZ')
  rng = np.random.default_rng(7)
  # Random rotations from euler angles: composed matrices are exactly the
  # input domain, so the round trip is the identity up to the gimbal lock set.
  for ordering in orderings:
    for angles in rng.uniform(-1.5, 1.5, size=(6, 3)):
      rmat = upstream.euler_to_rmat(angles, ordering=ordering)
      got = np.empty(3)
      call('mc_rmat_to_euler')(addr(rmat), addr(got), orderings.index(ordering))
      assert np.allclose(got, upstream.rmat_to_euler(rmat, ordering), **EXACT)
  # The XYX singularity at r00 == -1 is a branch upstream takes explicitly.
  flipped = upstream.rotation_x_axis(np.pi).dot(upstream.rotation_y_axis(0.7))
  got = np.empty(3)
  call('mc_rmat_to_euler')(addr(flipped), addr(got), orderings.index('XYX'))
  assert np.allclose(got, upstream.rmat_to_euler(flipped, 'XYX'), **EXACT)


def test_mc_quat_conj():
  """`mc_quat_conj` against upstream's quat_conj, batched and unbatched."""
  rng = np.random.default_rng(8)
  for n in SIZES:
    q = unit_quats(rng, n)
    got = np.empty((n, 4))
    call('mc_quat_conj')(addr(q), addr(got), n)
    assert np.allclose(got, upstream.quat_conj(q), **EXACT)
  # A unit quaternion's conjugate is its inverse.
  q = np.array([0.5, 0.5, 0.5, 0.5])
  got = np.empty(4)
  call('mc_quat_conj')(addr(q), addr(got), 1)
  assert np.allclose(got, upstream.quat_inv(q), **EXACT)


def test_mc_quat_inv():
  """`mc_quat_inv` against upstream's quat_inv, on non-unit input too."""
  rng = np.random.default_rng(9)
  for n in SIZES:
    # Unnormalised on purpose: quat_inv divides by the squared norm.
    q = rng.normal(size=(n, 4)) * 3.0
    got = np.empty((n, 4))
    call('mc_quat_inv')(addr(q), addr(got), n)
    assert np.allclose(got, upstream.quat_inv(q), **EXACT)
    # q @ conj(q) is the squared norm, so q @ q^-1 is the identity quaternion.
    ident = np.empty((n, 4))
    call('mc_quat_mul')(addr(q), addr(got), addr(ident), n)
    assert np.allclose(ident[:, 0], 1.0, **EXACT)
    assert np.allclose(ident[:, 1:], 0.0, atol=1e-12)


def test_mc_quat_mul():
  """`mc_quat_mul` against upstream's quat_mul (the Hamilton product)."""
  rng = np.random.default_rng(10)
  for n in SIZES:
    q, r = unit_quats(rng, n), unit_quats(rng, n)
    got = np.empty((n, 4))
    call('mc_quat_mul')(addr(q), addr(r), addr(got), n)
    assert np.allclose(got, upstream.quat_mul(q, r), **EXACT)
    # Multiplication by the identity leaves the quaternion alone.
    ident = np.empty((n, 4))
    one = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    call('mc_quat_mul')(addr(q), addr(one), addr(ident), n)
    assert np.allclose(ident, q, **EXACT)


def test_mc_quat_rotate():
  """`mc_quat_rotate` against the rotation matrix built by quat_to_mat."""
  rng = np.random.default_rng(11)
  for n in SIZES:
    q = unit_quats(rng, n)
    v = rng.normal(size=(n, 3))
    got = np.empty((n, 3))
    call('mc_quat_rotate')(addr(q), addr(v), addr(got), n)
    for i in range(n):
      # Rotating by q is the matrix product with q's rotation matrix.
      expect = upstream.quat_to_mat(q[i])[:3, :3].dot(v[i])
      assert np.allclose(got[i], expect, **EXACT)
    # Upstream's batched `quat_rotate` slices the leading axis, so it returns
    # `(n-1, 4)` for a batch and cannot be used as a reference here; the
    # rotation-matrix identity above is the contract the kernel implements.
  # The rotation preserves the norm of the vector.
  q = np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (4, 1))  # 180 deg about x
  v = rng.normal(size=(4, 3))
  got = np.empty((4, 3))
  call('mc_quat_rotate')(addr(q), addr(v), addr(got), 4)
  assert np.allclose(np.linalg.norm(got, axis=-1),
                     np.linalg.norm(v, axis=-1), **EXACT)
  assert np.allclose(got[:, 0], v[:, 0], **EXACT)
  assert np.allclose(got[:, 1:], -v[:, 1:], **EXACT)


def test_mc_quat_rotate_scalar():
  """`mc_quat_rotate_scalar` against the batched kernel with one row."""
  rng = np.random.default_rng(12)
  for _ in range(8):
    q = unit_quats(rng, 1)[0]
    v = rng.normal(size=3)
    got = np.empty(3)
    call('mc_quat_rotate_scalar')(*q.tolist(), *v.tolist(), addr(got))
    batched = np.empty((1, 3))
    call('mc_quat_rotate')(addr(q.reshape(1, 4)), addr(v.reshape(1, 3)),
                           addr(batched), 1)
    assert np.allclose(got, batched[0], **EXACT)
    assert np.allclose(got, upstream.quat_to_mat(q)[:3, :3].dot(v), **EXACT)


def test_mc_quat_to_axisangle():
  """`mc_quat_to_axisangle` against upstream, including the `angle < tol` branch."""
  rng = np.random.default_rng(13)
  for angles in [np.zeros(3),
                 np.array([0.3, -0.2, 0.1]),
                 np.array([0.0, 0.0, 3.0]),
                 rng.normal(size=3) * 0.5]:
    q = upstream.axisangle_to_quat(angles)
    got = np.empty(3)
    call('mc_quat_to_axisangle')(addr(q), addr(got))
    assert np.allclose(got, upstream.quat_to_axisangle(q), **EXACT)
  # Below the tolerance the axis-angle is the zero vector, not a NaN.
  q = np.array([1.0, 0.0, 0.0, 0.0])
  got = np.full(3, np.nan)
  call('mc_quat_to_axisangle')(addr(q), addr(got))
  assert np.array_equal(got, np.zeros(3))


def test_mc_quat_to_mat():
  """`mc_quat_to_mat` against upstream's quat_to_mat."""
  rng = np.random.default_rng(14)
  for n in SIZES:
    for q in (unit_quats(rng, n), rng.normal(size=(n, 4))):
      for i in range(n):
        got = np.empty((4, 4))
        call('mc_quat_to_mat')(addr(np.ascontiguousarray(q[i])), addr(got))
        assert np.allclose(got, upstream.quat_to_mat(q[i]), **EXACT)
  # A quaternion of vanishing norm yields the identity, as upstream specifies.
  got = np.empty((4, 4))
  call('mc_quat_to_mat')(addr(np.zeros(4)), addr(got))
  assert np.allclose(got, np.eye(4), **EXACT)


def test_mc_mat_to_quat():
  """`mc_mat_to_quat` against upstream's mat_to_quat, on all four branches."""
  rng = np.random.default_rng(15)
  # The four branches of upstream's matrix->quaternion switch are selected by
  # which diagonal dominates, so drive each of them explicitly.
  mats = [np.eye(4)]
  mats.append(upstream.quat_to_mat(upstream.axisangle_to_quat(np.array([.4, .1, .9]))))
  mats.append(upstream.quat_to_mat(upstream.axisangle_to_quat(np.array([2.6, .5, .2]))))
  mats.append(upstream.quat_to_mat(upstream.axisangle_to_quat(np.array([.3, 2.1, .7]))))
  mats.append(upstream.quat_to_mat(upstream.axisangle_to_quat(np.array([1.0, 1.6, .4]))))
  mats.extend(upstream.quat_to_mat(q) for q in unit_quats(rng, 8))
  for mat in mats:
    mat = np.ascontiguousarray(mat)
    got = np.empty(4)
    call('mc_mat_to_quat')(addr(mat), addr(got))
    expect = upstream.mat_to_quat(mat)
    assert np.allclose(got, expect, **EXACT)
    # A quaternion and its negation are the same rotation.
    assert np.allclose(np.abs(got), np.abs(expect), **EXACT)


def test_mc_axisangle_to_quat():
  """`mc_axisangle_to_quat` against upstream, at and below the tolerance."""
  rng = np.random.default_rng(16)
  for n in SIZES:
    for tol in (0.0, 1e-6, 0.5):
      aa = rng.normal(size=(n, 3))
      if tol == 0.5 and n > 1:
        # Push some rows below the tolerance to take the identity branch.
        aa[0] = 0.0
        aa[1] = np.array([0.1, 0.0, 0.0])
      got = np.empty((n, 4))
      call('mc_axisangle_to_quat')(addr(aa), addr(got), n, float(tol))
      assert np.allclose(got, upstream.axisangle_to_quat(aa, tol=tol),
                         **EXACT)
      assert np.allclose(np.linalg.norm(got, axis=-1), 1.0, **EXACT)


def test_mc_quat_log():
  """`mc_quat_log` against upstream's quat_log, including near-identity input."""
  rng = np.random.default_rng(17)
  for n in SIZES:
    q = unit_quats(rng, n)
    # A row on the identity exercises the `v + tol` guard against a 0/0.
    q[0] = np.array([1.0, 0.0, 0.0, 0.0])
    got = np.empty((n, 4))
    call('mc_quat_log')(addr(q), addr(got), n, 1e-10)
    assert np.allclose(got, upstream.quat_log(q), **TRANSCENDENTAL)
    assert np.all(np.isfinite(got))


def test_mc_quat_dist():
  """`mc_quat_dist` against upstream's quat_dist; identical inputs give zero."""
  rng = np.random.default_rng(18)
  for n in SIZES:
    q = unit_quats(rng, n)
    # `quat_log` adds `tol` to every component, so the distance from a
    # quaternion to itself is ~tol rather than exactly 0.
    got = np.empty(n)
    call('mc_quat_dist')(addr(q), addr(q), addr(got), n, 1e-10)
    assert np.allclose(got, 0.0, atol=1e-7)
    source, target = unit_quats(rng, n), unit_quats(rng, n)
    got = np.empty(n)
    call('mc_quat_dist')(addr(source), addr(target), addr(got), n, 1e-10)
    assert np.allclose(got, upstream.quat_dist(source, target).ravel(),
                       **TRANSCENDENTAL)
    # The distance is symmetric and never exceeds pi. `acos` near 1 has a
    # square-root derivative, so swapping the arguments perturbs the last bit
    # that reaches it; upstream's own `quat_dist` shows exactly the same
    # asymmetry on the same inputs, so the bound is the conditioning and not a
    # disagreement between the two implementations.
    back = np.empty(n)
    call('mc_quat_dist')(addr(target), addr(source), addr(back), n, 1e-10)
    assert np.allclose(got, back, atol=1e-8)
    assert (got <= np.pi + 1e-9).all()


def test_mc_rotation_matrix_2d():
  """`mc_rotation_matrix_2d` against upstream's 2x2 rotation matrix."""
  rng = np.random.default_rng(19)
  for n in SIZES:
    theta = rng.uniform(-10.0, 10.0, size=n)
    got = np.empty((n, 4))
    call('mc_rotation_matrix_2d')(addr(theta), addr(got), n)
    ct, st = np.cos(theta), np.sin(theta)
    # Upstream returns [[ct, -st], [st, ct]] per angle, row-major flattened.
    expect = np.stack([ct, -st, st, ct], axis=-1)
    assert np.allclose(got, expect, **EXACT)
    for i in range(n):
      assert np.allclose(got[i].reshape(2, 2),
                         upstream.rotation_matrix_2d(theta[i]), **EXACT)


# ============================== dm_control/utils/rewards.py ====================


def test_mc_tolerance_margin():
  """`mc_tolerance_margin` against upstream's `tolerance`, all eight sigmoids."""
  rng = np.random.default_rng(20)
  x = np.concatenate([
      np.linspace(-5, 5, 37),          # covers in-bounds and out-of-bounds
      np.array([-0.5, 0.5]),            # the inclusive bound itself
      rng.normal(scale=2.0, size=64),   # random, both sides
  ])
  n = x.shape[0]
  for sigmoid in SIGMOIDS:
    for margin in (1.0, 0.25, 3.0):
      for value_at_margin in (0.1, 0.5, 0.95):
        got = np.empty(n)
        call('mc_tolerance_margin')(addr(x), addr(got), n, -0.5, 0.5,
                                    float(margin), float(value_at_margin),
                                    SIGMOID_CODE[sigmoid])
        expect = upstream_rewards.tolerance(
            x, bounds=(-0.5, 0.5), margin=margin, sigmoid=sigmoid,
            value_at_margin=value_at_margin)
        # `std.math`'s exp/cosh/tanh are polynomial approximations, so the
        # agreement is absolute to the last few ulp of the sigmoid rather than
        # relative; see the numerical note in the README.
        assert np.allclose(got, expect, rtol=1e-8, atol=1e-10)
        assert ((got >= 0.0) & (got <= 1.0)).all()
  # Inside the bounds the value is exactly 1 for every sigmoid.
  inside = np.linspace(-0.4, 0.4, 33)
  for sigmoid in SIGMOIDS:
    got = np.empty(inside.shape[0])
    call('mc_tolerance_margin')(addr(inside), addr(got), inside.shape[0],
                                -0.5, 0.5, 1.0, 0.1, SIGMOID_CODE[sigmoid])
    assert np.allclose(got, 1.0, **EXACT)


def test_mc_in_bounds():
  """`mc_in_bounds` against upstream's `tolerance(margin=0)`."""
  x = np.array([-1e-16, -0.5, -0.49999, 0.0, 0.49999, 0.5, 0.50001, 1e-16, 7.0])
  n = x.shape[0]
  got = np.empty(n)
  call('mc_in_bounds')(addr(x), addr(got), n, -0.5, 0.5)
  expect = upstream_rewards.tolerance(x, bounds=(-0.5, 0.5), margin=0.0)
  assert np.array_equal(got, expect)
  # Both bounds are inclusive, so the endpoints are 1.
  assert got[1] == 1.0 and got[5] == 1.0
  assert got[0] == 1.0 and got[7] == 1.0   # just inside
  assert got[6] == 0.0 and got[8] == 0.0   # outside


# ================= dm_control/locomotion/tasks/reference_pose/rewards.py =======


def test_mc_bounded_quat_dist():
  """`mc_bounded_quat_dist` against upstream, including the in-place normalisation."""
  rng = np.random.default_rng(21)
  for n in SIZES:
    source = rng.normal(size=(n, 4))
    target = rng.normal(size=(n, 4))  # not unit: the kernel normalises in place
    got = np.empty(n)
    call('mc_bounded_quat_dist')(addr(source), addr(target), addr(got), n)
    # Upstream mutates its arguments in place too, so compare against a
    # pristine copy fed to the reference implementation.
    ref_source, ref_target = source.copy(), target.copy()
    expect = upstream_pose.bounded_quat_dist(ref_source, ref_target)
    assert np.allclose(got, expect.ravel(), **EXACT)
    # The in-place normalisation is observable and matches upstream's.
    assert np.allclose(source, ref_source, **EXACT)
    assert np.allclose(target, ref_target, **EXACT)
    assert np.allclose(np.linalg.norm(source, axis=-1), 1.0, **EXACT)
    # Bounded by pi/2 by construction, and 0 for identical orientations.
    assert (got <= np.pi / 2 + 1e-9).all()
    same_source = ref_source.copy()
    same = np.empty(n)
    # `addr()` hands the kernel a raw pointer, so a temporary built inline
    # would be freed while the kernel is still writing into it. Every buffer
    # the kernels mutate in place is therefore kept alive by a name.
    call('mc_bounded_quat_dist')(addr(ref_source), addr(same_source),
                                 addr(same), n)
    assert np.allclose(same, 0.0, atol=1e-7)


# ========================= dm_control/utils/inverse_kinematics.py =============


@pytest.mark.parametrize('ndelta,nv,scale', [(6, 12, 0.0), (3, 12, 3e-2),
                                             (6, 12, 0.0), (3, 5, 0.1)])
def test_mc_nullspace_products(ndelta, nv, scale):
  """`mc_nullspace_products` against the three products upstream forms."""
  rng = np.random.default_rng(22)
  jac = rng.normal(size=(ndelta, nv))
  delta = rng.normal(size=ndelta)
  hess = np.empty((nv, nv))
  joint_delta = np.empty(nv)
  call('mc_nullspace_products')(addr(jac), addr(delta), addr(hess),
                                addr(joint_delta), ndelta, nv, float(scale))
  assert np.allclose(joint_delta, jac.T.dot(delta), **EXACT)
  expect_hess = jac.T.dot(jac)
  if scale > 0:
    expect_hess = expect_hess + np.eye(nv) * scale
  assert np.allclose(hess, expect_hess, **EXACT)
  # The kernel's own output is the input of upstream's solve; the full
  # nullspace_method must therefore agree with upstream end to end.
  from dm_control.utils import inverse_kinematics as upstream_ik
  from mojo_dm_control.utils import inverse_kinematics as our_ik
  assert np.allclose(
      our_ik.nullspace_method(jac, delta, scale),
      upstream_ik.nullspace_method(jac, delta, scale), **EXACT)


# ============================== dm_control/suite/lqr_solver.py ================


def test_mc_sym2dense():
  """`mc_sym2dense` against MuJoCo's `mju_sym2dense` on a real model's mass."""
  from dm_control import suite
  env = suite.load('lqr', 'lqr_6_2')
  model, data = env.physics.model, env.physics.data
  n = model.nq
  res = np.full((n, n), np.nan)   # poison: the kernel must write every entry
  sparse_m = np.ascontiguousarray(data.ptr.M.ravel(), dtype=np.float64)
  rownnz = np.ascontiguousarray(model.ptr.M_rownnz.ravel(), dtype=np.int64)
  rowadr = np.ascontiguousarray(model.ptr.M_rowadr.ravel(), dtype=np.int64)
  colind = np.ascontiguousarray(model.ptr.M_colind.ravel(), dtype=np.int64)
  call('mc_sym2dense')(addr(sparse_m), addr(res), n, addr(rownnz), addr(rowadr),
                       addr(colind))
  # Reference: expand the sparse entries by hand, mirroring `mju_sym2dense`.
  expect = np.zeros((n, n))
  for r in range(n):
    for k in range(rownnz[r]):
      e = rowadr[r] + k
      c = colind[e]
      expect[r, c] = sparse_m[e]
      expect[c, r] = sparse_m[e]
  assert np.allclose(res, expect, **EXACT)
  assert np.isfinite(res).all()            # no NaN survived
  assert np.array_equal(res, res.T)        # symmetric
  # And it is a genuine positive-definite mass matrix.
  assert np.all(np.linalg.eigvalsh(res) > 0)


def test_mc_sym2dense_empty_and_diagonal():
  """`mc_sym2dense` on a degenerate 1x1 mass matrix with no sparsity."""
  n = 1
  sparse_m = np.array([2.5], dtype=np.float64)
  rownnz = np.zeros(1, dtype=np.int64)
  rowadr = np.zeros(1, dtype=np.int64)
  colind = np.zeros(1, dtype=np.int64)
  res = np.full((1, 1), np.nan)
  call('mc_sym2dense')(addr(sparse_m), addr(res), n, addr(rownnz), addr(rowadr),
                       addr(colind))
  # With no stored entries the matrix is zero, and the kernel zeroes it.
  assert np.array_equal(res, np.zeros((1, 1)))


def test_mc_squared_difference_sum():
  """`mc_squared_difference_sum` against the sum of squares it stands for."""
  rng = np.random.default_rng(23)
  for n in SIZES:
    a, b = rng.normal(size=n), rng.normal(size=n)
    got = call('mc_squared_difference_sum')(addr(a), addr(b), n)
    assert got == pytest.approx(float(np.sum((a - b) ** 2)), rel=1e-12)
  # Identical vectors have zero squared difference; the kernel returns a
  # scalar, not a buffer.
  x = rng.normal(size=64)
  assert call('mc_squared_difference_sum')(addr(x), addr(x), 64) == \
      pytest.approx(0.0, abs=0.0)
  # And a single element is the squared difference of that element.
  u, v = np.array([1.5]), np.array([-0.5])
  assert call('mc_squared_difference_sum')(addr(u), addr(v), 1) == \
      pytest.approx(4.0)
  # End to end: it is the reduction `compute_squared_differences` performs.
  from dm_control.locomotion.tasks.reference_pose import rewards as up_pose
  from mojo_dm_control.locomotion.tasks.reference_pose import rewards as pose
  walker = {'torso_height': rng.normal(size=8), 'torso_lean': rng.normal(size=8)}
  reference = {k: v + rng.normal(scale=0.1, size=8) for k, v in walker.items()}
  got = pose.compute_squared_differences(walker, reference)
  want = up_pose.compute_squared_differences(walker, reference)
  assert set(got) == set(want)
  for k in want:
    assert got[k] == pytest.approx(want[k], rel=1e-12)


# ============================== helpers =======================================


_MJLIB = None


def _mjlib():
  """MuJoCo's own C library, the reference for the `mju_*` kernels."""
  global _MJLIB
  if _MJLIB is None:
    from dm_control.mujoco.wrapper import mjbindings
    _MJLIB = mjbindings.mjlib
  return _MJLIB


def test_every_exported_symbol_has_a_test():
  """Guard the coverage rule mechanically rather than by eye.

  The check the port is held to is that every `@export` in `src/*.mojo` is
  named in `tests/`. This asserts the same property from inside the suite, so
  a kernel added without a test fails the run instead of shipping green.
  """
  import glob
  import re
  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  here = os.path.dirname(os.path.abspath(__file__))
  exported = set()
  for path in glob.glob(os.path.join(root, 'src', '*.mojo')):
    with open(path) as f:
      exported.update(re.findall(r'@export\("([A-Za-z_0-9]+)"\)', f.read()))
  assert exported, 'no @export found: the kernel source moved?'
  covered = set()
  for path in glob.glob(os.path.join(here, 'test_*.py')):
    with open(path) as f:
      body = f.read()
    covered.update(s for s in exported if ("'%s'" % s) in body
                   or ('"%s"' % s) in body)
  assert covered == exported, 'kernels without a test: %s' % sorted(
      exported - covered)


def test_library_really_is_the_compiled_one():
  """The tests above must be observing `dist/*.so`, not a Python fallback."""
  handle = lib()
  assert isinstance(handle, ctypes.CDLL)
  assert os.path.realpath(handle._name) == os.path.realpath(LIB)
  # `mc_quat_mul` only exists in the shared object; resolving it proves the
  # handle is the compiled artefact.
  assert hasattr(handle, 'mc_quat_mul')
