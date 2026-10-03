"""Coverage for the vector and worker paths.

Every kernel that walks memory in width-W vectors has a scalar tail, and the
transcendental kernels also run across workers once the batch clears
`PARALLEL_MIN` rows. Both are invisible at the handful of rows the parity tests
use, so they are exercised here directly: sizes straddling the vector width and
the parallel threshold, checked against the same functions at a size small
enough to stay on the serial, single-vector path.
"""

import numpy as np
import pytest

from dm_control.locomotion.tasks.reference_pose import rewards as upstream_pose
from dm_control.utils import rewards as upstream_rewards
from dm_control.utils import transformations as upstream
from mojo_dm_control.locomotion.tasks.reference_pose import rewards as pose
from mojo_dm_control.utils import rewards
from mojo_dm_control.utils import transformations as tr

# `simd_width_of[DType.float64]()` is 4 on this target; the odd sizes around
# each of these are what put an element in the scalar tail.
VECTOR_WIDTH = 4

# `PARALLEL_MIN` in src/ported.mojo, and the same value the ctypes layer sees.
PARALLEL_MIN = 1 << 18

EXACT = dict(rtol=1e-12, atol=1e-12)
EXPONENTIAL_TOL = dict(rtol=1e-8, atol=1e-10)

# Lengths that cover an empty batch, a partial vector, an exact vector, and a
# vector plus a tail, for widths 1, 2, 4 and 8.
TAIL_SIZES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64, 65, 1000)


def unit_quats(rng, n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


@pytest.fixture(scope='module')
def quats_1m():
    rng = np.random.default_rng(21)
    return unit_quats(rng, PARALLEL_MIN + 4096)


# ------------------------------------------------------------ SIMD scalar tails


@pytest.mark.parametrize('n', TAIL_SIZES)
def test_rotation_matrix_2d_vector_tail(n):
    theta = np.random.default_rng(3).normal(size=n)
    np.testing.assert_allclose(tr.rotation_matrix_2d(theta),
                               upstream.rotation_matrix_2d(theta), **EXACT)


@pytest.mark.parametrize('n', TAIL_SIZES)
@pytest.mark.parametrize('axis', ('rotation_x_axis', 'rotation_y_axis',
                                  'rotation_z_axis'))
def test_axis_rotation_vector_tail(axis, n):
    theta = np.random.default_rng(4).normal(size=n)
    for full in (False, True):
        np.testing.assert_allclose(getattr(tr, axis)(theta, full=full),
                                   getattr(upstream, axis)(theta, full=full),
                                   **EXACT)


@pytest.mark.parametrize('n', TAIL_SIZES)
def test_quat_log_vector_tail(n):
    quats = unit_quats(np.random.default_rng(5), n)
    np.testing.assert_allclose(tr.quat_log(quats), upstream.quat_log(quats),
                               **dict(rtol=1e-9, atol=1e-9))


@pytest.mark.parametrize('n', TAIL_SIZES)
@pytest.mark.parametrize('sigmoid', ('gaussian', 'hyperbolic', 'long_tail',
                                     'reciprocal', 'cosine', 'linear',
                                     'quadratic', 'tanh_squared'))
def test_tolerance_vector_tail(n, sigmoid):
    x = np.random.default_rng(6).normal(size=n) * 1.5
    tol = EXPONENTIAL_TOL if sigmoid in ('gaussian', 'hyperbolic',
                                        'tanh_squared') else EXACT
    np.testing.assert_allclose(
        rewards.tolerance(x, bounds=(-0.2, 0.2), margin=0.7, sigmoid=sigmoid),
        upstream_rewards.tolerance(x, bounds=(-0.2, 0.2), margin=0.7,
                                   sigmoid=sigmoid), **tol)
    np.testing.assert_allclose(
        rewards.tolerance(x, bounds=(-0.2, 0.2)),
        upstream_rewards.tolerance(x, bounds=(-0.2, 0.2)), **EXACT)


@pytest.mark.parametrize('n', TAIL_SIZES)
@pytest.mark.parametrize('ordering', ('XYZ', 'XYX', 'ZYX', 'XZY', 'YZX', 'ZXY',
                                      'YXZ'))
def test_euler_to_rmat_vector_tail(n, ordering):
    angles = np.random.default_rng(17).normal(size=(n, 3))
    for full in (False, True):
        np.testing.assert_allclose(tr.euler_to_rmat(angles, ordering, full=full),
                                   upstream.euler_to_rmat(angles, ordering,
                                                         full=full), **EXACT)


@pytest.mark.parametrize('n', TAIL_SIZES)
def test_bounded_quat_dist_vector_tail(n):
    rng = np.random.default_rng(7)
    source, target = unit_quats(rng, n), unit_quats(rng, n)
    np.testing.assert_allclose(
        pose.bounded_quat_dist(source.copy(), target.copy()),
        upstream_pose.bounded_quat_dist(source.copy(), target.copy()), **EXACT)


# ------------------------------------------------------------------- the workers


def test_axis_rotation_parallel_matches_upstream(quats_1m):
    theta = np.ascontiguousarray(
        np.random.default_rng(8).normal(size=quats_1m.shape[0]))
    for axis in ('rotation_x_axis', 'rotation_y_axis', 'rotation_z_axis'):
        for full in (False, True):
            ours = getattr(tr, axis)(theta, full=full)
            assert ours.shape[0] == theta.shape[0]
            np.testing.assert_allclose(ours[::997],
                                       getattr(upstream, axis)(theta[::997],
                                                               full=full),
                                       **EXACT)


def test_rotation_matrix_2d_parallel_matches_upstream(quats_1m):
    theta = np.ascontiguousarray(
        np.random.default_rng(9).normal(size=quats_1m.shape[0]))
    np.testing.assert_allclose(tr.rotation_matrix_2d(theta)[::997],
                               upstream.rotation_matrix_2d(theta)[::997],
                               **EXACT)


def test_quat_log_parallel_matches_upstream(quats_1m):
    np.testing.assert_allclose(tr.quat_log(quats_1m)[::997],
                               upstream.quat_log(quats_1m)[::997],
                               **dict(rtol=1e-9, atol=1e-9))


def test_bounded_quat_dist_parallel_matches_upstream(quats_1m):
    n = quats_1m.shape[0]
    target = unit_quats(np.random.default_rng(10), n)
    ours = pose.bounded_quat_dist(quats_1m.copy(), target.copy())
    assert ours.shape == (n, 1)
    np.testing.assert_allclose(ours[::997],
                               upstream_pose.bounded_quat_dist(
                                   quats_1m[::997].copy(),
                                   target[::997].copy()), **EXACT)


def test_euler_to_rmat_parallel_matches_upstream(quats_1m):
    n = quats_1m.shape[0]
    angles = np.ascontiguousarray(
        np.random.default_rng(18).normal(size=(n, 3)))
    assert angles.shape[0] >= PARALLEL_MIN
    for ordering in ('XYZ', 'ZYX'):
        ours = tr.euler_to_rmat(angles, ordering)
        np.testing.assert_allclose(ours[::997],
                                   upstream.euler_to_rmat(angles[::997],
                                                         ordering), **EXACT)
    np.testing.assert_allclose(
        tr.euler_to_rmat(angles, 'XYZ', full=True)[::997],
        upstream.euler_to_rmat(angles[::997], 'XYZ', full=True), **EXACT)


def test_parallel_and_serial_agree_across_the_threshold():
    """One input just below `PARALLEL_MIN` and one above it must agree."""
    n = PARALLEL_MIN - 1
    rng = np.random.default_rng(11)
    source, target = unit_quats(rng, n + 1), unit_quats(rng, n + 1)
    theta = np.ascontiguousarray(rng.normal(size=n + 1))
    angles = np.ascontiguousarray(rng.normal(size=(n + 1, 3)))
    assert source[:n].shape[0] < PARALLEL_MIN <= source.shape[0]

    np.testing.assert_allclose(tr.quat_log(source[:n].copy()),
                               tr.quat_log(source.copy())[:n])
    np.testing.assert_allclose(tr.rotation_matrix_2d(theta[:n]),
                               tr.rotation_matrix_2d(theta)[:, :, :n])
    # The same input straddling the threshold must give the same matrices
    # whether the serial or the worker path ran.
    for ordering in ('XYZ', 'ZYX'):
        np.testing.assert_allclose(
            tr.euler_to_rmat(angles[:n], ordering),
            tr.euler_to_rmat(angles, ordering)[:n], **EXACT)
    # `bounded_quat_dist` normalises its arguments in place, so each call
    # needs its own copy of the same rows.
    serial = pose.bounded_quat_dist(source[:n].copy(), target[:n].copy())
    parallel = pose.bounded_quat_dist(source.copy(), target.copy())[:n]
    np.testing.assert_allclose(serial, parallel)
    for axis in ('rotation_x_axis', 'rotation_y_axis', 'rotation_z_axis'):
        np.testing.assert_allclose(getattr(tr, axis)(theta[:n]),
                                   getattr(tr, axis)(theta)[:n])


def test_vector_body_and_scalar_tail_agree_exactly():
    """A whole vector body and a one-element scalar tail must give the same
    float64, not merely a close one: the transcendental SIMD functions in this
    toolchain are the same code as the scalar ones."""
    x = np.random.default_rng(12).normal(size=64) * 1.5
    for sigmoid in ('gaussian', 'hyperbolic', 'long_tail', 'reciprocal',
                    'cosine', 'linear', 'quadratic', 'tanh_squared'):
        vector = rewards.tolerance(x, bounds=(-0.2, 0.2), margin=0.7,
                                   sigmoid=sigmoid)
        scalar = np.array([
            rewards.tolerance(x[i:i + 1], bounds=(-0.2, 0.2), margin=0.7,
                              sigmoid=sigmoid)[0] for i in range(x.shape[0])])
        assert np.array_equal(vector, scalar), sigmoid

    theta = np.random.default_rng(13).normal(size=64)
    vector = tr.rotation_matrix_2d(theta)
    scalar = np.stack(
        [tr.rotation_matrix_2d(theta[i:i + 1])[:, :, 0] for i in range(64)],
        axis=-1)
    assert np.array_equal(vector, scalar)

    angles = np.random.default_rng(14).normal(size=(64, 3))
    for ordering in ('XYZ', 'ZYX', 'YXZ'):
        vector = tr.euler_to_rmat(angles, ordering)
        scalar = np.stack(
            [tr.euler_to_rmat(angles[i], ordering) for i in range(64)])
        assert np.array_equal(vector, scalar), ordering
