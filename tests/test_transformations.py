"""Parity of mojo_dm_control.utils.transformations against upstream dm_control.

`std.math.exp`/`cosh` are polynomial approximations accurate to about 4e-12
relative and `log` to about 1e-10, so the two transcendental-bearing functions
(`quat_log`, `quat_dist`) get a looser tolerance than the rest.
"""

import numpy as np
import pytest

from dm_control.utils import transformations as upstream
from mojo_dm_control.utils import transformations as tr

EXACT = dict(rtol=1e-12, atol=1e-12)
TRANSCENDENTAL = dict(rtol=1e-9, atol=1e-9)

ORDERINGS = ('XYZ', 'XYX', 'ZYX', 'XZY', 'YZX', 'ZXY', 'YXZ')


@pytest.fixture
def quats():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(7, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


@pytest.fixture
def axis_angles():
    return np.random.default_rng(1).normal(size=(5, 3))


def assert_same(ours, theirs, **tol):
    np.testing.assert_allclose(ours, theirs, **tol)


# --------------------------------------------------------------- quaternions


def test_quat_conj(quats):
    assert_same(tr.quat_conj(quats), upstream.quat_conj(quats), **EXACT)
    assert_same(tr.quat_conj(quats[0]), upstream.quat_conj(quats[0]), **EXACT)


def test_quat_inv(quats):
    assert_same(tr.quat_inv(quats), upstream.quat_inv(quats), **EXACT)
    # An inverse really inverts.
    assert_same(tr.quat_mul(quats, tr.quat_inv(quats)),
                np.broadcast_to([1.0, 0.0, 0.0, 0.0], quats.shape), **EXACT)


def test_quat_mul_broadcast(quats):
    assert_same(tr.quat_mul(quats, quats), upstream.quat_mul(quats, quats), **EXACT)
    # `qmat @ quat2[..., None]` broadcasts the batch dimensions upstream too.
    assert_same(tr.quat_mul(quats, quats[0]),
                upstream.quat_mul(quats, quats[0]), **EXACT)
    assert_same(tr.quat_mul(quats[:3], quats[3:6]),
                upstream.quat_mul(quats[:3], quats[3:6]), **EXACT)


def test_quat_diff(quats):
    assert_same(tr.quat_diff(quats, quats[0]),
                upstream.quat_diff(quats, quats[0]), **EXACT)


def test_quat_log(quats):
    assert_same(tr.quat_log(quats), upstream.quat_log(quats), **TRANSCENDENTAL)


def test_quat_log_rejects_unnormalized(quats):
    """Upstream raises, but its error message cannot format an array."""
    bad = quats.copy()
    bad[0, 0] = 2.0
    with pytest.raises(Exception):
        tr.quat_log(bad)
    with pytest.raises(Exception):
        upstream.quat_log(bad.copy())


def test_quat_dist(quats):
    assert_same(tr.quat_dist(quats, quats), upstream.quat_dist(quats, quats),
                **TRANSCENDENTAL)
    assert_same(tr.quat_dist(quats[:3], quats[3:6]),
                upstream.quat_dist(quats[:3], quats[3:6]), **TRANSCENDENTAL)
    # A quaternion and its negation are the same rotation, but upstream's
    # `tol`-regularised `log` is ill-conditioned there, so this checks the
    # invariant rather than bit-parity.
    assert np.allclose(tr.quat_dist(quats, -quats), 0.0, atol=1e-5)


def test_quat_rotate(quats, axis_angles):
    vec = axis_angles[0]
    assert_same(tr.quat_rotate(quats[0], vec), upstream.quat_rotate(quats[0], vec),
                **EXACT)
    # The identity quaternion leaves the vector untouched.
    assert_same(tr.quat_rotate(np.array([1.0, 0, 0, 0]), vec), vec, **EXACT)
    # Rotation preserves length.
    rotated = tr.quat_rotate(quats, vec)
    assert np.allclose(np.linalg.norm(rotated, axis=-1), np.linalg.norm(vec))


def test_axisangle_round_trip(axis_angles):
    quat = tr.axisangle_to_quat(axis_angles)
    assert_same(quat, upstream.axisangle_to_quat(axis_angles), **EXACT)
    assert_same(tr.quat_to_axisangle(quat[0]),
                upstream.quat_to_axisangle(quat[0]), **EXACT)
    assert_same(tr.axisangle_to_quat(axis_angles[0]),
                upstream.axisangle_to_quat(axis_angles[0]), **EXACT)


def test_axisangle_below_tol_is_identity(axis_angles):
    tiny = axis_angles * 0.0
    assert_same(tr.axisangle_to_quat(tiny, tol=1.0),
                upstream.axisangle_to_quat(tiny, tol=1.0), **EXACT)
    assert_same(tr.axisangle_to_quat(tiny), upstream.axisangle_to_quat(tiny),
                **EXACT)


def test_quat_to_mat_and_back(quats):
    assert_same(tr.quat_to_mat(quats[0]), upstream.quat_to_mat(quats[0]), **EXACT)
    # A vanishing quaternion falls back to the identity.
    assert_same(tr.quat_to_mat(np.zeros(4)), np.identity(4), **EXACT)
    assert_same(tr.mat_to_quat(tr.quat_to_mat(quats[0])),
                upstream.mat_to_quat(upstream.quat_to_mat(quats[0])), **EXACT)
    # `mat_to_quat` promotes a 3x3 to a 4x4 first.
    rot = tr.quat_to_mat(quats[0])[0:3, 0:3]
    assert_same(tr.mat_to_quat(rot), upstream.mat_to_quat(rot), **EXACT)


# ---------------------------------------------------------- axis rotations


@pytest.mark.parametrize('name', ['rotation_x_axis', 'rotation_y_axis',
                                  'rotation_z_axis'])
def test_rotation_axis(name, axis_angles):
    ours = getattr(tr, name)
    theirs = getattr(upstream, name)
    for theta in (0.37, axis_angles[:, 0]):
        assert_same(ours(theta), theirs(theta), **EXACT)
        assert_same(ours(theta, full=True), theirs(theta, full=True), **EXACT)
    # Orthogonality and unit determinant.
    r = ours(axis_angles[:, 0])
    assert np.allclose(r @ r.transpose(0, 2, 1), np.identity(3), atol=1e-12)
    assert np.allclose(np.linalg.det(r), 1.0, atol=1e-12)


# -------------------------------------------------------------- euler angles


@pytest.mark.parametrize('ordering', ORDERINGS)
def test_euler_to_rmat(ordering, axis_angles):
    assert_same(tr.euler_to_rmat(axis_angles, ordering),
                upstream.euler_to_rmat(axis_angles, ordering), **EXACT)
    assert_same(tr.euler_to_rmat(axis_angles[0], ordering),
                upstream.euler_to_rmat(axis_angles[0], ordering), **EXACT)
    assert_same(tr.euler_to_rmat(axis_angles, ordering, full=True),
                upstream.euler_to_rmat(axis_angles, ordering, full=True), **EXACT)


@pytest.mark.parametrize('ordering', ORDERINGS)
def test_rmat_to_euler(ordering, axis_angles):
    ours = tr.euler_to_rmat(axis_angles, ordering)
    theirs = upstream.euler_to_rmat(axis_angles, ordering)
    for i in range(axis_angles.shape[0]):
        assert_same(tr.rmat_to_euler(ours[i], ordering),
                    upstream.rmat_to_euler(theirs[i], ordering), **EXACT)


@pytest.mark.parametrize('ordering', ORDERINGS)
def test_rmat_to_euler_at_poles(ordering):
    """The pole branches are where the ordering matters most."""
    if ordering == 'XYX':
        poles = [np.array([0.0, np.pi, 0.0]), np.array([0.0, 0.0, 0.0]),
                 np.array([0.3, np.pi, -0.7])]
    else:
        poles = [np.array([0.0, np.pi / 2, 0.4]), np.array([0.0, -np.pi / 2, -1.1]),
                 np.array([0.3, 0.0, 0.7])]
    for euler in poles:
        rmat = tr.euler_to_rmat(euler, ordering)
        theirs = upstream.euler_to_rmat(euler, ordering)
        assert_same(tr.rmat_to_euler(rmat, ordering),
                    upstream.rmat_to_euler(theirs, ordering), **EXACT)


def test_euler_to_quat(axis_angles):
    for euler in axis_angles:
        assert_same(tr.euler_to_quat(euler, 'ZXZ'),
                    upstream.euler_to_quat(euler, 'ZXZ'), **EXACT)
        assert_same(tr.euler_to_quat(euler, 'XYZ'),
                    upstream.euler_to_quat(euler, 'XYZ'), **EXACT)


def test_quat_to_euler(quats):
    for q in quats:
        for ordering in ORDERINGS:
            assert_same(tr.quat_to_euler(q, ordering),
                        upstream.quat_to_euler(q, ordering), **EXACT)


def test_euler_ordering_changes_the_rotation(axis_angles):
    a = tr.euler_to_rmat(axis_angles[0], 'XYZ')
    b = tr.euler_to_rmat(axis_angles[0], 'ZXZ')
    assert not np.allclose(a, b)


# ------------------------------------------------------------------- 2D


def test_rotation_matrix_2d(axis_angles):
    assert_same(tr.rotation_matrix_2d(axis_angles[:, 0]),
                upstream.rotation_matrix_2d(axis_angles[:, 0]), **EXACT)
    assert_same(tr.rotation_matrix_2d(0.7), upstream.rotation_matrix_2d(0.7),
                **EXACT)
    assert tr.rotation_matrix_2d(0.7).shape == (2, 2)
