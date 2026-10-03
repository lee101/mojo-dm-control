"""Parity of mojo_dm_control.mujoco.math against upstream.

Upstream's `mj_quatprod` and friends hand a single `(4,)` quaternion to
MuJoCo's `mju_*` C helpers, so their parity is checked against those helpers.
The Mojo port loops over a batch; the batched path is checked against the
NumPy Hamilton product, which is what `mju_mulQuat` computes.
"""

import numpy as np
import pytest

from dm_control.mujoco import math as upstream
from dm_control.mujoco.wrapper import mjbindings
from mojo_dm_control.mujoco import math as mjmath

mjlib = mjbindings.mjlib

TOL = dict(rtol=1e-12, atol=1e-12)


@pytest.fixture
def quats():
    rng = np.random.default_rng(3)
    q = rng.normal(size=(6, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def hamilton(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w1, i1, j1, k1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, i2, j2, k2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1 * w2 - i1 * i2 - j1 * j2 - k1 * k2,
        w1 * i2 + i1 * w2 + j1 * k2 - k1 * j2,
        w1 * j2 - i1 * k2 + j1 * w2 + k1 * i2,
        w1 * k2 + i1 * j2 - j1 * i2 + k1 * w2,
    ], axis=-1)


@pytest.mark.parametrize('angles', [(0.0, 0.0, 0.0), (10.0, 20.0, 30.0),
                                    (-173.0, 91.5, 4.25)])
def test_euler2quat(angles):
    np.testing.assert_allclose(mjmath.euler2quat(*angles),
                               upstream.euler2quat(*angles), **TOL)


def test_euler2quat_is_unit(quats):
    q = mjmath.euler2quat(*(np.degrees([0.3, -1.2, 2.9])))
    assert np.isclose(np.linalg.norm(q), 1.0)


def test_mj_quatprod_matches_mujoco(quats):
    for a, b in zip(quats, quats[::-1]):
        np.testing.assert_allclose(mjmath.mj_quatprod(a, b),
                                   upstream.mj_quatprod(a, b), **TOL)


def test_mj_quatprod_batched(quats):
    np.testing.assert_allclose(mjmath.mj_quatprod(quats, quats[::-1]),
                               hamilton(quats, quats[::-1]), **TOL)
    # The identity is a left and right unit.
    identity = np.broadcast_to([1.0, 0.0, 0.0, 0.0], quats.shape)
    np.testing.assert_allclose(mjmath.mj_quatprod(quats, identity), quats, **TOL)
    np.testing.assert_allclose(mjmath.mj_quatprod(identity, quats), quats, **TOL)


def test_mj_quatprod_matches_mju_mulquat(quats):
    for a, b in zip(quats, quats[::-1]):
        expected = np.zeros(4)
        mjlib.mju_mulQuat(expected, np.ascontiguousarray(a),
                          np.ascontiguousarray(b))
        np.testing.assert_allclose(mjmath.mj_quatprod(a, b), expected, **TOL)


def test_mj_quatneg(quats):
    for q in quats:
        np.testing.assert_allclose(mjmath.mj_quatneg(q), upstream.mj_quatneg(q),
                                   **TOL)
    np.testing.assert_allclose(mjmath.mj_quatneg(quats)[:, 0], quats[:, 0], **TOL)


@pytest.mark.parametrize('dt', [0.25, 1.0, -0.5])
def test_mj_quat2vel(quats, dt):
    for q in quats:
        np.testing.assert_allclose(mjmath.mj_quat2vel(q, dt),
                                   upstream.mj_quat2vel(q, dt), **TOL)


def test_mj_quat2vel_identity(quats):
    np.testing.assert_allclose(
        mjmath.mj_quat2vel(np.array([1.0, 0.0, 0.0, 0.0]), 0.25),
        upstream.mj_quat2vel(np.array([1.0, 0.0, 0.0, 0.0]), 0.25), **TOL)


def test_mj_quatdiff(quats):
    for a, b in zip(quats, quats[::-1]):
        np.testing.assert_allclose(mjmath.mj_quatdiff(a, b),
                                   upstream.mj_quatdiff(a, b), **TOL)
    np.testing.assert_allclose(mjmath.mj_quatdiff(quats, quats),
                               np.tile([1.0, 0.0, 0.0, 0.0], (6, 1)), **TOL)


def test_shapes(quats):
    assert mjmath.mj_quatprod(quats[0], quats[1]).shape == (4,)
    assert mjmath.mj_quatprod(quats, quats[::-1]).shape == (6, 4)
    assert mjmath.mj_quat2vel(quats, 0.5).shape == (6, 3)
    assert mjmath.mj_quatneg(quats[0]).shape == (4,)
    assert mjmath.mj_quatdiff(quats, quats[::-1]).shape == (6, 4)
