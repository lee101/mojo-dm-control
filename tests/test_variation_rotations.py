"""Parity of mojo_dm_control.composer.variation.rotations against upstream."""

import numpy as np
import pytest

from dm_control.composer.variation import rotations as upstream
from mojo_dm_control.composer.variation import rotations

TOL = dict(rtol=1e-12, atol=1e-12)


def test_identity_quaternion():
    np.testing.assert_array_equal(rotations.IDENTITY_QUATERNION,
                                  upstream.IDENTITY_QUATERNION)


def test_uniform_quaternion():
    for seed in range(8):
        ours = rotations.UniformQuaternion()(
            random_state=np.random.RandomState(seed))
        theirs = upstream.UniformQuaternion()(
            random_state=np.random.RandomState(seed))
        np.testing.assert_allclose(ours, theirs, **TOL)


def test_uniform_quaternion_is_unit():
    samples = np.array([rotations.UniformQuaternion()(
        random_state=np.random.RandomState(s)) for s in range(200)])
    np.testing.assert_allclose(np.linalg.norm(samples, axis=-1), 1.0, **TOL)
    # Shoemake's construction puts the real part on [-1, 1] with both signs.
    assert samples[:, 0].min() < -0.9
    assert samples[:, 0].max() > 0.9
    assert abs(samples.mean()) < 0.1


def test_quaternion_from_axis_angle():
    axis = np.array([0.0, 1.0, 0.0])
    angle = 0.7
    ours = rotations.QuaternionFromAxisAngle(axis, angle)(
        random_state=np.random.RandomState(0))
    theirs = upstream.QuaternionFromAxisAngle(axis, angle)(
        random_state=np.random.RandomState(0))
    np.testing.assert_allclose(ours, theirs, **TOL)
    np.testing.assert_allclose(np.linalg.norm(ours), 1.0, **TOL)


def test_quaternion_from_axis_angle_with_variations():
    ours = rotations.QuaternionFromAxisAngle(
        lambda *a: np.array([1.0, 0.0, 0.0]),
        lambda *a: 0.3)(random_state=np.random.RandomState(0))
    theirs = upstream.QuaternionFromAxisAngle(
        lambda *a: np.array([1.0, 0.0, 0.0]),
        lambda *a: 0.3)(random_state=np.random.RandomState(0))
    np.testing.assert_allclose(ours, theirs, **TOL)


def test_quaternion_pre_multiply():
    quat = np.array([0.8, 0.1, 0.2, 0.3])
    initial = np.array([0.5, 0.5, 0.5, 0.5])
    for cumulative in (False, True):
        current = np.array([0.9, 0.1, 0.2, 0.3]) if cumulative else None
        ours = rotations.QuaternionPreMultiply(quat, cumulative)(
            initial_value=initial, current_value=current,
            random_state=np.random.RandomState(0))
        theirs = upstream.QuaternionPreMultiply(quat, cumulative)(
            initial_value=initial, current_value=current,
            random_state=np.random.RandomState(0))
        np.testing.assert_allclose(ours, theirs, **TOL)


def test_quaternion_rotate():
    quat = np.array([0.6, 0.2, -0.5, 0.4])
    vec = np.array([1.0, -2.0, 0.5])
    ours = rotations.QuaternionRotate(quat, vec)(
        random_state=np.random.RandomState(0))
    theirs = upstream.QuaternionRotate(quat, vec)(
        random_state=np.random.RandomState(0))
    np.testing.assert_allclose(ours, theirs, **TOL)


def test_quaternion_rotate_uses_current_value_when_cumulative():
    quat = np.array([0.6, 0.2, -0.5, 0.4])
    initial = np.array([1.0, 0.0, 0.0])
    current = np.array([0.0, 1.0, 0.0])
    ours = rotations.QuaternionRotate(quat, cumulative=True)(
        initial_value=initial, current_value=current,
        random_state=np.random.RandomState(0))
    theirs = upstream.QuaternionRotate(quat, cumulative=True)(
        initial_value=initial, current_value=current,
        random_state=np.random.RandomState(0))
    np.testing.assert_allclose(ours, theirs, **TOL)
    non_cumulative = rotations.QuaternionRotate(quat)(
        initial_value=initial, current_value=current,
        random_state=np.random.RandomState(0))
    assert not np.allclose(ours, non_cumulative)


@pytest.mark.parametrize('name,args', [
    ('UniformQuaternion', ()),
    ('QuaternionFromAxisAngle', (1, 2)),
    ('QuaternionPreMultiply', (1, True)),
    ('QuaternionRotate', (1, 2, True)),
])
def test_equality_and_repr(name, args):
    ours = getattr(rotations, name)(*args)
    theirs = getattr(upstream, name)(*args)
    assert repr(ours) == repr(theirs)
    assert (ours == ours) and (ours != theirs.__class__.__mro__[0].__call__)


def test_variation_algebra():
    """`base.Variation` operators route through the ported transforms."""
    quat = np.array([0.6, 0.2, -0.5, 0.4])
    initial = np.array([1.0, 0.0, 0.0])
    single = rotations.QuaternionRotate(quat)(
        initial_value=initial, current_value=initial,
        random_state=np.random.RandomState(0))
    doubled = (rotations.QuaternionRotate(quat) * 2.0)(
        initial_value=initial, current_value=initial,
        random_state=np.random.RandomState(0))
    np.testing.assert_allclose(doubled, 2.0 * single, **TOL)
