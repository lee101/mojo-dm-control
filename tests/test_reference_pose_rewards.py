"""Parity of the reference-pose reward functions against upstream."""

import copy

import numpy as np
import pytest

from dm_control.locomotion.tasks.reference_pose import rewards as upstream
from mojo_dm_control.locomotion.tasks.reference_pose import rewards

TOL = dict(rtol=1e-10, atol=1e-12)


@pytest.fixture
def quats():
    rng = np.random.default_rng(4)
    q = rng.normal(size=(6, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


@pytest.fixture
def features(quats):
    rng = np.random.default_rng(5)
    walker = {
        'center_of_mass': rng.normal(size=3),
        'joints_velocity': rng.normal(size=7),
        'appendages': rng.normal(size=(4, 3)),
        'body_quaternions': quats[:2],
    }
    reference = {
        'center_of_mass': rng.normal(size=3),
        'joints_velocity': rng.normal(size=7),
        'appendages': rng.normal(size=(4, 3)),
        'body_quaternions': quats[4:],
    }
    return walker, reference


def test_bounded_quat_dist(quats):
    ours = rewards.bounded_quat_dist(quats[:3].copy(), quats[2:5].copy())
    theirs = upstream.bounded_quat_dist(quats[:3].copy(), quats[2:5].copy())
    np.testing.assert_allclose(ours, theirs, **TOL)
    assert ours.shape == theirs.shape == (3, 1)


def test_bounded_quat_dist_normalises_in_place(quats):
    """Upstream divides its arguments by their norms; the port does too."""
    a, b = quats[:3].copy(), quats[2:5].copy()
    rewards.bounded_quat_dist(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=-1), 1.0, **TOL)
    np.testing.assert_allclose(np.linalg.norm(b, axis=-1), 1.0, **TOL)


def test_bounded_quat_dist_is_bounded(quats):
    d = rewards.bounded_quat_dist(quats.copy(), quats[::-1].copy())
    assert np.all(d >= 0.0)
    assert np.all(d <= np.pi / 2 + 1e-12)
    # Identical rotations are at distance zero.
    np.testing.assert_allclose(
        rewards.bounded_quat_dist(quats.copy(), quats.copy()), 0.0, **TOL)


def test_bounded_quat_dist_ignores_unnormalized_input(quats):
    scaled = 3.0 * quats[:2]
    ours = rewards.bounded_quat_dist(scaled.copy(), quats[4:].copy())
    theirs = upstream.bounded_quat_dist(scaled.copy(), quats[4:].copy())
    np.testing.assert_allclose(ours, theirs, **TOL)


def test_compute_squared_differences(features):
    walker, reference = features
    ours = rewards.compute_squared_differences(copy.deepcopy(walker),
                                              copy.deepcopy(reference))
    theirs = upstream.compute_squared_differences(copy.deepcopy(walker),
                                                  copy.deepcopy(reference))
    assert set(ours) == set(theirs)
    for key in theirs:
        np.testing.assert_allclose(ours[key], theirs[key], **TOL)


def test_compute_squared_differences_excludes(features):
    walker, reference = features
    ours = rewards.compute_squared_differences(
        copy.deepcopy(walker), copy.deepcopy(reference),
        exclude_keys=('appendages',))
    assert set(ours) == {'center_of_mass', 'joints_velocity', 'body_quaternions'}


def test_compute_squared_differences_singular_quaternion_key(quats):
    walker = {'quaternion': quats[0]}
    reference = {'quaternion': quats[1]}
    ours = rewards.compute_squared_differences(copy.deepcopy(walker),
                                              copy.deepcopy(reference))
    theirs = upstream.compute_squared_differences(copy.deepcopy(walker),
                                                  copy.deepcopy(reference))
    np.testing.assert_allclose(ours['quaternion'], theirs['quaternion'], **TOL)


def test_multi_term_pose_reward(features):
    walker, reference = features
    ours = rewards.multi_term_pose_reward_fn(copy.deepcopy(walker),
                                             copy.deepcopy(reference))
    theirs = upstream.multi_term_pose_reward_fn(copy.deepcopy(walker),
                                                copy.deepcopy(reference))
    np.testing.assert_allclose(ours.reward, theirs.reward, **TOL)
    assert list(ours.reward_terms) == list(theirs.reward_terms)
    for key in theirs.reward_terms:
        np.testing.assert_allclose(ours.reward_terms[key],
                                   theirs.reward_terms[key], **TOL)


def test_termination_reward():
    ours = rewards.termination_reward_fn(0.25, 1.0)
    theirs = upstream.termination_reward_fn(0.25, 1.0)
    assert ours.reward == theirs.reward == 0.75
    assert list(ours.reward_terms) == list(theirs.reward_terms)


def test_comic_reward(features):
    walker, reference = features
    ours = rewards.comic_reward_fn(0.25, 1.0, copy.deepcopy(walker),
                                   copy.deepcopy(reference))
    theirs = upstream.comic_reward_fn(0.25, 1.0, copy.deepcopy(walker),
                                      copy.deepcopy(reference))
    np.testing.assert_allclose(ours.reward, theirs.reward, **TOL)
    assert list(ours.reward_terms) == list(theirs.reward_terms)


def test_debug(features):
    walker, reference = features
    ours = rewards.debug(copy.deepcopy(reference), copy.deepcopy(walker))
    theirs = upstream.debug(copy.deepcopy(reference), copy.deepcopy(walker))
    assert ours.reward == 0.0 and ours.reward_terms is None
    assert set(ours.debug) == set(theirs.debug)


@pytest.mark.parametrize('key', ['termination_reward', 'multi_term_pose_reward',
                                 'comic'])
def test_get_reward_registry(key):
    assert rewards.get_reward(key).__name__ == upstream.get_reward(key).__name__
    assert (rewards.get_reward_channels(key) ==
            upstream.get_reward_channels(key))


def test_get_reward_rejects_unknown_key():
    with pytest.raises(ValueError):
        rewards.get_reward('nope')
    with pytest.raises(ValueError):
        rewards.get_reward_channels('nope')


def test_bounded_quat_dist_rejects_mismatched_shapes(quats):
    """The kernel indexes `target` with the row count of `source`."""
    with pytest.raises(ValueError):
        rewards.bounded_quat_dist(quats[:3].copy(), quats[:2].copy())


def test_compute_squared_differences_rejects_length_mismatch(quats):
    walker = {'joints_velocity': np.zeros(4)}
    reference = {'joints_velocity': np.zeros(3)}
    with pytest.raises(ValueError):
        rewards.compute_squared_differences(walker, reference)
