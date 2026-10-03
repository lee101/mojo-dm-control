"""Parity of mojo_dm_control.rl.control against upstream."""

import collections

import numpy as np
import pytest

from dm_control.rl import control as upstream
from mojo_dm_control.rl import control


def test_flat_observation_key():
    assert control.FLAT_OBSERVATION_KEY == upstream.FLAT_OBSERVATION_KEY


@pytest.mark.parametrize('steps', [1, 2, 5, 10, 100])
def test_compute_n_steps(steps):
    physics_timestep = 0.002
    assert (control.compute_n_steps(steps * physics_timestep, physics_timestep) ==
            upstream.compute_n_steps(steps * physics_timestep, physics_timestep))


def test_compute_n_steps_uses_the_tolerance():
    control_timestep = 0.002 * 2 + 1e-10
    assert control.compute_n_steps(control_timestep, 0.002, tolerance=1e-3) == 2
    with pytest.raises(ValueError):
        control.compute_n_steps(control_timestep, 0.002)


def test_compute_n_steps_rejects_short_control_step():
    with pytest.raises(ValueError):
        control.compute_n_steps(0.001, 0.002)


def test_compute_n_steps_rejects_non_multiple():
    with pytest.raises(ValueError):
        control.compute_n_steps(0.005, 0.002)


def test_flatten_observation():
    obs = collections.OrderedDict([
        ('a', np.arange(6, dtype=np.float64).reshape(2, 3)),
        ('b', np.ones(4)),
    ])
    ours = control.flatten_observation(obs)
    theirs = upstream.flatten_observation(collections.OrderedDict(obs))
    assert list(ours) == list(theirs) == [control.FLAT_OBSERVATION_KEY]
    np.testing.assert_array_equal(ours[control.FLAT_OBSERVATION_KEY],
                                  theirs[control.FLAT_OBSERVATION_KEY])
    assert ours[control.FLAT_OBSERVATION_KEY].shape == (10,)


def test_flatten_observation_sorts_plain_dicts():
    obs = {'b': np.ones(2), 'a': np.zeros(3)}
    ours = control.flatten_observation(obs)
    theirs = upstream.flatten_observation(dict(obs))
    np.testing.assert_array_equal(ours['observations'], theirs['observations'])
    np.testing.assert_array_equal(ours['observations'], np.array([0., 0., 0., 1., 1.]))


def test_flatten_observation_custom_key():
    obs = collections.OrderedDict([('a', np.ones(3)), ('b', np.zeros(2))])
    ours = control.flatten_observation(obs, output_key='flat')
    assert list(ours) == ['flat']
    assert ours['flat'].shape == (5,)


def test_flatten_observation_rejects_non_mapping():
    with pytest.raises(ValueError):
        control.flatten_observation([1, 2, 3])
