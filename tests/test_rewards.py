"""Parity of mojo_dm_control.utils.rewards against upstream dm_control.

`std.math.exp` and `std.math.cosh` are polynomial approximations in this
toolchain: measured against libm over [-8, 3] they carry up to ~6e-8 relative
error once the exponential's own rounding is amplified, so the sigmoids that
use them get a correspondingly looser tolerance.
"""

import numpy as np
import pytest

from dm_control.utils import rewards as upstream
from mojo_dm_control.utils import rewards

EXPONENTIAL = ('gaussian', 'hyperbolic', 'tanh_squared')
ALGEBRAIC = ('long_tail', 'reciprocal', 'cosine', 'linear', 'quadratic')
SIGMOIDS = EXPONENTIAL + ALGEBRAIC
# Upstream only allows `value_at_1 == 0` for these three.
ALLOW_ZERO = ('cosine', 'linear', 'quadratic')

TOL = dict(rtol=1e-8, atol=1e-10)
EXACT = dict(rtol=1e-12, atol=1e-12)


def tol_for(sigmoid):
    return TOL if sigmoid in EXPONENTIAL else EXACT


@pytest.fixture
def x():
    return np.random.default_rng(2).normal(size=2000) * 1.5


@pytest.mark.parametrize('sigmoid', SIGMOIDS)
def test_tolerance_with_margin(x, sigmoid):
    ours = rewards.tolerance(x, bounds=(-0.2, 0.2), margin=0.7, sigmoid=sigmoid)
    theirs = upstream.tolerance(x, bounds=(-0.2, 0.2), margin=0.7, sigmoid=sigmoid)
    np.testing.assert_allclose(ours, theirs, **tol_for(sigmoid))


@pytest.mark.parametrize('sigmoid', SIGMOIDS)
def test_tolerance_without_margin(x, sigmoid):
    ours = rewards.tolerance(x, bounds=(-0.2, 0.2), sigmoid=sigmoid)
    theirs = upstream.tolerance(x, bounds=(-0.2, 0.2), sigmoid=sigmoid)
    np.testing.assert_allclose(ours, theirs, **tol_for(sigmoid))


@pytest.mark.parametrize('sigmoid', SIGMOIDS)
def test_tolerance_scalar(sigmoid):
    for value in (-0.5, 0.0, 0.3, 1.4):
        ours = rewards.tolerance(value, bounds=(0.0, 1.0), margin=0.5,
                                 sigmoid=sigmoid)
        theirs = upstream.tolerance(value, bounds=(0.0, 1.0), margin=0.5,
                                   sigmoid=sigmoid)
        assert isinstance(ours, float)
        assert isinstance(theirs, float)
        np.testing.assert_allclose(ours, theirs, **tol_for(sigmoid))


def test_tolerance_2d_input(x):
    grid = np.stack(np.meshgrid(x[:50], x[50:100]), axis=-1)
    np.testing.assert_allclose(
        rewards.tolerance(grid, bounds=(-0.2, 0.2), margin=0.7),
        upstream.tolerance(grid, bounds=(-0.2, 0.2), margin=0.7), **TOL)


def test_tolerance_infinite_bounds(x):
    np.testing.assert_allclose(
        rewards.tolerance(x, bounds=(0.0, float('inf')), margin=0.5),
        upstream.tolerance(x, bounds=(0.0, float('inf')), margin=0.5), **TOL)
    np.testing.assert_allclose(
        rewards.tolerance(x, bounds=(float('-inf'), 0.0), margin=0.5),
        upstream.tolerance(x, bounds=(float('-inf'), 0.0), margin=0.5), **TOL)


def test_tolerance_is_one_inside_bounds(x):
    value = rewards.tolerance(x, bounds=(-1e9, 1e9), margin=1.0)
    assert np.all(value == 1.0)


def test_tolerance_value_at_margin(x):
    """At exactly `margin` from the bound the output is `value_at_margin`."""
    for sigmoid in SIGMOIDS:
        ours = rewards.tolerance(np.array([0.7]), bounds=(0.0, 1.0), margin=0.7,
                                 sigmoid=sigmoid, value_at_margin=0.25)
        theirs = upstream.tolerance(np.array([0.7]), bounds=(0.0, 1.0),
                                   margin=0.7, sigmoid=sigmoid,
                                   value_at_margin=0.25)
        np.testing.assert_allclose(ours, theirs, **tol_for(sigmoid))


def test_tolerance_rejects_bad_bounds(x):
    with pytest.raises(ValueError):
        rewards.tolerance(x, bounds=(1.0, -1.0))


def test_tolerance_rejects_negative_margin(x):
    with pytest.raises(ValueError):
        rewards.tolerance(x, margin=-1.0)


@pytest.mark.parametrize('sigmoid', ['gaussian', 'long_tail', 'reciprocal',
                                      'hyperbolic', 'tanh_squared'])
def test_sigmoids_reject_value_at_one_out_of_range(x, sigmoid):
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            rewards.tolerance(x, margin=1.0, sigmoid=sigmoid,
                              value_at_margin=bad)


@pytest.mark.parametrize('sigmoid', ALLOW_ZERO)
def test_sigmoids_allow_zero_value_at_one(x, sigmoid):
    ours = rewards.tolerance(x, margin=1.0, sigmoid=sigmoid, value_at_margin=0.0)
    theirs = upstream.tolerance(x, margin=1.0, sigmoid=sigmoid,
                                value_at_margin=0.0)
    np.testing.assert_allclose(ours, theirs, **EXACT)


def test_sigmoids_rejects_unknown_type(x):
    with pytest.raises(ValueError):
        rewards.tolerance(x, margin=1.0, sigmoid='cubic')
