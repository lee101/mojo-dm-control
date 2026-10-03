"""Soft indicator function evaluating whether a number is within bounds.

Mojo port of `dm_control.utils.rewards`.
"""

from __future__ import annotations

import numpy as np

from .._lib import addr, fn, lib, rows

_DEFAULT_VALUE_AT_MARGIN = 0.1

# Branch order of upstream's `_sigmoids` if/elif chain; the Mojo kernel selects
# the same branch from this code.
_SIGMOIDS = {
    'gaussian': 0,
    'hyperbolic': 1,
    'long_tail': 2,
    'reciprocal': 3,
    'cosine': 4,
    'linear': 5,
    'quadratic': 6,
    'tanh_squared': 7,
}


def _check_sigmoids(sigmoid, value_at_1):
  """Upstream's validation of `sigmoid` and `value_at_1`, kept in the wrapper."""
  if sigmoid in ('cosine', 'linear', 'quadratic'):
    if not 0 <= value_at_1 < 1:
      raise ValueError('`value_at_1` must be nonnegative and smaller than 1, '
                       'got {}.'.format(value_at_1))
  else:
    if not 0 < value_at_1 < 1:
      raise ValueError('`value_at_1` must be strictly between 0 and 1, '
                       'got {}.'.format(value_at_1))

  if sigmoid not in _SIGMOIDS:
    raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))
  return _SIGMOIDS[sigmoid]


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian',
              value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
  """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

  Args:
    x: A scalar or numpy array.
    bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
      the target interval. These can be infinite if the interval is unbounded
      at one or both ends, or they can be equal to one another if the target
      value is exact.
    margin: Float. Parameter that controls how steeply the output decreases as
      `x` moves out-of-bounds.
    * If `margin == 0` then the output will be 0 for all values of `x`
      outside of `bounds`.
    * If `margin > 0` then the output will decrease sigmoidally with
      increasing distance from the nearest bound.
    sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
       'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
    value_at_margin: A float between 0 and 1 specifying the output value when
      the distance from `x` to the nearest bound is equal to `margin`. Ignored
      if `margin == 0`.

  Returns:
    A float or numpy array with values between 0.0 and 1.0.

  Raises:
    ValueError: If `bounds[0] > bounds[1]`.
    ValueError: If `margin` is negative.
  """
  lower, upper = bounds
  if lower > upper:
    raise ValueError('Lower bound must be <= upper bound.')
  if margin < 0:
    raise ValueError('`margin` must be non-negative.')

  arr, flat = rows(x, 1)
  n = flat.shape[0]
  value = np.empty(n, dtype=np.float64)
  if margin == 0:
    fn('mc_in_bounds')(addr(flat), addr(value), n, lower, upper)
  else:
    # The distance, the sigmoid and the `where(in_bounds, 1.0, ...)` are one
    # kernel: see `tolerance_margin` in src/ported.mojo for why the two
    # branches upstream takes collapse into a clamp.
    fn('mc_tolerance_margin')(addr(flat), addr(value), n, lower, upper, margin,
                              float(value_at_margin),
                              _check_sigmoids(sigmoid, value_at_margin))

  value = value.reshape(arr.shape)
  return float(value) if np.isscalar(x) else value
