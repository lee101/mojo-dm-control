"""Define reward function options for reference pose tasks.

Mojo port of `dm_control.locomotion.tasks.reference_pose.rewards`.
"""

from __future__ import annotations

import collections

import numpy as np

from ...._lib import addr, lib, rows

RewardFnOutput = collections.namedtuple('RewardFnOutput',
                                        ['reward', 'debug', 'reward_terms'])


def bounded_quat_dist(source: np.ndarray,
                      target: np.ndarray) -> np.ndarray:
  """Computes a quaternion distance limiting the difference to a max of pi/2.

  This function supports an arbitrary number of batch dimensions, B.

  Args:
    source: a quaternion, shape (B, 4).
    target: another quaternion, shape (B, 4).

  Returns:
    Quaternion distance, shape (B, 1).
  """
  # `np.ascontiguousarray` returns the argument itself when it is already a
  # contiguous float64 array, so upstream's in-place normalisation of both
  # arguments is preserved for the inputs this module is called with.
  source = np.ascontiguousarray(source, dtype=np.float64)
  target = np.ascontiguousarray(target, dtype=np.float64)
  arr, flat_source = rows(source, 4)
  _, flat_target = rows(target, 4)
  if flat_source.shape != flat_target.shape:
    # The kernel walks both buffers with the row count of `source`, so a
    # mismatched `target` would be read out of bounds instead of raising the
    # broadcasting error upstream's product gives.
    raise ValueError(
        'source and target must have the same shape, got {} and {}.'.format(
            arr.shape, np.shape(target)))
  n = flat_source.shape[0]
  dist = np.empty(n, dtype=np.float64)
  lib().mc_bounded_quat_dist(addr(flat_source), addr(flat_target), addr(dist), n)
  # Divide by 2 and add an axis to ensure consistency with expected return
  # shape and magnitude.
  return dist.reshape(arr.shape[:-1] + (1,))


def sort_dict(d):
  return collections.OrderedDict(sorted(d.items()))


def compute_squared_differences(walker_features, reference_features,
                                exclude_keys=()):
  """Computes squared differences of features."""
  squared_differences = {}
  for k in walker_features:
    if k not in exclude_keys:
      if 'quaternion' not in k:
        a, flat_a = rows(walker_features[k], 1)
        b, flat_b = rows(reference_features[k], 1)
        if flat_a.shape[0] != flat_b.shape[0]:
          # The kernel reads `b` with the length of `a`; upstream's
          # elementwise difference would raise on the mismatch instead.
          raise ValueError(
              'length mismatch for feature {!r}: {} vs {}.'.format(
                  k, flat_a.shape[0], flat_b.shape[0]))
        squared_differences[k] = lib().mc_squared_difference_sum(
            addr(flat_a), addr(flat_b), flat_a.shape[0])
      elif 'quaternions' in k:
        quat_dists = bounded_quat_dist(
            walker_features[k], reference_features[k])
        squared_differences[k] = np.sum(quat_dists**2)
      else:
        squared_differences[k] = bounded_quat_dist(
            walker_features[k], reference_features[k])**2

  return squared_differences


def termination_reward_fn(termination_error, termination_error_threshold,
                          **unused_kwargs):
  """Termination error.

  This reward is intended to be used in conjunction with the termination error
  calculated in the task. Due to terminations if error > error_threshold this
  reward will be in [0, 1].

  Args:
    termination_error: termination error computed in tracking task
    termination_error_threshold: task termination threshold
    unused_kwargs: unused_kwargs

  Returns:
    RewardFnOutput tuple containing reward, debug information and reward terms.
  """
  debug_terms = {
      'termination_error': termination_error,
      'termination_error_threshold': termination_error_threshold
  }
  termination_reward = 1 - termination_error / termination_error_threshold
  return RewardFnOutput(reward=termination_reward, debug=debug_terms,
                        reward_terms=sort_dict(
                            {'termination': termination_reward}))


def debug(reference_features, walker_features, **unused_kwargs):
  debug_terms = compute_squared_differences(walker_features, reference_features)
  return RewardFnOutput(reward=0.0, debug=debug_terms, reward_terms=None)


def multi_term_pose_reward_fn(walker_features, reference_features,
                              **unused_kwargs):
  """A reward based on com, body quaternions, joints velocities & appendages."""
  differences = compute_squared_differences(walker_features, reference_features)
  com = .1 * np.exp(-10 * differences['center_of_mass'])
  joints_velocity = 1.0 * np.exp(-0.1 * differences['joints_velocity'])
  appendages = 0.15 * np.exp(-40. * differences['appendages'])
  body_quaternions = 0.65 * np.exp(-2 * differences['body_quaternions'])
  terms = {
      'center_of_mass': com,
      'joints_velocity': joints_velocity,
      'appendages': appendages,
      'body_quaternions': body_quaternions
  }
  reward = sum(terms.values())
  return RewardFnOutput(reward=reward, debug=terms,
                        reward_terms=sort_dict(terms))


def comic_reward_fn(termination_error, termination_error_threshold,
                    walker_features, reference_features, **unused_kwargs):
  """A reward that mixes the termination_reward and multi_term_pose_reward.

  This reward function was used in
    Hasenclever et al.,
    CoMic: Complementary Task Learning & Mimicry for Reusable Skills,
    International Conference on Machine Learning, 2020.
    [https://proceedings.icml.cc/static/paper_files/icml/2020/5013-Paper.pdf]

  Args:
    termination_error: termination error as described
    termination_error_threshold: threshold to determine whether to terminate
      episodes. The threshold is used to construct a reward between [0, 1]
      based on the termination error.
    walker_features: Current features of the walker
    reference_features: features of the current reference pose
    unused_kwargs: unused addtional keyword arguments.

  Returns:
    RewardFnOutput tuple containing reward, debug terms and reward terms.
  """
  termination_reward, debug_terms, termination_reward_terms = (
      termination_reward_fn(termination_error, termination_error_threshold))
  mt_reward, mt_debug_terms, mt_reward_terms = multi_term_pose_reward_fn(
      walker_features, reference_features)
  debug_terms.update(mt_debug_terms)
  reward_terms = {k: 0.5 * v for k, v in termination_reward_terms.items()}
  reward_terms.update(
      {k: 0.5 * v for k, v in mt_reward_terms.items()})
  return RewardFnOutput(
      reward=0.5 * termination_reward + 0.5 * mt_reward,
      debug=debug_terms,
      reward_terms=sort_dict(reward_terms))


_REWARD_FN = {
    'termination_reward': termination_reward_fn,
    'multi_term_pose_reward': multi_term_pose_reward_fn,
    'comic': comic_reward_fn,
}

_REWARD_CHANNELS = {
    'termination_reward': ('termination',),
    'multi_term_pose_reward':
        ('appendages', 'body_quaternions', 'center_of_mass', 'joints_velocity'),
    'comic': ('appendages', 'body_quaternions', 'center_of_mass', 'termination',
              'joints_velocity'),
}


def get_reward(reward_key):
  if reward_key not in _REWARD_FN:
    raise ValueError('Requested loss %s, which is not a valid option.' %
                     reward_key)

  return _REWARD_FN[reward_key]


def get_reward_channels(reward_key):
  if reward_key not in _REWARD_CHANNELS:
    raise ValueError('Requested loss %s, which is not a valid option.' %
                     reward_key)

  return _REWARD_CHANNELS[reward_key]
