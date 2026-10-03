"""Control-specific helpers.

Mojo port of the compute-bearing free functions in `dm_control.rl.control`. The
`Environment`/`Physics`/`Task` abstract classes in that module are pure
orchestration with no numerical content, so they are not ported; use
`dm_control.rl.control` directly for those.
"""

from __future__ import annotations

import collections

import numpy as np

FLAT_OBSERVATION_KEY = 'observations'


def compute_n_steps(control_timestep, physics_timestep, tolerance=1e-8):
  """Returns the number of physics timesteps in a single control timestep.

  Args:
    control_timestep: Control time-step, should be an integer multiple of the
      physics timestep.
    physics_timestep: The time-step of the physics simulation.
    tolerance: Optional tolerance value for checking if `physics_timestep`
      divides `control_timestep`.

  Returns:
    The number of physics timesteps in a single control timestep.

  Raises:
    ValueError: If `control_timestep` is smaller than `physics_timestep` or if
      `control_timestep` is not an integer multiple of `physics_timestep`.
  """
  if control_timestep < physics_timestep:
    raise ValueError(
        'Control timestep ({}) cannot be smaller than physics timestep ({}).'.
        format(control_timestep, physics_timestep))
  if abs((control_timestep / physics_timestep - round(
      control_timestep / physics_timestep))) > tolerance:
    raise ValueError(
        'Control timestep ({}) must be an integer multiple of physics timestep '
        '({})'.format(control_timestep, physics_timestep))
  return int(round(control_timestep / physics_timestep))



def flatten_observation(observation, output_key=FLAT_OBSERVATION_KEY):
  """Flattens multiple observation arrays into a single numpy array.

  Args:
    observation: A mutable mapping from observation names to numpy arrays.
    output_key: The key for the flattened observation array in the output.

  Returns:
    A mutable mapping of the same type as `observation`. This will contain a
    single key-value pair consisting of `output_key` and the flattened
    and concatenated observation array.

  Raises:
    ValueError: If `observation` is not a `collections.abc.MutableMapping`.
  """
  if not isinstance(observation, collections.abc.MutableMapping):
    raise ValueError('Can only flatten dict-like observations.')

  if isinstance(observation, collections.OrderedDict):
    keys = observation.keys()
  else:
    # Keep a consistent ordering for other mappings.
    keys = sorted(observation.keys())

  observation_arrays = [observation[key].ravel() for key in keys]
  return type(observation)([(output_key, np.concatenate(observation_arrays))])


