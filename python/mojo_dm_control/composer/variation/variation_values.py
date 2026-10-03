"""Utilities for handling nested structures of callables or constants.

Mojo port of `dm_control.composer.variation.variation_values`.
"""

from __future__ import annotations

import tree


def evaluate(structure, *args, **kwargs):
  """Evaluates a arbitrarily nested structure of callables or constant values.

  Args:
    structure: An arbitrarily nested structure of callables or constant values.
      By "structures", we mean lists, tuples, namedtuples, or dicts.
    *args: Positional arguments passed to each callable in `structure`.
    **kwargs: Keyword arguments passed to each callable in `structure`.

  Returns:
    The same nested structure, with each callable replaced by the value returned
    by calling it.
  """
  return tree.map_structure(
      lambda x: x(*args, **kwargs) if callable(x) else x, structure)
