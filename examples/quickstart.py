"""Runnable tour of the mojo-dm-control API.

    pixi run build
    pixi run python examples/quickstart.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from mojo_dm_control.utils import rewards, transformations as tr  # noqa: E402
from mojo_dm_control.mujoco import math as mjmath  # noqa: E402

# --- rotations: quaternion <-> rotation matrix ---------------------------
euler = np.array([0.3, -0.7, 1.1])
rmat = tr.euler_to_rmat(euler, ordering='ZYX')
quat = tr.mat_to_quat(rmat)
print('euler(ZYX)   ', np.round(euler, 6))
print('rmat[0]      ', np.round(rmat[0], 6))
print('quat         ', np.round(quat, 6))
print('back to euler', np.round(tr.quat_to_euler(quat, 'ZYX'), 6))

# --- rotations, batched ---------------------------------------------------
rng = np.random.default_rng(0)
quats = rng.normal(size=(1_000_000, 4))
quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
distances = tr.quat_dist(quats[:1], quats[1:])
print('quat_dist    ', distances.shape, 'mean', float(distances.mean()))

# --- a vector rotated by a batch of quaternions ---------------------------
v = np.array([0.0, 0.0, 1.0])
print('quat_rotate  ', np.round(tr.quat_rotate(quats[0], v), 6))

# --- MuJoCo's own quaternion helpers -------------------------------------
print('mj_quatprod  ', np.round(mjmath.mj_quatprod(quats[0], quats[1]), 6))
print('mj_quat2vel  ', np.round(mjmath.mj_quat2vel(quats[0], 0.005), 6))
print('euler2quat   ', np.round(mjmath.euler2quat(0.0, 0.0, 90.0), 6))

# --- reward shaping -------------------------------------------------------
x = np.linspace(-3.0, 3.0, 7)
for sigmoid in ('gaussian', 'linear', 'cosine', 'long_tail'):
    values = rewards.tolerance(x, bounds=(-0.5, 0.5), margin=1.0,
                               sigmoid=sigmoid)
    print(f'tolerance {sigmoid:<10}', np.round(values, 6))
