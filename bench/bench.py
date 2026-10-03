"""mojo-dm-control against upstream dm_control on the same data.

Every case runs the identical inputs through both implementations and checks
they agree before timing, so a fast-but-wrong kernel cannot win a row.

Run it with `pixi run bench`; that task holds a machine-wide flock so a
concurrent job on this box cannot distort the numbers.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from dm_control.composer.variation import rotations as up_rotations  # noqa: E402
from dm_control.locomotion.tasks.reference_pose import rewards as up_pose  # noqa: E402
from dm_control.mujoco import math as up_math  # noqa: E402
from dm_control.utils import inverse_kinematics as up_ik  # noqa: E402
from dm_control.utils import rewards as up_rewards  # noqa: E402
from dm_control.utils import transformations as up_tr  # noqa: E402
from mojo_dm_control.composer.variation import rotations  # noqa: E402
from mojo_dm_control.locomotion.tasks.reference_pose import rewards as pose  # noqa: E402
from mojo_dm_control.mujoco import math as mjmath  # noqa: E402
from mojo_dm_control.utils import inverse_kinematics as ik  # noqa: E402
from mojo_dm_control.utils import rewards  # noqa: E402
from mojo_dm_control.utils import transformations as tr  # noqa: E402

REPEAT = 5


def timeit(fn) -> float:
    best = math.inf
    for _ in range(REPEAT):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def quats(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("transformations.quat_conj (1M quats)")
def _():
    q = quats(1_000_000)
    return (lambda: tr.quat_conj(q), lambda: up_tr.quat_conj(q))


@case("transformations.quat_inv (1M quats)")
def _():
    q = quats(1_000_000)
    return (lambda: tr.quat_inv(q), lambda: up_tr.quat_inv(q))


@case("transformations.quat_mul (1M quats)")
def _():
    a, b = quats(1_000_000, 0), quats(1_000_000, 1)
    return (lambda: tr.quat_mul(a, b), lambda: up_tr.quat_mul(a, b))


@case("transformations.quat_diff (1M quats)")
def _():
    a, b = quats(1_000_000, 0), quats(1_000_000, 1)
    return (lambda: tr.quat_diff(a, b), lambda: up_tr.quat_diff(a, b))


@case("transformations.quat_log (1M quats)")
def _():
    q = quats(1_000_000)
    return (lambda: tr.quat_log(q), lambda: up_tr.quat_log(q))


@case("transformations.quat_dist (1M quats)")
def _():
    a, b = quats(1_000_000, 0), quats(1_000_000, 1)
    return (lambda: tr.quat_dist(a, b), lambda: up_tr.quat_dist(a, b))


@case("transformations.quat_rotate (100k calls)")
def _():
    q = quats(100_000)
    v = np.array([0.3, -1.2, 0.75])
    return (lambda: [tr.quat_rotate(x, v) for x in q],
            lambda: [up_tr.quat_rotate(x, v) for x in q])


@case("transformations.axisangle_to_quat (1M)")
def _():
    a = np.random.default_rng(2).normal(size=(1_000_000, 3))
    return (lambda: tr.axisangle_to_quat(a), lambda: up_tr.axisangle_to_quat(a))


@case("transformations.rotation_x_axis (1M)")
def _():
    t = np.random.default_rng(3).normal(size=1_000_000)
    return (lambda: tr.rotation_x_axis(t), lambda: up_tr.rotation_x_axis(t))


@case("transformations.euler_to_rmat ZYX (1M)")
def _():
    e = np.random.default_rng(4).normal(size=(1_000_000, 3))
    return (lambda: tr.euler_to_rmat(e, 'ZYX'),
            lambda: up_tr.euler_to_rmat(e, 'ZYX'))


@case("transformations.euler_to_rmat XYZ full (1M)")
def _():
    e = np.random.default_rng(5).normal(size=(1_000_000, 3))
    return (lambda: tr.euler_to_rmat(e, 'XYZ', full=True),
            lambda: up_tr.euler_to_rmat(e, 'XYZ', full=True))


@case("transformations.rotation_matrix_2d (1M)")
def _():
    t = np.random.default_rng(6).normal(size=1_000_000)
    return (lambda: tr.rotation_matrix_2d(t),
            lambda: up_tr.rotation_matrix_2d(t))


@case("rewards.tolerance gaussian margin=1 (1M)")
def _():
    x = np.random.default_rng(7).normal(size=1_000_000)
    return (lambda: rewards.tolerance(x, bounds=(-0.2, 0.2), margin=1.0),
            lambda: up_rewards.tolerance(x, bounds=(-0.2, 0.2), margin=1.0))


@case("rewards.tolerance linear margin=1 (1M)")
def _():
    x = np.random.default_rng(8).normal(size=1_000_000)
    return (lambda: rewards.tolerance(x, bounds=(-0.2, 0.2), margin=1.0,
                                      sigmoid='linear'),
            lambda: up_rewards.tolerance(x, bounds=(-0.2, 0.2), margin=1.0,
                                         sigmoid='linear'))


@case("rewards.tolerance margin=0 (1M)")
def _():
    x = np.random.default_rng(9).normal(size=1_000_000)
    return (lambda: rewards.tolerance(x, bounds=(-0.2, 0.2)),
            lambda: up_rewards.tolerance(x, bounds=(-0.2, 0.2)))


@case("mujoco.math.mj_quatprod (1M quats)")
def _():
    a, b = quats(1_000_000, 0), quats(1_000_000, 1)
    return (lambda: mjmath.mj_quatprod(a, b), lambda: up_tr.quat_mul(a, b))


@case("mujoco.math.mj_quatdiff (1M quats)")
def _():
    a, b = quats(1_000_000, 0), quats(1_000_000, 1)
    return (lambda: mjmath.mj_quatdiff(a, b),
            lambda: tr.quat_mul(tr.quat_conj(a), b))


@case("mujoco.math.euler2quat (20k calls)")
def _():
    e = np.random.default_rng(10).normal(size=(20_000, 3))
    return (lambda: np.array([mjmath.euler2quat(*t) for t in e]),
            lambda: np.array([up_math.euler2quat(*t) for t in e]))


@case("reference_pose.bounded_quat_dist (1M)")
def _():
    a, b = quats(1_000_000, 0), quats(1_000_000, 1)
    return (lambda: pose.bounded_quat_dist(a.copy(), b.copy()),
            lambda: up_pose.bounded_quat_dist(a.copy(), b.copy()))


@case("variation.QuaternionRotate x20k calls")
def _():
    q = quats(20_000)
    v = np.array([0.1, 0.2, 0.3])
    var = rotations.QuaternionRotate(q[0], v)
    up_var = up_rotations.QuaternionRotate(q[0], v)
    return (lambda: [var(initial_value=q[i], current_value=q[i],
                         random_state=np.random) for i in range(len(q))],
            lambda: [up_var(initial_value=q[i], current_value=q[i],
                            random_state=np.random) for i in range(len(q))])


@case("inverse_kinematics.nullspace_method x200 (6x12)")
def _():
    rng = np.random.default_rng(11)
    jacs = [rng.normal(size=(6, 12)) for _ in range(200)]
    deltas = [rng.normal(size=6) for _ in range(200)]
    return (lambda: [ik.nullspace_method(j, d, 3e-2) for j, d in zip(jacs, deltas)],
            lambda: [up_ik.nullspace_method(j, d, 3e-2) for j, d in zip(jacs, deltas)])


def main() -> None:
    print(f"mojo-dm-control vs dm_control  (dm_control {up_math.__name__} "
          f"bindings, python {platform.python_version()})")
    print(f"host: {platform.platform()}")
    print(f"cpus: {os.cpu_count()}")
    print()
    print(f"{'case':<50}{'mojo-dm-control':>17}{'dm_control':>13}{'ratio':>9}")
    print("-" * 89)
    for name, build in CASES:
        ours, theirs = build()
        a = np.asarray(ours())
        b = np.asarray(theirs())
        if a.shape != b.shape:
            raise SystemExit(f"{name}: shape mismatch {a.shape} vs {b.shape}")
        error = np.nanmax(np.abs(a - b))
        scale = max(1.0, float(np.nanmax(np.abs(b))))
        ta, tb = timeit(ours), timeit(theirs)
        flag = "" if error <= 1e-7 * scale else f"  MISMATCH {error:.3e}"
        print(f"{name:<50}{ta * 1e3:>14.2f}ms{tb * 1e3:>10.2f}ms"
              f"{tb / ta:>8.2f}x{flag}")
    print()
    print(f"ratio > 1 means the Mojo kernel is faster; best of {REPEAT} runs.")


if __name__ == "__main__":
    main()
