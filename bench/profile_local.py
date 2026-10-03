"""Times the port's own kernels, without upstream, to see where the time goes.

`bench.py` is the benchmark of record; this one is the companion for it. It
measures the same cases at the same sizes through the same Python API, so a row
that is slow here is slow for the same reason it is slow there, but it leaves out
`dm_control` and so runs in a fraction of the time. Use it while iterating on a
kernel, and re-run `pixi run bench` before believing any number.

Run it under `flock /tmp/mojo-bench.lock`, the same lock `pixi run bench` takes:

    flock /tmp/mojo-bench.lock python bench/profile_local.py [substring]
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

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


@case("quat_conj 1M")
def _():
    q = quats(1_000_000)
    return lambda: tr.quat_conj(q)


@case("quat_inv 1M")
def _():
    q = quats(1_000_000)
    return lambda: tr.quat_inv(q)


@case("quat_mul 1M")
def _():
    a, b = quats(1_000_000, 0), quats(1_000_000, 1)
    return lambda: tr.quat_mul(a, b)


@case("quat_diff 1M")
def _():
    a, b = quats(1_000_000, 0), quats(1_000_000, 1)
    return lambda: tr.quat_diff(a, b)


@case("quat_log 1M")
def _():
    q = quats(1_000_000)
    return lambda: tr.quat_log(q)


@case("quat_dist 1M")
def _():
    a, b = quats(1_000_000, 0), quats(1_000_000, 1)
    return lambda: tr.quat_dist(a, b)


@case("quat_rotate 100k")
def _():
    q = quats(100_000)
    v = np.array([0.3, -1.2, 0.75])
    return lambda: [tr.quat_rotate(x, v) for x in q]


@case("axisangle_to_quat 1M")
def _():
    a = np.random.default_rng(2).normal(size=(1_000_000, 3))
    return lambda: tr.axisangle_to_quat(a)


@case("rotation_x_axis 1M")
def _():
    t = np.random.default_rng(3).normal(size=1_000_000)
    return lambda: tr.rotation_x_axis(t)


@case("euler_to_rmat ZYX 1M")
def _():
    e = np.random.default_rng(4).normal(size=(1_000_000, 3))
    return lambda: tr.euler_to_rmat(e, 'ZYX')


@case("euler_to_rmat XYZ full 1M")
def _():
    e = np.random.default_rng(5).normal(size=(1_000_000, 3))
    return lambda: tr.euler_to_rmat(e, 'XYZ', full=True)


@case("rotation_matrix_2d 1M")
def _():
    t = np.random.default_rng(6).normal(size=1_000_000)
    return lambda: tr.rotation_matrix_2d(t)


@case("tolerance gaussian 1M")
def _():
    x = np.random.default_rng(7).normal(size=1_000_000)
    return lambda: rewards.tolerance(x, bounds=(-0.2, 0.2), margin=1.0)


@case("tolerance linear 1M")
def _():
    x = np.random.default_rng(8).normal(size=1_000_000)
    return lambda: rewards.tolerance(x, bounds=(-0.2, 0.2), margin=1.0,
                                    sigmoid='linear')


@case("tolerance margin=0 1M")
def _():
    x = np.random.default_rng(9).normal(size=1_000_000)
    return lambda: rewards.tolerance(x, bounds=(-0.2, 0.2))


@case("mj_quatprod 1M")
def _():
    a, b = quats(1_000_000, 0), quats(1_000_000, 1)
    return lambda: mjmath.mj_quatprod(a, b)


@case("mj_quatdiff 1M")
def _():
    a, b = quats(1_000_000, 0), quats(1_000_000, 1)
    return lambda: mjmath.mj_quatdiff(a, b)


@case("euler2quat 20k")
def _():
    e = np.random.default_rng(10).normal(size=(20_000, 3))
    return lambda: np.array([mjmath.euler2quat(*t) for t in e])


@case("bounded_quat_dist 1M")
def _():
    a, b = quats(1_000_000, 0), quats(1_000_000, 1)
    return lambda: pose.bounded_quat_dist(a.copy(), b.copy())


@case("QuaternionRotate 20k")
def _():
    q = quats(20_000)
    v = np.array([0.1, 0.2, 0.3])
    var = rotations.QuaternionRotate(q[0], v)
    return lambda: [var(initial_value=q[i], current_value=q[i],
                         random_state=np.random) for i in range(len(q))]


@case("nullspace_method 200 (6x12)")
def _():
    rng = np.random.default_rng(11)
    jacs = [rng.normal(size=(6, 12)) for _ in range(200)]
    deltas = [rng.normal(size=6) for _ in range(200)]
    return lambda: [ik.nullspace_method(j, d, 3e-2) for j, d in zip(jacs, deltas)]


@case("euler2quat_batch 1M")
def _():
    e = np.random.default_rng(12).normal(size=(1_000_000, 3))
    return lambda: tr.euler_to_quat(e)


@case("euler_to_quat 1M ZYX")
def _():
    e = np.random.default_rng(12).normal(size=(1_000_000, 3))
    return lambda: tr.euler_to_quat(e, 'ZYX')


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    total = 0.0
    for name, build in CASES:
        if only and only not in name:
            continue
        fn = build()
        t = timeit(fn)
        total += t
        print(f"{name:<36}{t * 1e3:>10.2f}ms")
    print(f"{'TOTAL':<36}{total * 1e3:>10.2f}ms")


if __name__ == "__main__":
    main()