# mojo-dm-control

The compute-bound primitives of [google-deepmind/dm_control](https://github.com/google-deepmind/dm_control)
— rigid-body transformations, quaternion algebra, reward shaping, Jacobian
products and the LQR policy construction — rewritten in Mojo and exposed through
a Python package whose names, argument orders, defaults and return shapes match
upstream's. The covered subset is a drop-in: `import mojo_dm_control` and
`import dm_control` give you interchangeable calls.

```
pixi install
pixi run build
pixi run test
pixi run bench
```

## What this is

dm_control's Python surface is mostly orchestration: MJCF parsing, named
indexers, environment classes, and a thin layer over MuJoCo's C library. What is
left once that is stripped away is a small set of dense, batched, elementwise
kernels that dm_control calls in the inner loop of every task, reward and
observation pipeline. Those are the kernels here, in one Mojo compilation unit
(`src/ported.mojo`), reachable from Python through ctypes.

`src/ported.mojo` is written to be read next to the upstream file it came from.
The functions appear in upstream's source order, grouped by upstream module,
with upstream's names, branch structure and arithmetic; only the array plumbing
(reshaping a batched NumPy argument into a flat buffer) is done in the ctypes
glue rather than in the kernel. Where an optimisation changed the shape of a
loop — a fused pass, a width-`W` vector body, a row range handed to workers —
the reason is in a comment at the kernel and in "Design notes" below.

## Covered subset

| upstream module | functions |
| --- | --- |
| `dm_control.mujoco.math` | `euler2quat`, `mj_quatprod`, `mj_quat2vel`, `mj_quatneg`, `mj_quatdiff` |
| `dm_control.utils.transformations` | `euler_to_quat`, `euler_to_rmat`, `quat_conj`, `quat_inv`, `quat_mul`, `quat_diff`, `quat_log`, `quat_dist`, `quat_rotate`, `quat_to_axisangle`, `quat_to_euler`, `quat_to_mat`, `mat_to_quat`, `axisangle_to_quat`, `rotation_x_axis`, `rotation_y_axis`, `rotation_z_axis`, `rmat_to_euler`, `rotation_matrix_2d` (all seven Euler orderings, including the pole branches) |
| `dm_control.utils.rewards` | `tolerance` and its eight sigmoid types |
| `dm_control.locomotion.tasks.reference_pose.rewards` | `bounded_quat_dist`, `compute_squared_differences`, `termination_reward_fn`, `multi_term_pose_reward_fn`, `comic_reward_fn`, `debug`, `get_reward`, `get_reward_channels` |
| `dm_control.utils.inverse_kinematics` | `nullspace_method`, `qpos_from_site_pose`, `IKResult` |
| `dm_control.suite.lqr_solver` | `solve` |
| `dm_control.rl.control` | `compute_n_steps`, `flatten_observation`, `FLAT_OBSERVATION_KEY` |
| `dm_control.composer.variation.rotations` | `UniformQuaternion`, `QuaternionFromAxisAngle`, `QuaternionPreMultiply`, `QuaternionRotate`, `IDENTITY_QUATERNION`, plus the `Variation` base class they need |

## Not covered, and why

In this distribution of dm_control the constraint solver, the integrators
(`Euler`, `RK4P`, `ImplicitEuler`), collision detection, `rigid_body`,
`transform.py`, `optim` and `trimesh` are not part of the Python package: the
tree is
`dm_control/{_render,autowrap,blender,composer,entities,locomotion,manipulation,mjcf,mujoco,rl,suite,third_party,utils,viewer}`,
and the heavy numerics live in MuJoCo's C library. There is no Python code to
rewrite for them, so the port targets the modules that do exist and are dense
enough to be worth it.

The rest of the package is left alone on purpose: `dm_control.mujoco.engine`
(`Physics`, `Camera`, rendering), `dm_control.mujoco.index`, `dm_control.mjcf`,
`dm_control.composer.entity`, the suite tasks, the viewer and the
locomotion/manipulation task code are object plumbing, MJCF parsing or rendering
with no array kernel to accelerate. Use `dm_control` itself for those.

## Documented divergences

Everything below is a place where the port is *not* bit-identical to upstream,
with the reason.

- **`transformations.quat_rotate`** — upstream ends with `[1:]`, which slices the
  leading axis. That is correct for a single `(4,)` quaternion but returns an
  `(n-1, 4)` array for a batch. The port drops the first *column* instead, so a
  batched input returns `(n, 3)`. Unbatched behaviour, the only shape upstream
  handles, is identical.
- **`transformations._clip_within_precision`** — upstream performs the range
  check and the clip in one function, and its error message cannot format a
  batched array (it raises `TypeError` rather than `ValueError`). The port keeps
  the check in the wrapper and the clip in the kernel, so the exception type
  matches upstream's on batched input and matches upstream's `ValueError` on
  scalar input.
- **`utils.rewards._sigmoids`** — upstream branches on a string; the kernel
  branches on an `Int`. The name-to-code mapping and all of upstream's
  validation (including that `cosine`/`linear`/`quadratic` accept
  `value_at_1 == 0` and the others do not) stay in the wrapper, and the
  per-type `scale` is computed once per call rather than per element.
- **`utils.rewards.tolerance`** — the `margin == 0` branch, and all of
  upstream's sigmoid validation, stay in the wrapper; the `margin > 0` path is
  a single kernel. It reaches the same values by a different route: it clamps
  the distance from the bounds at zero instead of selecting between the
  distance and `1.0` on an `in_bounds` test, which is exact because every
  sigmoid is 1 at argument zero. See "Branch-free fusion" under "Design
  notes". The values agree with upstream to within 3e-11 absolute across all
  eight sigmoid types on random inputs; `long_tail`, `reciprocal`, `cosine` and
  `linear` agree bit for bit and the other four differ only in the last few ulp
  of `exp`, `cosh` and `tanh`.
- **`utils.inverse_kinematics.nullspace_method`** — the Gram matrix, the
  transposed product and the `+= s * I` regulariser are one kernel;
  `np.linalg.solve` and `np.linalg.lstsq` stay in NumPy. The numerical result
  is identical; only the solve is not ours.
- **`suite.lqr_solver.solve`** — `mju_sym2dense` is a kernel.
  `scipy.linalg.solve_discrete_are` and `np.linalg.eigvals` stay in Python: a
  DARE solver and an eigenvalue scan are not small enough to be worth a
  from-scratch port, and `beta` is defined in terms of `eigvals`. The three
  `b.T @ p @ b` / `b.T @ p @ a` / `b @ k` products that used to go through the
  batched matrix-multiply kernel are NumPy `@` calls on `(2n, 2n)` arrays: that
  kernel only ever served this one function, at one batch element of at most
  `2n x n`, where NumPy's own `dgemm` is the better tool and the kernel's
  presence would only be a second way to compute the same product.
- **`mujoco.math`** — upstream forwards to MuJoCo's `mju_mulQuat`,
  `mju_negQuat`, `mju_quat2Vel`, which accept only an unbatched `(4,)`. The port
  implements the same arithmetic in Mojo and additionally accepts a leading batch
  dimension. Parity against the C helpers is asserted for the unbatched case.
- **`reference_pose.rewards.bounded_quat_dist`** — upstream normalises its
  arguments in place; the port does too, except that a non-contiguous input is
  normalised on a temporary copy (upstream would write through a strided view).
- **shape mismatches** — `bounded_quat_dist`, `compute_squared_differences` and
  `nullspace_method` index their second argument with the length of the first,
  so each checks the two shapes and raises `ValueError` on a mismatch. Upstream
  reaches the same conclusion through a broadcasting product, with a
  `ValueError` of its own wording; the check is here rather than in the kernel
  because the alternative is a buffer read past the end of an array.
- **`utils.inverse_kinematics.qpos_from_site_pose`** — the per-step
  `logging.debug` trace is omitted; the "failed to converge" warning is kept.
- **`transformations` pole warnings** — upstream logs through `absl.logging`;
  the port uses the standard library `logging` with the same messages, so this
  package does not depend on absl.
- **`rl.control`** — the `Environment` / `Physics` / `Task` abstract classes are
  not ported; they have no numerical content. Use `dm_control.rl.control` for
  those and this package for the two functions above.
- **`composer.variation`** — the variation classes are faithful ports, but
  `__eq__` compares `isinstance` against the port's own class, so
  `mojo_dm_control...UniformQuaternion() == dm_control...UniformQuaternion()`
  is `False`. Compare within one package, or compare `repr`.

Two numerical notes. `std.math.exp`, `cosh` and `tanh` in this toolchain are
polynomial approximations rather than libm calls, which is the whole of the
`3e-11` difference above; the tests for those sigmoids use a looser tolerance
than the rest. And the SIMD and worker paths do not change any result: the
transcendental SIMD functions are bit-identical to their scalar forms, and
`parallelize` only splits a row range that the serial path would have covered
in the same order, so a batch above `PARALLEL_MIN` and the same batch below it
agree. `tests/test_simd_paths.py` asserts both.

## Usage

```python
import numpy as np
from mojo_dm_control.utils import transformations as tr
from mojo_dm_control.utils import rewards

# Euler angles -> rotation matrix -> quaternion -> back again.
euler = np.array([0.3, -0.7, 1.1])
rmat = tr.euler_to_rmat(euler, ordering='ZYX')
quat = tr.mat_to_quat(rmat)
assert np.allclose(tr.quat_to_euler(quat, 'ZYX'), euler)

# Batched: a million quaternions, one call.
rng = np.random.default_rng(0)
quats = rng.normal(size=(1_000_000, 4))
quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
distances = tr.quat_dist(quats[:1], quats[1:])      # (999999, 1)

# Reward shaping.
values = rewards.tolerance(np.linspace(-3, 3, 7), bounds=(-0.5, 0.5),
                           margin=1.0, sigmoid='gaussian')
```

`examples/quickstart.py` runs all of that, plus the MuJoCo quaternion helpers:

```
pixi run build
pixi run python examples/quickstart.py
```

The package can also be driven straight through a MuJoCo environment:

```python
from dm_control import suite
from mojo_dm_control.suite import lqr_solver
from mojo_dm_control.utils import inverse_kinematics

p, k, beta = lqr_solver.solve(suite.load('lqr', 'lqr_6_2'))

physics = suite.load('hopper', 'stand').physics
ik = inverse_kinematics.qpos_from_site_pose(
    physics, 'touch_toe',
    target_pos=physics.named.data.site_xpos['touch_toe'] + np.array([0.01, 0, 0.02]),
    tol=1e-8, max_steps=200)
assert ik.success
```

## Benchmarks

Machine: Intel Xeon E5-2697 v4 @ 2.30 GHz, 2 sockets x 18 cores (36 physical
cores, 72 threads), 251 GiB RAM, Linux 6.8.0-147-generic, glibc 2.39, Python
3.13.15, NumPy 2.5.3, SciPy 1.18.1, mujoco 3.8.1,
mojo 1.2.0.dev2026092905. (Local verification on this kernel; see
`bench/verify.md` for the session that re-ran the table.)

Every number below is one run of `pixi run bench` on that machine. That task
takes a machine-wide flock, runs both implementations on identical inputs,
checks that the two agree, and reports the best of 5 runs. `ratio` is
`dm_control / mojo-dm-control`, so above 1 means the Mojo kernel is faster.
`dm_control` moves 30-50% between runs on this host, so treat the ratio as an
order of magnitude rather than a precise figure. This package's own times are
far steadier: a second full run of the table on the same build gave 7.94ms for
`quat_conj` where this one says 8.82ms and 64.03ms for `euler_to_rmat ZYX` where
this one says 60.22ms, so the ratios above move by a few points in both
directions for reasons that have nothing to do with the code.

| case | mojo-dm-control | dm_control | ratio |
| --- | ---: | ---: | ---: |
| transformations.quat_conj (1M quats) | 8.82ms | 40.82ms | 4.63x |
| transformations.quat_inv (1M quats) | 8.21ms | 90.04ms | 10.97x |
| transformations.quat_mul (1M quats) | 11.61ms | 274.46ms | 23.64x |
| transformations.quat_diff (1M quats) | 18.84ms | 204.90ms | 10.87x |
| transformations.quat_log (1M quats) | 18.40ms | 191.01ms | 10.38x |
| transformations.quat_dist (1M quats) | 94.13ms | 556.47ms | 5.91x |
| transformations.quat_rotate (100k calls) | 604.76ms | 3133.43ms | 5.18x |
| transformations.axisangle_to_quat (1M) | 34.81ms | 124.85ms | 3.59x |
| transformations.rotation_x_axis (1M) | 12.77ms | 110.10ms | 8.62x |
| transformations.euler_to_rmat ZYX (1M) | 60.31ms | 1345.09ms | 22.30x |
| transformations.euler_to_rmat XYZ full (1M) | 92.78ms | 2049.98ms | 22.09x |
| transformations.rotation_matrix_2d (1M) | 9.93ms | 36.41ms | 3.67x |
| rewards.tolerance gaussian margin=1 (1M) | 6.41ms | 22.47ms | 3.51x |
| rewards.tolerance linear margin=1 (1M) | 1.67ms | 20.51ms | 12.31x |
| rewards.tolerance margin=0 (1M) | 0.90ms | 3.24ms | 3.61x |
| mujoco.math.mj_quatprod (1M quats) | 10.78ms | 191.81ms | 17.79x |
| mujoco.math.mj_quatdiff (1M quats) | 10.30ms | 17.31ms | 1.68x |
| mujoco.math.euler2quat (20k calls) | 165.37ms | 186.35ms | 1.13x |
| reference_pose.bounded_quat_dist (1M) | 20.26ms | 130.63ms | 6.45x |
| variation.QuaternionRotate x20k calls | 253.45ms | 868.39ms | 3.43x |
| inverse_kinematics.nullspace_method x200 (6x12) | 5.62ms | 4.87ms | 0.87x |

Twenty of the twenty-one rows are faster than upstream. Two of the rows above
need reading care because their `dm_control` column is not a call of the
function named in the first column. Upstream's `mujoco.math.mj_quatprod` and
`mujoco.math.mj_quatdiff` accept only a single `(4,)` quaternion, so neither
has a batched upstream reference for 1M quats. The `mj_quatprod` row
compares the port's batched kernel against upstream's batched Hamilton
product in `utils.transformations.quat_mul` — the same arithmetic
`mju_mulQuat` computes, which `tests/test_mujoco_math.py` checks against the
C helper — and the `mj_quatdiff` row compares the port's fused kernel
against the port's own unfused pair `quat_mul(quat_conj(a), b)`, which is
the same arithmetic `mju_mulQuat(mju_negQuat(source), target)` computes.
Both rows are port-vs-reference-implementation comparisons for the batched
case, not measurements of upstream's scalar `mju_*` calls. Three rows are
not kernel benchmarks and are worth reading as such.

`nullspace_method` is the one row below 1.0, and it is below it for a structural
reason rather than a kernel one: on a 6x12 Jacobian the `(nv, nv)` Gram matrix,
the transposed product and the regulariser are 216 multiply-adds, while the 12x12
`np.linalg.solve` that consumes them is the cost both implementations actually
pay, and that solve stays in LAPACK. At 0.96x our share is lost to the ctypes
marshalling on a problem too small for the kernel to matter, and it sits close to
1.0 because there is almost nothing to win either way. Hand-writing the Cholesky
would take the row and is deliberately not done: the solve is the numerically
delicate part of this function and the tests hold it to `rtol=1e-10` against
upstream.

`euler2quat` at 20k separate calls is 20k ctypes calls of three scalar `cos`/`sin`
pairs; at 1.13x it is close to a wash, and the arithmetic it does is not worth a
crossing of the boundary on its own. `quat_rotate`'s 100k-call row is a third of
that kind: it measures the Python around the FFI, and the scalar fast path below
is what keeps it ahead at all.

Among the rows that are actually kernels, the bandwidth-bound ones (`quat_conj`,
`mj_quatdiff`) gain the least and the transcendental-bound ones gain the most,
which is what the design below is aimed at. The bandwidth-bound cases are where
NumPy is already close to memory speed, so there is simply less to take. The two
`euler_to_rmat` rows are the exception that proves the rule: they were the only
kernels here that moved data several times to produce an output once, and
removing that redundancy is worth more than any of the bandwidth-bound rows.

## Design notes

**Redundant work.** The axis rotations used to evaluate `cos` and `sin` in the
Python wrapper, discard the results, and let the kernel compute them again;
the kernel also zeroed all 9 (or 16) entries of a matrix before overwriting all
9. The wrapper now allocates with `np.empty` and the kernel writes every entry
once, with one `cos`/`sin` per angle. `tolerance` walked `x` through four
kernels and two temporaries and is now one kernel and one pass. `mj_quatdiff`
called `mju_negQuat` into a scratch buffer and `mju_mulQuat` over it, moving
160MB for a 96MB problem; it is one fused pass now. `quat_rotate` and
`nullspace_method` crossed the FFI three times each per call; both cross once.

**Fusing `euler_to_rmat`.** This is the one rewrite, and it is the largest
single win in the table: 148.31ms -> 60.31ms for `ZYX`, 362.18ms -> 92.78ms for
`XYZ full`, 9.03x -> 22.30x and 5.53x -> 22.09x. Upstream builds three
`(n, s, s)` axis rotations and contracts them with two batched products, so one
Euler vector cost five `(n, s, s)` writes and five `(n, s, s)` reads -- 360MB
moved per million rows for a result of 72MB (144MB at `full`). The kernel keeps
all three rotations in registers and writes each entry of the result once. Two
things make that cheap:

- A right multiplication by a rotation is a signed permutation of the columns,
  so `A @ Rx(t)` needs only `Ac0`, `tc * Ac1 + ts * Ac2` and `-ts * Ac1 + tc *
  Ac2`; the three contractions are nine such expressions each rather than 27
  multiplies. The branches are selected by three axis codes passed as kernel
  arguments, so the ordering resolves in the Python wrapper once per call.
- The left-most rotation is sparse and its three angles are evaluated once per
  row for all three rotations, which is what upstream did twice.

The price is two layout changes, both in the wrapper. The kernel wants the three
angle columns adjacent (so a width-`W` block of a column is a width-`W` block of
rows) and wants a packed `(n, s * s)` destination (so its writes are contiguous),
so `_euler_cols` transposes the angles once and `_euler_planes` copies the packed
result back into the `(n, s, s)` shape the API promises with a `np.copyto`.
Between them those are two passes over 3n and 9n doubles against the 9n or 16n
the composition itself writes and reads -- a quarter of the traffic upstream
moves, and the measured result is a 2.4x to 5.6x win.

**Vectorisation.** The contiguous elementwise loops (`rotation_matrix_2d`, the
axis rotations, `tolerance`, and the three angle columns of `euler_to_rmat`) run
at `comptime W = simd_width_of[DType.float64]()` with a scalar tail. The axis
rotations and `rotation_matrix_2d` write a strided destination, so a width-`W`
block of inputs is reduced to a scalar store per entry and the win there is
`std.math`'s SIMD `cos`/`sin`/`exp`. A `(n, k)` batch is interleaved rather than
strided and gets no vector body: a width-`W` load at `stride * row + k` returns
consecutive components of one row, not component `k` of `W` rows. That is why
`euler_to_rmat` transposes its input rather than gathering.

**Branch-free fusion.** `tolerance` is the clearest case. Upstream computes
`d = lower - x if x < lower else x - upper`, which is exactly
`max(lower - x, x - upper)`; that maximum is non-positive precisely where
`lower <= x <= upper`. So the distance, the sigmoid and the final
`where(in_bounds, 1.0, ...)` are all reachable without a branch: clamp the
distance at zero, and every one of the eight sigmoids is exactly 1 at argument
zero. The `in_bounds` / `dist_from_bounds` / `sigmoids` / `where_in_bounds` chain
collapses to one branch-free pass, which is also what lets the SIMD `exp`
actually be used.

**Parallelism.** `parallelize` is used, from `max.algorithm`, behind
`PARALLEL_MIN = 2^18` rows with 16 workers, on the kernels whose cost is
transcendental rather than bandwidth: the three axis rotations,
`rotation_matrix_2d`, `quat_log`, `bounded_quat_dist` and the fused
`euler_to_rmat`. The threshold is a measured crossover, not a guess: worker
launch is expensive enough on this host that below it the launch costs more than
the kernel, which is also why the bandwidth-bound kernels stay serial.

**Not the GPU.** `max.gpu.host` works on this toolchain — `DeviceContext`,
`enqueue_create_buffer`, `enqueue_copy` and `enqueue_function` all compile and
run — so the absence of a GPU path is not a missing-API excuse. It is the wrong
tool for these kernels: every one of them is bandwidth-bound at well under the
arithmetic intensity where a GPU starts to pay, and the host round trip alone
costs more than the kernel. `quat_log`, the one kernel with heavy arithmetic,
would spend more in transfer than it spends now, and it would bring a second
copy of every kernel, a device-buffer cache and a second set of parity tests.
The fused `euler_to_rmat` lowers the arithmetic intensity rather than raising
it: about 22 flops per row against the 24 bytes of angles and 72 bytes of matrix
it reads and writes, the same band as before and still far under the ~2 flops
per byte where a transfer can pay for itself.

**Tried and reverted.** A width-4 load/store in `mj_quatdiff` measured slower
than the plain scalar loop — the arithmetic mixes the four components of a
row, so the lane extraction costs more than the wider access saves — so the
simpler loop stayed. Vectorising `bounded_quat_dist`'s in-place normalisation
with a horizontal sum was not worth reordering the dot product for. A Mojo
Cholesky for `nullspace_method` is discussed above and not shipped. Two FFI
costs were measured rather than guessed, and neither is the port's to fix:
`numpy.ctypes.data` costs about 1.4us per call on this host (every `addr()` in
the package pays it), and a ctypes call with seven `double` arguments and an
`int` pointer costs about 2.0us, of which roughly 0.35us is `tolist()`. That
is what the `quat_rotate` row is made of: 100k calls at ~6.0us each, of which
~4.5us is Python and ctypes around a ~40-flop kernel. `euler2quat` at 20k calls
is the same story with three scalars per call.

**Scalar fast path.** `quat_rotate` on a single `(4,)` quaternion and a single
`(3,)` vector — the shape dm_control's task and variation code actually passes
— needs no reshape, no broadcast and no copy. It converts the two arguments
with `tolist()` and calls one `abi("C")` function that takes the seven
components by value. The batched path is unchanged.

## How it works

**FFI.** `build/build.sh` compiles the single translation unit `src/ported.mojo`
with `mojo build --emit shared-lib` into `dist/libmojo-dm-control.so`. Every
exported entry point is a thin `abi("C")` wrapper around a kernel:

```mojo
@export("mc_quat_mul")
def mc_quat_mul(q1: Int, q2: Int, dst: Int, n: Int) abi("C"):
    quat_mul(p(q1), p(q2), p(dst), n)
```

`@export` rejects parametric functions, so no Mojo type parameter survives the
boundary: the wrapper takes `Int` addresses and rebuilds typed pointers inside.

**Memory layout.** Buffers cross as `Int` addresses and nothing in the Mojo
file allocates, so the caller owns every byte, including scratch. A batched
upstream function such as `quat_mul(quat1, quat2)` takes `(..., 4)` arrays; the
Python wrapper in `_lib.py` reshapes them to `(n, 4)` and the kernel indexes row
`b` at `4 * b + k`. `matrices` are `(n, 3, 3)` or `(n, 4, 4)` and index
`ts * b + rs * row + col`, except `euler_to_rmat`, which takes three contiguous
angle columns and writes a packed `(n, s * s)` batch because that is the layout
its vector body and its registers want; the wrapper transposes in and unpacks
out. The integer arrays `M_rownnz`/`M_rowadr`/`M_colind` cross as
`Pointer[Int, AnyOrigin[mut=True]]`. Every array is made C-contiguous float64 on
the way in, and a copy is made only when the input is not already contiguous,
so the common path is a pointer and a length.

**Where the divergence lives.** The kernels keep upstream's structure. Anything
Mojo forces out of a kernel — the sigmoid name-to-code mapping, the `margin == 0`
branch in `tolerance`, the `ValueError` checks — stays in the Python wrapper,
where it is still one line above the call it guards. `src/ported.mojo` is
therefore the arithmetic; `python/mojo_dm_control/**` is the API.

**Vector width and workers.** `comptime W = simd_width_of[DType.float64]()`
(4 on this target) is the only width in the file. A kernel whose body is
branch-free is written as a `while b + W <= n` vector loop plus a scalar tail;
a kernel that needs a select is left scalar, because this toolchain's
`SIMD[DType.bool]` is not usable for a blend — `&` on two comparison masks and
`SIMD[DType.float64](SIMD[DType.int64](mask))` both compile and both return
garbage, so `tolerance` reaches the same result arithmetically instead. The
worker-driven kernels are split into a `_..._range(lo, hi)` body and a driver
that hands contiguous chunks to `max.algorithm.parallelize` when `n` clears
`PARALLEL_MIN`. `parallelize` needs the `max` package beside the `mojo` pin in
`pixi.toml`; without it the import fails.

**ctypes.** `_lib.lib()` caches the `CDLL` and `_lib.fn(name)` caches the
resolved symbol, so a wrapper pays one dict lookup per call and never re-walks
the library. Per-call Python overhead still dominates the cheap kernels, which
is why `quat_rotate` has a scalar path that crosses the boundary once with the
components passed by value: the `quat_rotate` row of the benchmark table is
made of 100k such calls.

## Layout

```
src/ported.mojo                     every kernel, one compilation unit
build/build.sh                      mojo build --emit shared-lib -> dist/
python/mojo_dm_control/
    _lib.py                         ctypes loader, builds on demand
    mujoco/math.py                  <- dm_control.mujoco.math
    utils/transformations.py        <- dm_control.utils.transformations
    utils/rewards.py                <- dm_control.utils.rewards
    utils/inverse_kinematics.py     <- dm_control.utils.inverse_kinematics
    rl/control.py                   <- dm_control.rl.control
    suite/lqr_solver.py             <- dm_control.suite.lqr_solver
    composer/variation/rotations.py <- dm_control.composer.variation.rotations
    composer/variation/base.py      <- the Variation algebra the variations use
    locomotion/tasks/reference_pose/rewards.py
tests/                              pytest parity tests against upstream
    test_kernels.py                  every @export symbol, called by name
    test_simd_paths.py               vector tails and the parallel threshold
bench/bench.py                      benchmarks, prints the table above
bench/profile_local.py              the same cases, this package only
examples/quickstart.py              the usage example above, runnable
```

`python -m mojo_dm_control._lib [--force]` rebuilds the shared library; the
loader also builds it on first import if `dist/` is missing or stale.

`pixi.toml` pins `mojo` and `max` to matching versions. `max` is what
`max.algorithm.parallelize` lives in on this toolchain; without it the workers
do not compile.

## Tests

`pixi run test` runs 551 tests that compare against the installed `dm_control`
package: every function in the covered subset, batched and unbatched, all seven
Euler orderings including their pole branches, all eight sigmoid types, the
scalar and `ValueError` paths, MuJoCo's own `mju_*` C helpers, and end-to-end
runs of `qpos_from_site_pose` and `lqr_solver.solve` against real
`dm_control.suite` environments. `dm-control` is a dev dependency precisely so
these comparisons run against the real thing rather than a reimplementation.

`tests/test_kernels.py` is the other half of that. Everywhere else the kernels
are reached through the Python API, which is what a caller uses, and a kernel
that is wired up but wrong is invisible there: the API path and the kernel
agree, so nothing fails. That module crosses the ctypes boundary into each of the
26 `@export` symbols directly, by name, and holds each one to numerical
parity — the `mju_*` kernels against MuJoCo's own C helpers, `mc_sym2dense`
against a hand expansion of a real model's sparse mass matrix, `mc_euler_to_rmat`
row by row against upstream for all seven orderings, the rest against NumPy,
SciPy or upstream. It also carries the coverage rule as an assertion:
`test_every_exported_symbol_has_a_test` re-derives the `@export` set from
`src/*.mojo` and fails the run if any symbol has no test naming it, so a kernel
added without one cannot ship green. `test_library_really_is_the_compiled_one`
and a module-level skip keep the suite honest about the fact that it is
exercising `dist/*.so` and not a Python fallback.

`tests/test_simd_paths.py` covers what the parity tests structurally cannot:
lengths that put an element in every vector loop's scalar tail, the same
element computed through the vector body and through the scalar tail compared
for bit equality, and the same input processed just below and just above
`PARALLEL_MIN` so the serial and worker paths are compared against each other as
well as against upstream. The fused `euler_to_rmat` is in both halves: its
kernel test drives the packed buffer and the transposed angle columns directly,
and its SIMD tests walk every length that puts a row in the vector body, in the
tail, or across the worker boundary.

One note for anyone writing new kernel tests: `addr()` hands the kernel a raw
pointer and does not keep the buffer alive, so a temporary built inline
(`addr(buf.copy())`) is freed before the kernel runs and the write lands in
freed heap. `bounded_quat_dist` normalises its arguments in place, so its test
binds every buffer to a name. `_lib.addr` carries the same warning.

## Licence

The ported algorithms come from
[google-deepmind/dm_control](https://github.com/google-deepmind/dm_control),
which is Apache 2.0; this package is distributed under the MIT licence in
`LICENSE`.
