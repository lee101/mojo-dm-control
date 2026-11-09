# mojo-dm-control — ANALYZE: where the time goes

Measured 2026-11-09 on the bench host. Everything below comes from runs done in
this session; no estimates. Numbers are "best of 5" (`timeit` in `bench/bench.py`
style) unless noted. `pixi run bench` (flocked, single job at a time) is the
source of the upstream ratios; kernel-only times are direct ctypes calls with
preallocated buffers (no wrapper, no `dst` allocation); API times are the public
Python entry points; numpy baselines are vectorised NumPy on the same inputs.

Host: Linux-6.8.0-147-generic-x86_64, 72 CPUs (2× Intel Xeon E5-2697 v4,
2.3 GHz, AVX2), Python 3.13.15, NumPy 2.5.3. `dm_control` importable (the bench
imports `dm_control.utils.transformations`, `rewards`, `mujoco.math`,
`inverse_kinematics` and runs; no `__version__` attribute, so no pin is quoted).
Mojo toolchain per `MOJO_NOTES.md` (`mojo ==1.2.0.dev2026092905`). See
`MOJO_NOTES.md` §0 on `MODULAR_HOME` and §6 on build/ctypes cost.

## 1. Hotspot table (1M rows unless noted)

Time = public API time (the number a caller pays). Bytes touched = minimal
one-pass traffic (reads + writes) for that batch, in MiB (N = 1,000,000).
Flops = arithmetic per row × N, counted from the kernel source; the
transcendentals (sin/cos/log/acos/...) are counted as ~20 flop-equivalents
each only for the flops/byte ratio; the point of the ratio is the order of
magnitude, not the last digit. Kernel vs glue = direct-kernel time vs
(API − kernel).

| rank | function (input) | time (API) | kernel only | bytes touched | flops | flops/byte | bound | Python glue / kernel |
|---:|---|---|---|---|---|---|---|---|
| 1 | `transformations.quat_rotate` — 100k scalar calls (1 quat + 1 vec each) | 640.69 ms (bench) / 684.35 ms (re) | ~1.6 ms batched | 100k × 56 B = 5.4 MiB | ~32/row | 0.6 | glue-bound | ~99% Python/ctypes (100k FFI crossings at ~6.8 µs each) |
| 2 | `variation.QuaternionRotate` — 20k `__call__`s | 267.37 ms (bench) / 285.87 ms (re) | ~0.3 ms batched equivalent | 20k × 56 B ≈ 1.1 MiB | ~32/row | 0.6 | glue-bound | ~99% Python (`evaluate` + `quat_rotate` scalar path) |
| 3 | `mujoco.math.euler2quat` — 20k calls (1 angle triple each) | 126.30 ms (bench) / 132.82 ms (re) | ~3–6 µs/call, no batched kernel exists | 20k × 56 B ≈ 1.1 MiB | ~40/row + 6 trig | ~1 | glue-bound | 100% Python loop — no batched path to call |
| 4 | `transformations.quat_dist` — 1M | 92.31 ms (bench) / 87.43 ms (re) | 94.64 ms | 68 MiB (64 in + 8 out... see below) | ~150/row (inv+mul+norm+log+norm) | ~2 | compute (transcendentals) | kernel (~100%); glue ~0 |
| 5 | `transformations.euler_to_rmat` XYZ full — 1M | 86.99 ms (bench) / 88–111 ms (re) | 17.45 ms | 24 in + 128 out + copies | ~90/row | <1 | glue-bound | **~70–85% Python** (copy-back dominates) |
| 6 | `transformations.euler_to_rmat` ZYX — 1M | 58.51 ms (bench) / 60.02 ms (re) | 16.47 ms | 24 in + 72 out + copies | ~60/row | <1 | glue-bound | **~73% Python** (copy-back dominates) |
| 7 | `transformations.axisangle_to_quat` — 1M | 35.47 ms (bench) / 33.72 ms (re) | 40.70 ms | 56 MiB (24 in + 32 out) | ~50/row + trig | ~1 | kernel (sin/cos/sqrt) | kernel (~100%) |
| 8 | `transformations.quat_diff` — 1M | 18.71 ms (bench) / 17–22 ms (re) | 22 ms (conj+mul) | 96 MiB + temp 32 MiB | ~64/row | 0.7 | glue + 2× kernel | 2 FFI calls + intermediate (n,4) buffer |
| 9 | `reference_pose.bounded_quat_dist` — 1M (with 2 `.copy()`s in the bench lambda) | 20.55 ms (bench) / 23.28 ms (re); 11.26 ms reused in-place | 12.30 ms | 68 MiB | ~40/row + acos | ~0.6 | kernel | kernel; ~9 ms of the 20 ms is the two caller copies |
| 10 | `transformations.quat_log` — 1M | 19.83 ms (bench) / 18.80 ms (re) | 11.40 ms | 64 MiB | ~50/row + log/acos | ~0.8 | kernel + glue | ~40% Python (`_clip_within_precision` clip pass: 6.37 ms) |
| 11 | `transformations.rotation_x_axis` — 1M | 12.57 ms (bench) / 14.41 ms (re) | 10.94 ms | 80 MiB (8 in + 72 out) | ~10/row + sin/cos | ~0.2 | memory-bound | kernel; small glue |
| 12 | `transformations.rotation_matrix_2d` — 1M | 9.37 ms (bench) / 9.24 ms (re) | 8.73 ms | 40 MiB (8 in + 32 out) | ~10/row + sin/cos | ~0.3 | memory-bound | kernel; transpose is a view (0.00 ms) |
| 13 | `transformations.quat_mul` — 1M | 12.27 ms (bench) / 13.33 ms (re) | 12.09 ms | 96 MiB | ~56/row | 0.6 | memory-bound | kernel (~90%) |
| 14 | `transformations.quat_conj` — 1M | 8.56 ms (bench) / 9.03 ms (re) | 8.51 ms | 64 MiB | 3/row | 0.05 | memory-bound | kernel (~95%) |
| 15 | `transformations.quat_inv` — 1M | 8.32 ms (bench) / 8.99 ms (re) | 8.38 ms | 64 MiB | ~15/row | 0.2 | memory-bound | kernel (~95%) |
| 16 | `rewards.tolerance` gaussian margin=1 — 1M | 7.06 ms (bench) / 7.33 ms (re) | 7.54 ms | 16 MiB | ~30/row + exp | ~2 | kernel (SIMD) | kernel (~100%) |
| 17 | `rewards.tolerance` linear margin=1 — 1M | 1.54 ms (bench) / 1.81 ms (re) | 1.76 ms | 16 MiB | ~5/row | 0.3 | memory-bound | kernel (~100%) |
| 18 | `rewards.tolerance` margin=0 (`mc_in_bounds`) — 1M | 0.95 ms (bench) / 0.97 ms (re) | 0.99 ms | 16 MiB | 2/row | 0.1 | memory-bound | kernel (~100%) |
| 19 | `mujoco.math.mj_quatprod` — 1M | 10.62 ms (bench) / 9.49 ms (re) | 11.88 ms | 96 MiB | ~56/row | 0.6 | memory-bound | kernel (~100%) |
| 20 | `mujoco.math.mj_quatdiff` — 1M | 9.39 ms (bench) / 10.13 ms (re) | 11.64 ms | 96 MiB | ~56/row | 0.6 | memory-bound | kernel (~100%) |
| 21 | `inverse_kinematics.nullspace_method` — 200× (6×12) | 5.98 ms (bench) / 5.03 ms (re) | 7.24 µs/call → 1.45 ms/200 | tiny (≤1 KiB/call) | ~1.7k/call | ≫10 | NumPy-solve-bound | ~70% NumPy `solve`/`lstsq` + Python, ~30% kernel |

Bytes detail (1M): `(n,4)` float64 = 32 MiB; `(n,3)` = 24 MiB; `(n,)` = 8 MiB;
`(n,3,3)` = 72 MiB; `(n,4,4)` = 128 MiB. `quat_dist`: 32+32 reads, 8 write.
`bounded_quat_dist`: same. `euler_to_rmat`: input 24 MiB, output 72 or 128 MiB,
plus a 24 MiB `_euler_cols` copy and a full output copy-back.

`pixi run bench` raw (ratio = upstream/mojo; >1 = Mojo faster; best of 5):

```
transformations.quat_conj (1M quats)                        8.56ms     40.94ms    4.79x
transformations.quat_inv (1M quats)                         8.32ms     90.84ms   10.92x
transformations.quat_mul (1M quats)                        12.27ms    279.58ms   22.78x
transformations.quat_diff (1M quats)                       18.71ms    258.75ms   13.83x
transformations.quat_log (1M quats)                        19.83ms    188.60ms    9.51x
transformations.quat_dist (1M quats)                       92.31ms    519.99ms    5.63x
transformations.quat_rotate (100k calls)                  640.69ms   3283.77ms    5.13x
transformations.axisangle_to_quat (1M)                     35.47ms    121.21ms    3.42x
transformations.rotation_x_axis (1M)                       12.57ms    107.15ms    8.52x
transformations.euler_to_rmat ZYX (1M)                     58.51ms   1400.66ms   23.94x
transformations.euler_to_rmat XYZ full (1M)                86.99ms   2099.25ms   24.13x
transformations.rotation_matrix_2d (1M)                     9.37ms     41.51ms    4.43x
rewards.tolerance gaussian margin=1 (1M)                    7.06ms     25.98ms    3.68x
rewards.tolerance linear margin=1 (1M)                      1.54ms     19.43ms   12.62x
rewards.tolerance margin=0 (1M)                             0.95ms      3.54ms    3.73x
mujoco.math.mj_quatprod (1M quats)                         10.62ms    191.88ms   18.06x
mujoco.math.mj_quatdiff (1M quats)                          9.39ms     17.02ms    1.81x
mujoco.math.euler2quat (20k calls)                        126.30ms    157.09ms    1.24x
reference_pose.bounded_quat_dist (1M)                      20.55ms    151.85ms    7.39x
variation.QuaternionRotate x20k calls                     267.37ms    899.12ms    3.36x
inverse_kinematics.nullspace_method x200 (6x12)             5.98ms      6.09ms    1.02x
```

No row is slower than upstream (minimum ratio is `nullspace_method` at
1.02× and `euler2quat` at 1.24×). There is nothing to mark as "port slower
than upstream"; the two near-parity rows are called out in §3 because they are
floor time, not a regression.

NumPy baselines (vectorised, 1M, best of 5; from `bench/_analyze3.py`):

```
numpy quat_conj 1M                         20.20 ms   (mojo 9.03 ms — 2.2x)
numpy quat_inv 1M                          73.42 ms   (mojo 8.99 ms — 8.2x)
numpy quat_mul 1M                         187.89 ms   (mojo 13.33 ms — 14.1x)
numpy tolerance gauss 1M                   17.36 ms   (mojo 7.33 ms — 2.4x)
numpy rotation_matrix_2d 1M                87.03 ms   (mojo 9.24 ms — 9.4x)
numpy axisangle_to_quat 1M                158.35 ms   (mojo 33.72 ms — 4.7x)
```

## 2. What the table says — the five things that matter

### 2.1 The two biggest hotspots are not kernels. They are Python loops.

`quat_rotate` over 100k single calls (640.69 ms) and `QuaternionRotate` over
20k calls (267.37 ms) are ~50% of the total bench time between them, and both
are a Python `for` that re-enters ctypes once per element (`quat_rotate`
scalar path: 6.84 µs/call; `QuaternionRotate.__call__` goes through
`variation_values.evaluate` → `quat_rotate`). The batched kernel that already
exists (`mc_quat_rotate`) does the same 100k rows in 1.59 ms. The bench
itself measures this correctly — both ports loop in Python — but a caller who
has arrays in hand should not be looping; the actionable gap is a batched
entry point for these two functions (see §4). This is the single largest
available win in the tree measured in milliseconds.

### 2.2 `euler_to_rmat` pays 3–5× in Python glue on top of its kernel.

Direct kernel: 16.47 ms (ZYX) / 17.45 ms (XYZ full). Public API: 58.51 /
86.99 ms. The difference is measured, step by step:

- `_euler_cols` (transpose-copy of `(n,3)` → `3n` contiguous columns): 6.90 ms
- `mc_euler_to_rmat` kernel: 15.16 ms
- `_euler_planes` copy-back `(n,s,s)` → result: **59.11 ms** (s=3) /
  **99.37 ms** (s=4) — the dominant term, larger than the kernel itself by
  3.9×/5.7×.

In other words, the "24× faster than upstream" headline is currently bought
with a full extra pass over the output that costs more than the computation.
`euler_to_quat` (91.85 ms API) inherits it because it calls `euler_to_rmat`
then `mat_to_quat`. `rotation_matrix_2d` does **not** have this problem — its
`.transpose(1,2,0)` is a view (0.00 ms) — which is why its API (9.24 ms) is
within 0.5 ms of its kernel (8.73 ms). The copy-back has a written `NOTE`
in the wrapper claiming `np.copyto` is "a strided memcpy with no intermediate".
It is, but a strided memcpy over 72–128 MiB is exactly the most expensive
thing in this function. This is the largest *kernel-adjacent* hotspot by
absolute milliseconds.

### 2.3 Everything else big is a single memory-bound pass — and that is fine.

`quat_conj` (64 MiB in 8.51 ms → 7.7 GB/s), `quat_inv`, `quat_mul`,
`rotation_matrix_2d`, `tolerance` (16 MiB in 0.97 ms → 17 GB/s) are all
Streaming one read + one write. Flops/byte 0.05–0.6. They run at or near
what a single-threaded streaming loop can do; the numpy baselines show the
same shape (2–14× slower because NumPy allocates intermediates). Nothing
structural is wrong here.

The two exceptions are `quat_dist` (94.64 ms kernel) and
`axisangle_to_quat` (40.70 ms), both carrying `sqrt`/`acos`/`log`/`sin`/`cos`
per row. Their flops/byte is still only ~1–2 (the transcendentals dominate
but do not change the arithmetic intensity class). They are the only two
where a compute optimisation (vectorised transcendentals) is even on the
table, and even there the next phase is `simd`, not `gpu` — see §4.

### 2.4 Per-call overhead is measurable and regressive at small n.

Single-call (n=1) API overhead, best of 50 (`bench/_analyze4.py`):

```
quat_conj n=1             52.92 us
quat_mul n=1             101.00 us
quat_log n=1              73.11 us
quat_dist n=1             99.33 us
tolerance n=1              9.07 us
euler2quat single          4.19 us
mj_quatneg n=1             49.02 us
```

Why `tolerance` (9 µs) and `euler2quat` (4 µs) are 5–10× cheaper than the
`quat_*` family: they call `fn(name)` (the cached-lookup helper) while the
`quat_*` wrappers call `lib()` + attribute access per call, and `quat_mul`
additionally does `np.broadcast_arrays`. `euler2quat`'s scalar kernel takes
three floats + one address — no `rows()`/`np.asarray`/`np.empty((n,w))`
plumbing — and is the evident floor for what a ctypes wrapper costs
(~4 µs). The scalar entries in the bench (`quat_rotate` 100k, `euler2quat`
20k, `QuaternionRotate` 20k) all live at this floor, not at kernel time.

### 2.5 `nullspace_method` is not a Mojo problem at this size.

5.98 ms (bench) / 5.03 ms (re) for 200 calls of (6×12); upstream 6.09 / 4.44
ms — the only row where a re-run flipped the sign. Split (single call,
best of 200): `mc_nullspace_products` 7.24 µs; `np.linalg.solve` 9.02 µs;
`np.linalg.lstsq` 36.84 µs. The 200-call batch (reg strength 3e-2 →
`solve` path) is 5.03 ms = 25.13 µs/call, i.e. ~7 µs kernel + ~9 µs solve +
~9 µs Python/alloc (`ascontiguousarray` ×2, `np.empty` ×2, attribute
lookups). The kernel is 29% of it. See §4 — `none`.

### 2.6 `quat_diff` is two kernels + a temp. `mj_quatdiff` is one.

`quat_diff` (18.71 ms) = `quat_conj` (8.5 ms) + `quat_mul` (12 ms) with a
32 MiB `(n,4)` intermediate. `mj_quatdiff` does the identical arithmetic in
one pass (9.39 ms API / 11.64 ms direct — the direct figure is higher because
it is timed without the wrapper's warm buffers; the API figures are the
comparable ones). That is a 2× gap on the exact same computation,
documented in the kernel header ("mju_mulQuat(mju_negQuat(source), target)
in one pass"). `quat_diff` trialling at 17–22 ms across runs (and 33–45 ms
in one session — see §5, Open questions) is consistent with
`conj + mul + temp` and inconsistent with anything else. The 20.86 ms in
`bench/_analyze4.py` and 41.14 ms in `bench/_analyze3.py` are the same binary
on the same day; the variance is contention, not algorithm — treat 17–22 ms
as the number.

## 3. Rows near parity with upstream — flagged, not dropped

- **`nullspace_method` 1.02×** (bench) / **0.89×** in the re-run (5.03 vs 4.44
  ms). This is measurement noise on ~5 ms of mostly-NumPy time, not a
  regression. Verdict in §4 is `none`; the dense product is 29% of the call
  and the solve is LAPACK either way. If the ratio matters, the phase that
  can move it is `parallel` across the 200 calls (batch the Jacobians), not
  inside one 6×12.
- **`euler2quat` 1.24×.** Both sides are Python loops over 20k calls ending in
  a 4-float ctypes crossing; the Mojo side is 4.19 µs/call vs 7.71 µs/call
  upstream because it does not go through MuJoCo's `mju_*` autowrap. There is
  no batched `euler2quat` kernel to compare. Parity here means "both are
  glue-bound", not "the kernel is almost there".
- **`mj_quatdiff` 1.81×** — the smallest batched ratio, and the honest floor
  of the batch: both implementations are one streaming pass; upstream is
  MuJoCo's C via autowrap (17.02 ms), we are Mojo direct (9.39 ms). The
  ratio is the C-vs-Mojo streaming gap, not a headroom signal.

No row in `pixi run bench` has the port slower than upstream. That claim is
the bench's own output (ratio column, §1), reproduced verbatim above.

## 4. Python-level loops, allocations, and copies

**Python loops that re-enter ctypes per element (the primary flag):**

1. `transformations.quat_rotate` called per-element — bench `quat_rotate
   (100k calls)`, 640.69 ms at 6.84 µs/call. Cause: wrapper has a scalar fast
   path (`mc_quat_rotate_scalar`) but the *bench and the typical caller* loop
   in Python; there is no `(quat (n,4), vec (3,))` → `(n,3)` batched entry
   point exercised by that loop (the batched `mc_quat_rotate` requires both
   args batched). The fix is a wrapper change, not a kernel change:
   accept `(n,4) × (3,)` and broadcast on the vec side.
2. `mujoco.math.euler2quat` per-element — bench `euler2quat (20k calls)`,
   126.30 ms at 6.64 µs/call. Cause: kernel is scalar-only by signature
   (`mc_euler2quat(ax, ay, az, dst)`); no batched symbol exists. Same
   wrapper-shape fix as (1).
3. `variation.QuaternionRotate.__call__` per-element — bench
   `QuaternionRotate x20k`, 267.37 ms. Cause: `__call__` → 
   `variation_values.evaluate` (tree.map_structure) → `quat_rotate` scalar
   path. The kernel cost inside is negligible; the three Python layers
   (`evaluate`, `__call__` dispatch, scalar `quat_rotate`) are the time.
   Fix belongs to (1).

**Per-call allocations/copies that are not the output:**

- `euler_to_rmat`: `_euler_cols` allocates `3n` doubles (24 MiB at 1M) and
  transposes into them (6.90 ms); `packed` buffer `9n`/`16n`; then
  `_euler_planes` allocates *another* `9n`/`16n` (72/128 MiB) and copies into
  it (59.11 / 99.37 ms). Three large buffers touched where one suffices if
  the kernel wrote the `(n,s,s)` layout directly. This is the
  `allocations or copies` flag for the whole tree — it is 60–80% of this
  function's time.
- `quat_log`: `_clip_within_precision` does a full `(n,1)` subtract + clip
  pass (6.37 ms) **before** the kernel, which then does the same `acos(clip)`
  per row. The clip is validation (upstream raises on out-of-range `w`),
  but it is a second pass over `w` that a scalar check per row in the kernel
  could absorb. Measured: dropping it would take `quat_log` from ~19 ms to
  ~12 ms (kernel time).
- `bounded_quat_dist` bench lambda does `a.copy(), b.copy()` (64 MiB of
  copying, ~9 ms) so the in-place normalisation does not destroy the caller's
  arrays. Caller-side, not wrapper: reusing the buffers drops API time from
  23.28 → 11.26 ms. Noted here so the next phases do not "optimise" a bench
  artifact.
- `quat_diff`: allocates a `(n,4)` intermediate (32 MiB) between the two
  kernel calls. `mj_quatdiff` proves it is avoidable in one pass.
- `tolerance`: `rows()` → `np.asarray(..., dtype=float64)` + `reshape` is
  free for contiguous input (0.00 ms measured for `broadcast_arrays` /
  `ascontiguousarray` / `np.empty` at 1M). No flag.

## 5. Verdicts for the next phases

One verdict per hotspot: `unroll` (scalar-tail/loop restructure),
`simd` (width-`W` vector body), `parallel` (row-range `parallelize`, already
present at `PARALLEL_MIN = 1<<18`), `gpu` (offload), `none` (leave it).
A `gpu` verdict requires a flops/byte figure > ~2 **and** a reason to pay
the PCIe round trip; `MOJO_NOTES.md` §5 states a CPU-only port is
legitimate below ~2 flops/byte. The relevant threshold anchor measured here:
at 1M rows even the memory-bound kernels run 8–12 ms single-threaded; the
`PARALLEL_MIN` crossover measured by the size sweep (§6) is where per-row
cost drops from 41 ns (n=100k, serial) to 12 ns (n=1M, parallel) — but the
parallel path is *already in* the kernels, so `parallel` as an action means
"lower `PARALLEL_MIN` / add a missing row-range split", not "add
parallelism where none exists".

| function | verdict | flops/byte behind it |
|---|---|---|
| `quat_rotate` batched entry point | **none** (kernel) — add a batched wrapper | 0.6; the 430× gap is 100k ctypes crossings, not arithmetic |
| `euler2quat` batched entry point | **none** (kernel) — add a batched wrapper | ~1; same reason |
| `QuaternionRotate` | **none** (kernel) — inherits the `quat_rotate` wrapper fix | 0.6 |
| `euler_to_rmat` (both) | **none** (kernel) — delete `_euler_planes` copy-back, write `(n,s,s)` directly; delete `_euler_cols` transpose via a gather or strided kernel | <1; the copy is a *fourth* pass, not intensity |
| `quat_diff` | **unroll** (fuse to one pass, as `mj_quatdiff` already does) | 0.7; 2 passes + temp → 1 pass, 2× measured |
| `quat_log` | **unroll** (absorb the clip into the kernel) | 0.8 |
| `quat_dist`, `axisangle_to_quat` | **simd** (vectorised sin/cos/sqrt/acos/log body) | ~2 / ~1; the only two where transcendentals are a material share — `simd` because they are still streaming, so SIMD transcendentals is the whole play; `gpu` would need ≥10× that intensity to justify copy-in/out |
| `bounded_quat_dist` | **parallel** (already has the row-range split; verdict is "lower `PARALLEL_MIN` if real callers pass ≥262k rows" — the 1M bench already benefits) | 0.6; `parallel` only as a threshold tweak |
| `quat_conj`, `quat_inv`, `quat_mul`, `mj_quatprod`, `mj_quatdiff`, `rotation_*`, `rotation_matrix_2d`, `tolerance` (all sigmoids), `mc_in_bounds` | **none** | 0.05–0.6; memory-bound streaming, already at 8–17 GB/s |
| `nullspace_method` | **none** | — (tiny; 29% kernel, rest LAPACK + Python) |
| `lqr_solver.solve` | **none** | — (1.18 / 3.42 ms, parity with upstream; DARE + `eigvals` stay in SciPy/NumPy by design) |

**On GPU, explicitly:** no hotspot in this tree reaches the flops/byte that
makes a GPU path pay. The highest is `quat_dist` at ~2, and it is also the
one where the PCIe copy of 64 MiB in + 8 MiB out (at ~25 GB/s host link, ~3 ms)
would have to be amortised against an 88 ms compute saving that SIMD
transcendentals can capture in place. `MOJO_NOTES.md` §5's rule ("below
roughly 2 flops per byte, CPU-only is legitimate") holds for every row
here. **No `gpu` verdicts.** If a future phase wants to revisit, the
threshold-crossing candidate would be a *batched* `quat_dist` at n ≥ 10M
where the streaming time exceeds the transfer — that size does not occur in
`bench/bench.py` or in `tests/`.

**On `parallel` (general):** five kernels already split rows at
`n ≥ 262,144` (`axis_rotation`, `euler_to_rmat`, `quat_log`,
`rotation_matrix_2d`, `bounded_quat_dist`). The size sweep below shows the
crossover is real (per-element cost 41 ns → 12 ns between 100k and 1M on the
same kernel). The kernels *without* the split (`quat_conj`, `quat_mul`,
`quat_dist`, `axisangle_to_quat`, `tolerance_margin`) do not need it — they
are already at memory bandwidth in one core — except `quat_dist` if the
transcendentals are first SIMD'd and then become the bottleneck; in that
case the next phase is `parallel` *after* `simd`.

## 6. Supporting measurements

**Size sweep — axis_rotation direct (same kernel, serial → parallel crossover
at `PARALLEL_MIN = 262,144`):**

```
n=10      0.009 ms (944.5 ns/elem)
n=100     0.012 ms (124.1 ns/elem)
n=1000    0.046 ms (45.8 ns/elem)
n=10000   0.423 ms (42.3 ns/elem)
n=100000  4.140 ms (41.4 ns/elem)   <- serial
n=262144  6.581 ms (25.1 ns/elem)   <- parallel split kicks in
n=500000  8.532 ms (17.1 ns/elem)
n=1000000 12.064 ms (12.1 ns/elem)
```

So `PARALLEL_MIN` sits exactly where the serial per-element cost would
otherwise dominate; below it per-call overhead is 0.5–1 µs and irrelevant.

**Kernel-only 1M (direct ctypes, preallocated, best of 5):**

```
bounded_quat_dist direct 1M                12.30 ms
mj_quatprod direct 1M                      11.88 ms
mj_quatdiff direct 1M                      11.64 ms
euler_to_rmat ZYX direct 1M                16.47 ms
euler_to_rmat XYZ full direct 1M           17.45 ms
quat_conj direct 1M                         8.51 ms
quat_inv direct 1M                          8.38 ms
quat_mul direct 1M                         12.09 ms
quat_log direct 1M                         11.40 ms
quat_dist direct 1M                        94.64 ms
axisangle_to_quat direct 1M                40.70 ms
rotation_x_axis direct 1M                   10.94 ms
rotation_matrix_2d direct 1M                8.73 ms
tolerance gauss direct 1M                   7.54 ms
tolerance linear direct 1M                  1.76 ms
in_bounds direct 1M                         0.99 ms
```

**Public API 1M (best of 5):**

```
quat_conj api 1M                            9.03 ms
quat_inv api 1M                             8.99 ms
quat_mul api 1M                            13.33 ms
quat_diff api 1M                           41.14 ms  (see Open questions)
quat_log api 1M                            18.80 ms
quat_dist api 1M                           87.43 ms
axisangle_to_quat api 1M                   33.72 ms
rotation_x_axis api 1M                     14.41 ms
rotation_matrix_2d api 1M                   9.24 ms
tolerance gauss api 1M                      7.33 ms
tolerance linear api 1M                     1.81 ms
tolerance margin0 api 1M                    0.97 ms
mj_quatprod api 1M                          9.49 ms
mj_quatdiff api 1M                         10.13 ms
euler_to_rmat ZYX api 1M                   61.62 ms
euler_to_rmat XYZ full api 1M             111.03 ms
bounded_quat_dist api 1M                   23.19 ms
euler_to_quat api 1M                       91.85 ms
```

**`euler_to_rmat` glue breakdown (1M, best of 5):**

```
_euler_cols 1M: 6.90 ms                            (24 MiB transpose-copy)
kernel from cols (excl copy-back): 15.16 ms
_euler_planes copy-back 1M (n=3): 59.11 ms       (72 MiB copy-back — dominates)
_euler_planes copy-back 1M (n=4): 99.37 ms       (128 MiB copy-back — dominates)
rot2d transpose view 1M: 0.00 ms                  (view, not a copy — the contrast case)
```

**`quat_log` glue:** `np.clip` pass 6.37 ms; `quat_log` API 19.19 ms;
kernel alone 11.40 ms. The clip is ~33% of the API.

**`quat_diff` breakdown:** API 17–22 ms (17.39 ms best interleaved re-run;
18.71 ms bench; 20.86 ms and 41.14 ms in two separate sessions — §5 Open
questions) vs `mj_quatdiff` 9.39–10.13 ms. Manual `quat_mul(quat_conj(a), b)`:
16.22–22.45 ms. Fused one-pass is the 2×.

**`bounded_quat_dist` glue:** API with `.copy()`s 23.28 ms; API reusing
buffers 11.26 ms; kernel 12.30 ms. ~9–12 ms is caller copies.

**`nullspace_method` split (single call, best of 200):**
`mc_nullspace_products` 7.24 µs; `np.linalg.solve` 9.02 µs; `np.linalg.lstsq`
36.84 µs. 200-call totals: Mojo 5.03 ms (25.13 µs/call), upstream 4.44 ms.

**`lqr_solver.solve`:** Mojo 1.18 ms / upstream 1.18 ms (`lqr_2_1`);
Mojo 3.42 ms / upstream 4.04 ms (`lqr_6_2`). The Mojo contribution
(`mc_sym2dense`) is one sparse expansion; the DARE and `eigvals` dominate.

## 7. Open questions

1. **`quat_diff` variance 17–45 ms.** `bench/bench.py` says 18.71 ms;
   a clean interleaved re-run gives 17.39 ms (and 21–22 ms across three
   rounds); `bench/_analyze3.py` measured 41.14 ms. Same binary, same day,
   same 1M inputs. Suspected cause: scheduler/thermal contention on the
   72-CPU host (two kernel crossings + a 32 MiB intermediate make it the
   most allocation-sensitive batched call). Not blocking — the
   `mj_quatdiff` fusion verdict holds at every observed number — but the
   re-run on the quiet machine the next phase uses should be the number
   of record.
2. **Does any real caller pass n ≥ 262,144?** The `PARALLEL_MIN` question
   (§4) is only answerable from caller data, not from the bench. If real
   batches are ~10–100k rows, the serial path is what ships and the
   `parallel` verdicts for `bounded_quat_dist` etc. are moot.
3. **`euler_to_rmat`'s `_euler_planes` NOTE.** The wrapper comment says
   the copy-back "cost is one extra pass over the output, which is why it
   is paid here and not on every call inside the kernel." Measured: it is
   3.9–5.7× the kernel. That note is the thing the next phase must delete
   or justify with a number; this file is that number.
