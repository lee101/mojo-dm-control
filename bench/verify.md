# VERIFY — re-measurement of the README benchmark table

Method: `pixi run build && pixi run test` (exit 0; 551 passed), then
`pixi run bench` (flocked, best of 5, identical inputs, agreement checked
in-source). Six full `pixi run bench` runs were executed in this session
plus targeted isolated re-measurements under the same flock. The 20%
rule was applied to the final full run (`/tmp/bench_final.txt` from this
session), which is the run the README corrections below come from.
Prior full runs in the session bracketed it; their ranges are given
where they matter.

Host: Linux-6.8.0-147-generic-x86_64-with-glibc2.39, 72 CPUs (this
session's kernel; README had quoted 6.8.0-142). Python 3.13.15. Upstream
`dm_control` / `mujoco` / `numpy` versions are those installed by
`pixi.lock` in this checkout; `dm_control` exposes no `__version__`
attribute, so no pin is quoted. The box is shared (load average
13–17 during the session; other benches running concurrently), which is
why early full runs showed `euler_to_rmat ZYX` at 87–90ms where later
flocked runs settle back to ~60ms. README's own caveat that
`dm_control` moves 30–50% between runs is borne out here.

## 1. Prescribed run: green

- `pixi run build` — exit 0 (only deprecation warnings for positional
  `__getitem__` in `src/ported.mojo`).
- `pixi run test` — 551 passed (matches README's claimed 551), 4
  pre-existing warnings.
- `pixi run bench` — exit 0. No `MISMATCH` flags: every case agreed
  within `1e-7 * scale` on identical inputs before timing.

## 2. Row-by-row: final full-run numbers vs README

All three columns (mojo, dm_control, ratio) measured by the final
`pixi run bench`. Drift = |measured − README| / README.

| case | README | measured (final run) | drift | verdict |
| --- | --- | --- | --- | --- |
| quat_conj | 8.35 / 55.86 / 6.69x | 8.82 / 40.82 / 4.63x | dm 26.9%, ratio 30.8% | **corrected** |
| quat_inv | 8.21 / 90.04 / 10.97x | 8.35 / 87.27 / 10.46x | ≤4.6% | reproduces |
| quat_mul | 11.61 / 274.46 / 23.64x | 11.45 / 280.57 / 24.50x | ≤3.6% | reproduces |
| quat_diff | 18.93 / 270.94 / 14.31x | 18.84 / 204.90 / 10.87x | dm 24.4%, ratio 24.0% | **corrected** |
| quat_log | 18.40 / 191.01 / 10.38x | 19.30 / 190.95 / 9.89x | ≤4.9% | reproduces |
| quat_dist | 94.13 / 556.47 / 5.91x | 89.78 / 541.10 / 6.03x | ≤4.6% | reproduces |
| quat_rotate | 604.76 / 3133.43 / 5.18x | 611.97 / 3047.17 / 4.98x | ≤3.9% | reproduces |
| axisangle_to_quat | 34.81 / 124.85 / 3.59x | 33.44 / 121.05 / 3.62x | ≤3.9% | reproduces |
| rotation_x_axis | 12.77 / 110.10 / 8.62x | 13.54 / 113.32 / 8.37x | ≤6.0% | reproduces |
| euler_to_rmat ZYX | 60.31 / 1345.09 / 22.30x | 60.22 / 1384.96 / 23.00x | ≤3.1% | reproduces |
| euler_to_rmat XYZ full | 92.78 / 2049.98 / 22.09x | 88.68 / 2047.89 / 23.09x | ≤4.5% | reproduces |
| rotation_matrix_2d | 9.93 / 36.41 / 3.67x | 9.16 / 35.26 / 3.85x | ≤7.8% | reproduces |
| tolerance gaussian | 6.41 / 22.47 / 3.51x | 6.51 / 23.49 / 3.61x | ≤4.5% | reproduces |
| tolerance linear | 1.67 / 20.51 / 12.31x | 1.57 / 18.85 / 11.97x | ≤8.1% | reproduces |
| tolerance margin=0 | 0.90 / 3.24 / 3.61x | 0.83 / 3.40 / 4.10x | ≤13.6% | reproduces |
| mj_quatprod | 10.78 / 191.81 / 17.79x | 9.86 / 182.68 / 18.52x | ≤8.5% | reproduces numerically (see §4) |
| mj_quatdiff | 10.30 / 17.31 / 1.68x | 10.37 / 16.73 / 1.61x | ≤6.7% | reproduces numerically (see §4) |
| euler2quat | 124.06 / 151.38 / 1.22x | 165.37 / 186.35 / 1.13x | mojo 33.3%, dm 23.1% | **corrected** |
| bounded_quat_dist | 20.26 / 130.63 / 6.45x | 20.64 / 141.58 / 6.86x | ≤8.4% | reproduces |
| QuaternionRotate | 253.45 / 868.39 / 3.43x | 264.49 / 879.29 / 3.32x | ≤4.4% | reproduces |
| nullspace_method | 5.62 / 4.87 / 0.87x | 5.16 / 4.96 / 0.96x | ≤10.3% | reproduces (still slower) |

`nullspace_method` stays the one row below 1.0 (0.96x measured) and
stays in the table as-is, un-reworded. Twenty of twenty-one rows above
1.0 in this run, as before.

### The three corrected rows (all corrections make the README worse-looking, not better)

- **quat_conj**: README's dm_control 55.86ms did not reproduce. All five
  other full runs this session put upstream at 40.60–43.96ms (one
  30.42ms outlier); a controlled re-measurement shows why upstream is
  so variable — 72.8ms cold (no agreement warmup) vs 39.3ms after the
  agreement call that `bench.py` performs, i.e. the README figure sits
  between cold and warm. Port-side 8.35ms reproduces (session range
  7.76–8.85ms). Corrected to final-run 8.82 / 40.82 / 4.63x.
- **quat_diff**: README's 270.94ms upstream did not reproduce. Session
  full runs: 233.26, 224.46, 204.90; flocked isolated: 221.36. The
  final-run 204.90ms is >20% below README. Corrected to final-run
  18.84 / 204.90 / 10.87x.
- **euler2quat**: final-run 165.37 / 186.35 is >20% above README on both
  sides. This was the noisiest row of the session — earlier full runs
  gave 124.98/144.93 and 120.06/152.16, flocked isolated gave
  132.57/146.73 — so 165/186 is plausibly a contention spike (box load
  ~17 at that moment) rather than a code change (no source was edited).
  It is corrected to the final-run numbers anyway, because the rule is
  to publish what the prescribed run measured, and the correction is
  conservative (ratio 1.22x → 1.13x).

Session ranges worth noting: `euler_to_rmat ZYX` was 87.46/90.24ms in
the first two full runs, then 59.52/59.51/62.59/60.22ms — the early
values coincide with other benchmarks running on the box and did not
reproduce under the flock in isolation (63.65ms). README's 60.31ms is
within 0.1% of the final run's 60.22ms. No correction needed.

## 3. Same-input and no-skipped-work audit (`bench/bench.py`)

Every case builds one input pair, calls both sides on it before timing
(agreement check), then times `ours()` / `theirs()` best-of-5. Inputs
are fixed-seed, shared between sides, and nothing mutates them across
timed repetitions:

- `quat_conj/inv/mul/diff/log/dist`, `axisangle_to_quat`,
  `rotation_x_axis`, `euler_to_rmat`, `rotation_matrix_2d`,
  `tolerance` (all three): identical arrays, no in-place mutation in
  either implementation. `quat_diff` on the port side is
  `quat_mul(quat_conj(a), b)` — two kernels plus a 32MiB temporary —
  matching upstream's own composition; nothing skipped.
- `quat_rotate` (100k calls) and `euler2quat` (20k calls): both sides
  loop in Python over the same quaternions/angles; the loop overhead
  is paid by both.
- `bounded_quat_dist`: the timed lambda copies both inputs
  (`a.copy(), b.copy()`) per call on *both* sides (~9ms of the ~21ms
  row). Both implementations normalise in place, so the copies are
  required work, paid equally. Fair.
- `QuaternionRotate` (20k calls): the variation holds a fixed
  quaternion and vector, so all 20k calls compute the same rotation —
  degenerate as a workload, but both ports execute exactly the same
  per-call work (upstream and port `__call__` are line-identical apart
  from which `transformations` module they call). `random_state`
  is passed but never consumed on either side (both values are
  constant arrays). No skipped work; the ratio measures the scalar
  `quat_rotate` FFI path.
- `nullspace_method` (200 × 6×12, reg 3e-2): both sides take the
  `np.linalg.solve` branch on identical Jacobians. Fair.

No case skips work on one side. Two rows have a different problem
(§4): the `dm_control` column is not upstream's function named in the
first column.

## 4. The two `mujoco.math` rows: reference is not upstream's function

Upstream `mujoco.math.mj_quatprod` / `mujoco.math.mj_quatdiff` call
`mju_mulQuat` / `mju_negQuat` on a single `(4,)` quaternion and reject
a batch (`mju_mulQuat(): incompatible function arguments` on a
`(n,4)` input — reproduced in this session). There is no batched
upstream implementation to compare 1M quats against, so `bench.py`
substitutes batched equivalents — silently, in the case names:

- `mj_quatprod (1M quats)`: `theirs = up_tr.quat_mul` (numpy batched
  Hamilton product). The arithmetic is identical to `mju_mulQuat`;
  `tests/test_mujoco_math.py` asserts parity against MuJoCo's C helper
  directly. The 18.52x is port-batch vs upstream-batch on the same
  maths. Correct as a *batched* comparison, mislabelled as a
  `mujoco.math` one.
- `mj_quatdiff (1M quats)`: `theirs = tr.quat_mul(tr.quat_conj(a), b)`
  — the **port's own** unfused pair. This row compares the port against
  itself (fused 1 kernel vs unfused 2 kernels + temporary), not against
  upstream at all. Numerically it reproduces (10.37 / 16.73 / 1.61x
  vs 10.30 / 17.31 / 1.68x), but the 1.61x is not an upstream speedup.

Measured in this session (3000-call loops, best of 5, extrapolated —
not bench numbers, not in the README): upstream scalar
`mj_quatprod` 2.17µs/call → ~2173ms per 1M; upstream scalar
`mj_quatdiff` 3.87µs/call → ~3874ms per 1M. Port scalar calls are
12.74 / 12.52µs each (~12.7s per 1M by loop): the port's scalar
`mj_quatprod`/`mj_quatdiff` wrappers pay `np.broadcast_arrays` +
`rows()` reshaping per call (~12µs, profiled: `broadcast_arrays` and
`addr()` dominate) and are 6× *slower* per scalar call than upstream's
direct ctypes-to-`mju_*` call. The batched rows therefore show the
port winning on a workload (huge batch) that upstream's `mujoco.math`
cannot express, while single-quaternion callers pay more per call for
the batched plumbing. `euler2quat` (1.13x, the only `mujoco.math`
row measured against upstream's real function) is the honest
scalar-call figure. README now discloses the substitution instead of
letting the row names imply otherwise. This is a benchmark-labelling
issue; per the session rules (code changes only for bugs shown by a
failing test) `bench/bench.py` and the wrappers were not changed.

## 5. Test parity audit (`git log -p -- tests/`)

`git log --all --oneline` shows three commits touching this tree:
`f9f4c2e` "mojo-dm-control: Mojo port of dm-control" (creates `tests/`
in full), `8aa4ac6` "phase analyze: agent pass, gate green" (adds only
`bench/analysis.md`), and two toolchain commits that predate the port.
`git diff f9f4c2e..8aa4ac6 -- tests/` is empty. **No assertions were
removed and no tolerances loosened by later phases** — there is no
earlier-phase diff to restore from.

Tolerances present now, with measured errors (this session):

- Exponential sigmoids (`gaussian`, `hyperbolic`, `tanh_squared`):
  `rtol=1e-8, atol=1e-10` in `tests/test_rewards.py` /
  `tests/test_simd_paths.py`. Measured max abs error vs upstream:
  2.9e-11 (gaussian), 2.8e-13 (hyperbolic), 5.0e-16 (tanh_squared) —
  consistent with README's `3e-11` note on `std.math` polynomial
  `exp`/`cosh`/`tanh`. The allowed tolerance is ~300× looser than the
  observed error; it was already this value in the initial commit.
  Left as-is: tightening without a failing test would violate the
  session rules, and the README's documented-divergence section
  explains the cause.
- Algebraic sigmoids (`long_tail`, `reciprocal`, `cosine`, `linear`,
  bit-identical): `rtol=1e-12` — measured error 0 to 1.1e-16. Parity.
- `quat_log` / `quat_dist`: `rtol=1e-9` — measured max abs error 0.0
  on this session's 2000-quat sample. Parity.
- Everything else compares at `rtol=1e-10`–`1e-12` against upstream,
  including `mju_*` C helpers, `qpos_from_site_pose` end-to-end (steps,
  success, err_norm, qpos) and `lqr_solver.solve`.

No tests were changed in this session.

## 6. What changed in this session

- `README.md` table: three rows corrected to final-run numbers —
  `quat_conj` 8.82/40.82/4.63x, `quat_diff` 18.84/204.90/10.87x,
  `euler2quat` 165.37/186.35/1.13x. In-text `nullspace_method`
  ratio 0.87x → 0.96x and `euler2quat` 1.22x → 1.13x updated to match
  the corrected table. Kernel date corrected (6.8.0-142 → 6.8.0-147)
  and statement added that the two `mujoco.math` rows' reference
  column is not upstream's scalar function. No other rows touched
  (all within 20% of the final run). Pre-fusion historical figures in
  "Design notes" (148.31ms → 60.31ms etc.) were not re-measured and
  are left as history, flagged here.
- This file (`bench/verify.md`).
- No code, kernel, wrapper, bench, or test changes. In particular the
  redundant `_euler_planes` copy in `transformations.euler_to_rmat`
  (a full extra pass over the packed output that is already
  `(n,s,s)`-shaped — kernel-only ~13ms vs API ~60ms) is a known
  inefficiency, noted here and in `bench/analysis.md` §2.2, and left
  alone per the bug-fix-only rule.

## 7. What could not be reproduced

- Two transient values from the first two full runs
  (`euler_to_rmat ZYX` at 87.46 / 90.24ms) did not reproduce in later
  flocked runs or in isolation (~60–64ms). Attributed to contention
  on the shared box (concurrent benches; load 13–17) rather than to
  code — no source changed between those runs — and excluded from the
  corrections.
- README's `quat_conj` upstream figure (55.86ms) could not be
  reproduced as a single stable number: cold 72.8ms, warm (post-agreement)
  ~39–44ms across five runs. The final-run warm value (40.82ms) is
  what the table now carries.
- Historical pre-fusion figures in "Design notes" and third-party
  numbers in `bench/analysis.md` were not re-measured in this session
  and are not claimed as reproduced.

## 8. Finish

`pixi run build && pixi run test` — both green (551 passed;
rerun after the README/verify.md edits, which cannot affect them).
