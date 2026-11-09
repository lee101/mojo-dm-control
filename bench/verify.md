# VERIFY — re-measurement of the README benchmark table

Method: `pixi run build && pixi run test` (exit 0; 551 passed), then
`pixi run bench` (flocked `/tmp/mojo-bench.lock`, best of 5, identical
inputs, agreement checked in-source before timing — no MISMATCH flags).
Four full `pixi run bench` runs were executed in this session. The 20%
rule was applied to the final full run (`/tmp/bench_final.txt` from this
session), which is the run the README corrections below come from.
Prior full runs in the session bracketed it; their ranges are given
where they matter.

Host: Intel Xeon E5-2697 v4 @ 2.30 GHz, 72 threads, Linux
6.8.0-147-generic, Python 3.13.15, NumPy 2.5.3 — same machine as the
previous verify pass, seven days later. Box load was moderate
(9–17 across the session, vs 13–17 last time).

## 1. Prescribed run: green

- `pixi run build` — exit 0 (only deprecation warnings for positional
  `__getitem__` in `src/ported.mojo`).
- `pixi run test` — 551 passed, 4 warnings. Matches README's 551.
- `pixi run bench` — exit 0. No `MISMATCH` flags: every case agreed
  within `1e-7 * scale` on identical inputs before timing.

## 2. Row-by-row: final full-run numbers vs README

All three columns (mojo, dm_control, ratio) measured by the final
`pixi run bench`. Drift = |measured − README| / README.

| case | README (prev verify) | measured (final run) | drift | verdict |
| --- | --- | --- | --- | --- |
| quat_conj | 8.82 / 40.82 / 4.63x | 8.08 / 38.09 / 4.71x | ≤8.4% | reproduces |
| quat_inv | 8.21 / 90.04 / 10.97x | 8.89 / 96.54 / 10.86x | ≤8.3% | reproduces |
| quat_mul | 11.61 / 274.46 / 23.64x | 11.44 / 275.52 / 24.09x | ≤1.9% | reproduces |
| quat_diff | 18.84 / 204.90 / 10.87x | 17.26 / 213.87 / 12.39x | ≤14.0% | reproduces |
| quat_log | 18.40 / 191.01 / 10.38x | 18.26 / 191.13 / 10.47x | ≤0.9% | reproduces |
| quat_dist | 94.13 / 556.47 / 5.91x | 89.80 / 537.68 / 5.99x | ≤4.6% | reproduces |
| quat_rotate | 604.76 / 3133.43 / 5.18x | 685.07 / 3112.47 / 4.54x | mojo 13.3%, ratio 12.4% | reproduces |
| axisangle_to_quat | 34.81 / 124.85 / 3.59x | 33.85 / 124.80 / 3.69x | ≤2.8% | reproduces |
| rotation_x_axis | 12.77 / 110.10 / 8.62x | 11.87 / 109.84 / 9.25x | ≤7.3% | reproduces |
| euler_to_rmat ZYX | 60.31 / 1345.09 / 22.30x | 59.96 / 1344.27 / 22.42x | ≤0.6% | reproduces |
| euler_to_rmat XYZ full | 92.78 / 2049.98 / 22.09x | 94.15 / 2040.66 / 21.68x | ≤1.9% | reproduces |
| rotation_matrix_2d | 9.93 / 36.41 / 3.67x | 8.74 / 38.87 / 4.45x | ratio 21.3% | **corrected** |
| tolerance gaussian | 6.41 / 22.47 / 3.51x | 7.63 / 27.53 / 3.61x | mojo 19.0%, dm 22.5% | **corrected** |
| tolerance linear | 1.67 / 20.51 / 12.31x | 1.55 / 19.82 / 12.80x | ≤7.2% | reproduces |
| tolerance margin=0 | 0.90 / 3.24 / 3.61x | 0.83 / 3.52 / 4.23x | ≤17.2% | reproduces |
| mj_quatprod | 10.78 / 191.81 / 17.79x | 10.50 / 192.12 / 18.30x | ≤2.9% | reproduces numerically (see §4) |
| mj_quatdiff | 10.30 / 17.31 / 1.68x | 9.11 / 16.35 / 1.80x | ≤11.6% | reproduces numerically (see §4) |
| euler2quat | 165.37 / 186.35 / 1.13x | 124.18 / 148.43 / 1.20x | mojo 24.9%, dm 20.3% | **corrected** |
| bounded_quat_dist | 20.26 / 130.63 / 6.45x | 19.87 / 131.94 / 6.64x | ≤2.9% | reproduces |
| QuaternionRotate | 253.45 / 868.39 / 3.43x | 292.63 / 835.51 / 2.86x | ≤16.6% | reproduces |
| nullspace_method | 5.62 / 4.87 / 0.87x | 5.60 / 4.61 / 0.82x | ≤5.7% | reproduces (still slower) |

`nullspace_method` stays the one row below 1.0 (0.82x measured) and
stays in the table as-is, un-reworded. Twenty of twenty-one rows above
1.0 in this run, as before.

### The three corrected rows

- **rotation_matrix_2d**: final run 8.74 / 38.87 / 4.45x; ratio 21.3%
  above the README's 3.67x, so corrected to final-run numbers. The
  ratio moves because both sides drift ~7–12% in opposite directions
  (mojo 9.93→8.74, dm 36.41→38.87); the row is a small timed kernel
  (best-of-5 on ~10–40ms) and plainly runnable to a few ms of noise.
- **tolerance gaussian**: final run 7.63 / 27.53 / 3.61x; dm 22.5%
  above README (ratio barely moved: 3.51x→3.61x). Corrected to
  final-run numbers. Tolerance is a short kernel (1–25ms across rows)
  and upstream was the wobblier side across the session's runs.
- **euler2quat**: final run 124.18 / 148.43 / 1.20x. Both sides >20%
  below the previous verify pass's 165.37 / 186.35. This row was the
  noisiest in the previous session too (120–166ms across its runs);
  it is 20k Python-loop calls, so it tracks box load. Corrected to
  the final-run numbers anyway, because the rule is to publish what
  the prescribed run measured.

Session ranges for the threshold-flirting rows: the first full run
put `quat_conj` at 6.39 / 30.27 and `quat_mul` at 9.10 / 172.78
(>20% off in both columns), but the two intervening full runs put
them at 8.54/7.34 and 11.43/12.85, and the final run lands within
8.4% of README on both — so those first-run values read as
contention/noise, not a code change (no source was edited between
runs). `euler_to_rmat ZYX` reproduced at 60.31→59.96ms (0.6%),
confirming the previous pass's note that its 87–90ms early values
were contention.

## 3. Same-input and no-skipped-work audit (`bench/bench.py`)

Re-read `bench/bench.py` in full. Every case builds one input pair,
calls both sides on it before timing (agreement check), then times
`ours()` / `theirs()` best-of-5. Inputs are fixed-seed, shared
between sides, and nothing mutates them across timed repetitions:

- `quat_conj/inv/mul/diff/log/dist`, `axisangle_to_quat`,
  `rotation_x_axis`, `euler_to_rmat`, `rotation_matrix_2d`,
  `tolerance` (all three): identical arrays on both sides, no
  in-place mutation in either implementation. `quat_diff` on the
  port side is `quat_mul(quat_conj(a), b)` — two kernels plus a
  32MiB temporary — matching upstream's own composition; nothing
  skipped.
- `quat_rotate` (100k calls) and `euler2quat` (20k calls): both
  sides loop in Python over the same quaternions/angles; the loop
  overhead is paid by both.
- `bounded_quat_dist`: the timed lambda copies both inputs
  (`a.copy(), b.copy()`) per call on *both* sides (~9ms of the ~20ms
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
(§4): the `dm_control` column is not upstream's function named in
the first column.

Numerical spot-check (this session, 100k samples): `tolerance`
gaussian max abs error vs upstream 2.9e-11 — matching README's
documented `3e-11` divergence note; `tolerance` linear and
`rotation_matrix_2d` bit-identical (0.0).

## 4. The two `mujoco.math` rows: reference is not upstream's function

Unchanged from the previous session's finding (re-confirmed in
`bench/bench.py`, lines 163–174): upstream
`mujoco.math.mj_quatprod` / `mujoco.math.mj_quatdiff` accept only a
single `(4,)` quaternion, so there is no batched upstream
implementation to compare 1M quats against. `bench.py` substitutes:

- `mj_quatprod (1M quats)`: `theirs = up_tr.quat_mul` (numpy batched
  Hamilton product). The arithmetic is identical to `mju_mulQuat`;
  `tests/test_mujoco_math.py` asserts parity against MuJoCo's C
  helper directly. The 18.30x is port-batch vs upstream-batch on the
  same maths. Correct as a *batched* comparison, mislabelled as a
  `mujoco.math` one.
- `mj_quatdiff (1M quats)`: `theirs = tr.quat_mul(tr.quat_conj(a), b)`
  — the **port's own** unfused pair. This row compares the port
  against itself (fused 1 kernel vs unfused 2 kernels + temporary),
  not against upstream at all. Numerically it reproduces
  (9.11 / 16.35 / 1.80x vs 10.30 / 17.31 / 1.68x), but the 1.80x is
  not an upstream speedup.

README already discloses the substitution (paragraph after the
table); the previous pass documented the scalar-call cost
(upstream scalar `mju_*` ~2–4µs/call vs port scalar wrappers
~12.5µs, dominated by `broadcast_arrays` + `addr()`), and that
measurement is not re-run here — it concerns scalar plumbing, and
the README's disclosure is what this pass relies on. Per the
session rules (code changes only for bugs shown by a failing test)
`bench/bench.py` and the wrappers were not changed.

## 5. Test parity audit (`git log -p -- tests/`)

`git log --all --oneline` shows five commits:
`f9f4c2e` "mojo-dm-control: Mojo port of dm-control" (creates
`tests/` in full), `8aa4ac6` "phase analyze: agent pass" (adds only
`bench/analysis.md`), `18a0f71` "phase verify: agent pass" (touches
only `README.md` and `bench/verify.md`), and two toolchain commits
that predate the port. `git diff f9f4c2e..HEAD -- tests/` is empty.
**No assertions were removed and no tolerances loosened by later
phases** — there is no earlier-phase diff to restore from.

Tolerances present now (identical to the initial commit):

- Exponential sigmoids (`gaussian`, `hyperbolic`, `tanh_squared`):
  `rtol=1e-8, atol=1e-10` in `tests/test_rewards.py` /
  `tests/test_simd_paths.py`. Measured max abs error vs upstream
  this session: 2.9e-11 (gaussian) — the ~300× headroom is the
  `std.math` polynomial `exp`/`cosh`/`tanh` divergence README
  documents. Left as-is: tightening without a failing test would
  violate the session rules.
- Algebraic sigmoids (`long_tail`, `reciprocal`, `cosine`, `linear`,
  bit-identical): `rtol=1e-12` — measured error 0.0. Parity.
- Everything else compares at `rtol=1e-10`–`1e-12` against upstream,
  including `mju_*` C helpers, `qpos_from_site_pose` end-to-end
  (steps, success, err_norm, qpos) and `lqr_solver.solve`.
  `quat_log`/`quat_dist` use `rtol=1e-9`.

No tests were changed in this session.

## 6. What changed in this session

- `README.md` table: three rows corrected to final-run numbers —
  `rotation_matrix_2d` 8.74/38.87/4.45x, `tolerance gaussian`
  7.63/27.53/3.61x, `euler2quat` 124.18/148.43/1.20x. In-text
  ratios updated to match: `nullspace_method` 0.96x → 0.82x and
  `euler2quat` 1.13x → 1.20x. No other rows touched (all within
  20% of the final run). Pre-fusion historical figures in
  "Design notes" (148.31ms → 60.31ms etc.) were not re-measured and
  are left as history.
- This file (`bench/verify.md`).
- No code, kernel, wrapper, bench, or test changes. The
  `_euler_planes` redundant copy noted in the previous pass was
  again left alone per the bug-fix-only rule (no failing test).

## 7. What could not be reproduced

- The first full run's `quat_conj` (6.39 / 30.27) and `quat_mul`
  (9.10 / 172.78) values — >20% off on both columns from README —
  did not reproduce in the two intervening or the final run, which
  bracket README to ≤8.4%. With no source edits between runs, this
  reads as box contention; excluded from the corrections.
- Historical pre-fusion figures in "Design notes", the scalar-call
  `mju_*` micro-measurement from the previous pass, and third-party
  numbers in `bench/analysis.md` were not re-measured in this
  session and are not claimed as reproduced.
- The previous pass's `quat_conj` cold-vs-warm observation (72.8ms
  cold vs ~39ms warm) is consistent with this session's spread
  (first-run warm 30.27ms, final 38.09ms) but was not re-probed.

## 8. Finish

`pixi run build && pixi run test` — both green (551 passed;
rerun after the README/verify.md edits, which cannot affect them).
