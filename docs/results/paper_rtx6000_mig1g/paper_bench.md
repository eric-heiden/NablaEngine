# Paper timing kit: NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 595.58.03)

- started 2026-11-06T22:20:01Z, finished 2026-11-06T22:24:36Z, 300.4 s in all; setup: ok
- NablaEngine 0.1.0 commit 7dd445d (dirty: False); nvcc 12.9.86; arch sm_120
- CPU AMD EPYC 9B45: 192 allocated of 192 logical (affinity 0-191)
- warp-lang wheel of the native venvs: default (1.18.0)
- packages of the main environment: {"jax": "0.12.0.1", "jaxlib": "0.12.0.1", "mujoco": "3.17.0", "mujoco-mjx": "3.17.0", "warp-lang": "1.18.0+cu12", "mujoco-warp": "3.17.0", "numpy": "2.5.4"}
- corpora: mf_s0, mf_s1; the first is the corpus of the sweeps, Jacobians and splits (states tiled where nworld > 4096)
- MuJoCo C threads: 192 (len(os.sched_getaffinity(0))); scheduler CPU counts null

What is timed: ours `fwd` = the forward without tape, `fwd_tape` = the forward that writes the tape, `bwd` = the adjoint on the tape alone, `vjp` = `fwd_tape` + `bwd` in one jit (the gradient from the input state). MuJoCo Warp + PR #1535: its adjoint graph contains its own taped forward recompute, so the adjoint graph alone is the gradient from the input state (compare with ours `vjp`), and forward + adjoint is the pattern of a differentiated rollout (the forward, then the adjoint). Jacobian rows: the 71 x 99 Jacobian per world; ours = one taped forward + 71 adjoints (one per output row); MJX = 99 forward-mode directions; MuJoCo C = chained per-substep finite differences (the 4096 figure of the per-world-rate rows is a linear extrapolation, the measured rows are labelled).

ms per control step, median [p10-p90] over the kept rounds; 4 x 5 ms substeps, 10 Newton iterations.

## Forward, 4096 worlds (mjlab G1)

| engine | op | mf_s0 | mf_s1 | note |
|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd (no tape) | 4.029 [4.028-4.030] | 4.128 [4.128-4.130] |  |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape (with tape) | 4.205 [4.204-4.206] | 4.304 [4.301-4.310] |  |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | 15.449 [15.448-15.450] | 15.557 [15.555-15.558] | naconmax 94687 (nconmax 24) / njmax 128; the documented sizing rule (native headline) |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | 15.728 [15.725-15.734] | 15.829 [15.826-15.831] | naconmax 262144 (nconmax 64) / njmax 192; our engine's capacity |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (default capacity) | NEFC overflow in 317 worlds | NEFC overflow in 378 worlds | naconmax 196608 (nconmax 48) / njmax 64; put_data's defaults; capacity overflow: not timed |

## Gradient, 4096 worlds

| engine | op | mf_s0 | mf_s1 | note |
|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | bwd (exact) | 1.846 [1.845-1.846] | 1.859 [1.856-1.977] | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | vjp = fwd_tape + bwd, one jit (exact) | 6.100 [6.098-6.103] | 7.129 [7.127-7.134] | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | bwd (interface) | 1.857 [1.857-1.883] | 2.090 [2.089-2.091] | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | vjp = fwd_tape + bwd, one jit (interface) | 6.088 [6.086-6.095] | 7.132 [7.128-7.137] | interface defaults: clamp_kv=0, normal=3 |
| MuJoCo Warp 3.17.0 (PyPI, native) | backward | - | - | none in any release (no timing) |

## Jacobian 71 x 99 (per world: us/world)

| engine | op | corpus | B | ms per call | us per world | note |
|---|---|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (exact) | mf_s0 | 4096 | 137.063 [136.950-137.081] | 33.46254 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (exact) | mf_s0 | 4096 | 132.803 [132.764-132.869] | 32.42258 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (interface) | mf_s0 | 4096 | 137.172 [137.141-137.200] | 33.48919 | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (interface) | mf_s0 | 4096 | 132.903 [132.896-132.937] | 32.44711 | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (exact) | mf_s1 | 4096 | 154.884 [154.723-154.950] | 37.81343 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (exact) | mf_s1 | 4096 | 149.838 [149.777-149.893] | 36.58155 | exact: clamp_kv=1, normal=0 |
| ours (NablaEngine, jax_ffi under jit) | fwd_tape + 71 bwd (interface) | mf_s1 | 4096 | 155.004 [154.989-155.052] | 37.84268 | interface defaults: clamp_kv=0, normal=3 |
| ours (NablaEngine, jax_ffi under jit) | 71 bwd on one tape (interface) | mf_s1 | 4096 | 150.018 [149.851-150.056] | 36.6255 | interface defaults: clamp_kv=0, normal=3 |

## nworld sweep (mf_s0)

| engine | op | 128 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd | 0.646 [0.646-0.646] | 0.893 [0.893-0.899] | 1.340 [1.339-1.340] | 2.407 [2.402-2.411] | 4.700 [4.699-4.705] | 9.193 [9.174-9.208] | 17.911 [17.909-18.072] |
| ours (NablaEngine, jax_ffi under jit) | vjp (exact) | 0.839 [0.839-0.839] | 1.314 [1.313-1.314] | 2.015 [2.011-2.016] | 3.627 [3.622-3.634] | 7.040 [7.027-7.048] | 13.662 [13.654-13.667] | 26.764 [26.746-26.777] |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | 3.314 [3.313-3.314] | 4.218 [4.217-4.218] | 5.703 [5.702-5.704] | 8.794 [8.792-8.794] | 15.449 [15.448-15.450] | 29.845 [29.844-29.853] | 58.922 [58.917-58.927] |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | 3.311 [3.311-3.312] | 4.287 [4.286-4.287] | 5.761 [5.760-5.762] | 8.937 [8.935-8.939] | 15.728 [15.725-15.734] | 30.446 [30.440-30.456] | 60.216 [60.209-60.223] |

## Headline cells over P processes (ms per control step, 4096 worlds)

The main part's process and the repeat processes (`rep<r>_<part>`), each its own fresh process; median over the kept rounds per process.

| engine | op | corpus | P | per-process medians | median | spread (max / min - 1) |
|---|---|---|---|---|---|---|
| ours (NablaEngine, jax_ffi under jit) | fwd (no tape) | mf_s0 | 3 | 4.029, 4.056, 3.991 | 4.029 | 1.6% |
| ours (NablaEngine, jax_ffi under jit) | vjp (exact) | mf_s0 | 3 | 6.100, 6.083, 6.012 | 6.083 | 1.5% |
| ours (NablaEngine, jax_ffi under jit) | vjp (interface) | mf_s0 | 3 | 6.088, 6.065, 5.999 | 6.065 | 1.5% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | mf_s0 | 3 | 15.449, 15.456, 15.457 | 15.456 | 0.1% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | mf_s0 | 3 | 15.728, 15.735, 15.676 | 15.728 | 0.4% |
| ours (NablaEngine, jax_ffi under jit) | fwd (no tape) | mf_s1 | 3 | 4.128, 4.076, 4.054 | 4.076 | 1.8% |
| ours (NablaEngine, jax_ffi under jit) | vjp (exact) | mf_s1 | 3 | 7.129, 6.104, 6.059 | 6.104 | 17.7% |
| ours (NablaEngine, jax_ffi under jit) | vjp (interface) | mf_s1 | 3 | 7.132, 6.054, 6.042 | 6.054 | 18.0% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (procedure capacity) | mf_s1 | 3 | 15.557, 15.562, 15.563 | 15.562 | 0.0% |
| MuJoCo Warp 3.17.0 (PyPI, native) | forward graph (matched capacity) | mf_s1 | 3 | 15.829, 15.810, 15.841 | 15.829 | 0.2% |

jax_ffi.init: 0.677 s (built now: False), arch sm_120, flags {"forward": ["-arch=compute_90", "-code=sm_120"], "adjoint": ["-arch=compute_90", "-code=sm_120"]}, forward defines ['-DSPLIT_PRE_MINB=16']

## Parts

| part | status | s | SM MHz median (min) | power W max | hw slowdown | other GPU procs | note |
|---|---|---|---|---|---|---|---|
| ours | ok | 100.9 | 2385.0 (180.0) | 230.33 | False | 0 |  |
| mjw_native | ok | 37.0 | 2355.0 (2355.0) | 187.77 | False | 0 |  |
| mjw_native_nw128 | ok | 7.3 | 2370.0 (2355.0) | 134.52 | False | 0 |  |
| mjw_native_nw512 | ok | 7.3 | 2370.0 (2355.0) | 144.15 | False | 0 |  |
| mjw_native_nw1024 | ok | 7.3 | 2377.0 (2355.0) | 156.96 | False | 0 |  |
| mjw_native_nw2048 | ok | 7.4 | 2377.0 (2355.0) | 174.77 | False | 0 |  |
| mjw_native_nw8192 | ok | 8.0 | 2385.0 (2355.0) | 192.79 | False | 0 |  |
| mjw_native_nw16384 | ok | 9.0 | 2385.0 (2355.0) | 198.44 | False | 0 |  |
| rep2_ours | ok | 31.5 | 2385.0 (2355.0) | 253.17 | False | 0 |  |
| rep2_mjw_native | ok | 13.0 | 2385.0 (2355.0) | 207.72 | False | 0 |  |
| rep3_ours | ok | 31.6 | 2407.0 (2355.0) | 209.8 | False | 0 |  |
| rep3_mjw_native | ok | 13.1 | 2385.0 (2355.0) | 210.37 | False | 0 |  |
