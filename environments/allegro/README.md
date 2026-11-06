# Allegro cube interaction

Sixteen finger hinges follow Newton's sinusoidal controller while the fixed root rocks. A free 7.2 cm cube rests in the hand through contact and friction. Eight 2.5 ms integrations form one control step. Both engines use the same cuboid model: 18 hand boxes, one cube and one floor box.

```bash
python -m environments.allegro.benchmark --num-envs 128 --output allegro.json
```

Use 128, 512, 2048, 4096 or 8192 worlds. The command runs both engines for 20 simulated seconds and measures physics at controls 16, 64 and 111. Each timing restores shared inputs and replays an eight-integration CUDA graph with prepared controls and root pose fixed. CUDA events exclude restoration and controller work. Contacts refresh every integration. After warm-up, five rounds of 48 replays per state are averaged over the three states; the result is the median round mean.

The baseline uses Newton collision detection and MuJoCo Warp dynamics. The CUDA step keeps 14 hand-picked cube pairs, 12 contact slots and 64 constraint rows. Its only narrow phase tests the six cuboid face axes, clips an incident face and keeps up to four points. It omits edge-cross axes, so contacts are approximate. There is no mesh traversal, hull preprocessing or GJK/EPA. Only hinge/free joints, joint limits and four-sided friction constraints remain. Uniform contact parameters are compiled in; the code rejects other geometry and material configurations.

Each hand box fits the authored collider's local bounds. Masses, inertias, centers of mass, joints, actuator gains and initial joint coordinates are preserved. The root reads the controller transform directly; validation checks that it follows the commanded pose. There are no hand self-contacts or attachments.

Validation requires finite sampled state, zero overflow flags, nearby hand bodies, root-pose error below 1e-5 and at least half the cubes held after 20 seconds. Final held fraction must be within 10 percentage points of the baseline and mean cube height within 5 cm. Common initial conditions are repeated across worlds; these checks do not establish randomized robustness or orientation-tracking equivalence.

| File | Role |
|---|---|
| `model.usda` | Cuboid geometry and preserved body/joint properties |
| `environment.py` | Original controller, model loading and solver adapters |
| `compact.py` | Fixed dimensions, constants, topology maps and CUDA build |
| `task_specs.py` | The 14 cube contact pairs |
| `task_step.cuh` | Full dynamics, cuboid contacts, constraints and integration |
| `direct_io.cuh` | Newton state and contact-buffer conversion |
| `benchmark.py` | Paired timings and behavioral validation |

```cpp
// One warp per world, one kernel per integration.
prepare(dd, s, world, lane, scratch_matrix, scratch_rhs);
solve_integrate(dd, s, solve, world, lane, io, true);
```

The model derives from Newton's `wonik_allegro` asset at revision `a0547548eaa966c2f5478bee496c3cfba1fa98fc`; its license is in `LICENSE.model`. The example adapter retains its Apache-2.0 attribution. USD loading needs `usd-core==26.8` and `newton-usd-schemas==0.6.0`; the remaining runtime pins match the other environments.
