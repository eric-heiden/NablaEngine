# Allegro cube interaction

The Newton Allegro example's 16 finger hinges follow sinusoidal targets while the fixed hand root rocks. A free 7.2 cm cube interacts with the hand through contact and friction. The controller runs at every 2.5 ms integration; eight integrations form a 20 ms control step. There are no hand self-contacts or attachments.

```bash
python -m environments.allegro.benchmark --num-envs 128 --output allegro.json
```

Use 128, 512, 2048, 4096 or 8192 worlds. The command runs both engines for 20 simulated seconds and measures physics at controls 16, 64 and 111. Each timing replays a captured eight-integration graph from restored shared inputs, with prepared controls and root pose held fixed. CUDA events exclude restoration and controller work. Both engines refresh contacts every integration. Five rounds of 48 replays follow warm-up; each round averages the three sampled states, then the median of those five round means is reported.

The baseline is Newton collision detection plus MuJoCo Warp dynamics, as in the original example. CUDA reads the live root transform directly and performs kinematics, mass/bias, collision, constraint solution and integration in one kernel per integration. It keeps the 19 cube–hand/ground candidate pairs and authored masses/inertias. The fixed hull–cube routine tests face axes and constructs up to four contacts from support vertices. It omits edge-cross-product separating axes and uses a fallback contact when the vertex manifold is empty; this is an approximation, not an exact convex collision solver. Rounded face normals and cached face bounds avoid repeated hull scans. Capsule and box proxies failed the retention test, so this task retains the hand's convex hulls.

Validation checks finite sampled state, zero overflow flags, nearby hand bodies and the original requirement that at least half the cubes remain held. The paired comparison additionally requires final held fraction within 10 percentage points and mean cube height within 5 cm. These checks do not establish dexterous manipulation or orientation-tracking equivalence. All worlds use the original common initial state and controller; this is not a randomized robustness test.

`environment.py` contains the controller and both solver adapters. `compact.py` compiles the fixed topology, hull data and actuator maps. `task_step.cuh` contains the full handwritten CUDA step; `direct_io.cuh` reads and writes Newton buffers. `model.usda` contains only simulation geometry and physics, derived from Newton's `wonik_allegro` asset at revision `a0547548eaa966c2f5478bee496c3cfba1fa98fc`. Model values were checked against the source example: inertia, joints, actuators, collision meshes and initial joint coordinates match. The original model license is in `LICENSE.model`; the example adapter retains its Apache-2.0 attribution.

Extra dependencies for USD loading and hull preprocessing: `usd-core==26.8`, `newton-usd-schemas==0.6.0`, and SciPy. The study uses the same pinned Newton/MuJoCo Warp runtime as the other environments.
