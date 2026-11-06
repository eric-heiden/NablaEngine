# Newton / MuJoCo Warp environments

Five standalone scenes use task-specific fused CUDA physics: Go2, H1, Franka reach, Franka lift and Allegro cube interaction. The first four use MJCF models; Allegro uses a physics-only USD model. All have Newton buffers, a MuJoCo Warp baseline and CUDA sources. Linux, an NVIDIA GPU and a CUDA toolkit with nvcc are required.

## Run

Use a separate environment from the upstream G1 package, which pins different dependencies. From the repository root:

```bash
uv venv .venv-tasks --python 3.12
uv pip install --python .venv-tasks/bin/python -r environments/requirements.txt
export CUDA_ARCH=sm_89  # NVIDIA L4; set this to your GPU's architecture
.venv-tasks/bin/python -m environments.run --task go2 --engine fused --num-envs 128
.venv-tasks/bin/python -m environments.run --task h1 --engine fused --num-envs 128
.venv-tasks/bin/python -m environments.run --task franka_reach --engine fused --num-envs 128
.venv-tasks/bin/python -m environments.run --task franka_lift --engine fused --num-envs 128
```

`NVCC` can point to an explicit compiler path. Kernels compile on first use and are cached in each task's ignored `build/` directory. The runner uses `cuda:0`; environment counts must be multiples of eight. The pinned Newton revision emits a dependency warning for MuJoCo 3.15; these versions match the investigation's runtime.

The headless examples hold nominal joint targets for 100 control steps; task success requires a controller. Supply one through the small Python API:

```python
from environments.environment import Environment

env = Environment('franka_reach', num_envs=128, engine='fused')
targets = env.default_action.copy()  # 7 arm angles and 2 finger openings
for _ in range(100):
    env.step(targets)
state = env.observations()          # CPU snapshots, outside physics timing
```

Go2 accepts 12 joint angles and updates torque-limited PD control every 5 ms. H1 accepts 19 joint targets with implicit position/velocity actuators; each 20 ms control advances eight 2.5 ms integrations. Franka accepts nine absolute joint targets; position/velocity actuators and a finger equality remain active. In lift, fingers hold the object through contact and friction; there is no gripper-object attachment. The table is kinematic. Eight object families cycle across worlds: four boxes, a sphere, and three capsules. The native baseline uses their convex meshes; fused physics uses the matching analytic primitives.

Run `python -m environments.verify --task franka_lift` with the task environment's Python to check reset semantics, one-control agreement and 100 finite control steps in both engines. This is a packaging check, separate from the original behavioral evaluation.

## Allegro cube interaction

The [Allegro package](allegro/README.md) follows Newton's moving-hand example, including the sinusoidal finger controller and animated root. Run its paired 20-second evaluation and full physics timing protocol with:

```bash
python -m environments.allegro.benchmark --num-envs 128 --output allegro.json
```

Repeat for 512, 2048, 4096 and 8192 worlds. Ratios on L4: 4.70×, 2.87×, 1.78×, 1.72×, 1.78×. All 5/5 retention comparisons pass. Both engines use cuboids. The CUDA contact manifold uses six face axes and at most four clipped points; it omits edge-cross axes. Retaining a cube does not establish orientation-tracking equivalence. The public Allegro benchmark reproduces this sweep directly.

## Benchmark

```bash
for task in go2 h1 franka_reach franka_lift; do
  for count in 128 512 2048 4096 8192; do
    for engine in mjwarp fused; do
      .venv-tasks/bin/python -m environments.benchmark \
        --task "$task" --num-envs "$count" --engine "$engine"
    done
  done
done
```

CUDA events measure one captured physics control interval from a restored nominal state. Reset, JIT compilation, controller preparation and observations are excluded. Both engines refresh contacts every integration: four 5 ms integrations for Go2, eight 2.5 ms integrations for H1, and eight 4.167 ms integrations for Franka. The fused solvers execute four launches per control interval, except reach, which uses one. A separate wall-clock number includes the minimal Python control loop and CPU state observations.

The [tiled plot](../docs/assets/task_speedups.svg) and [reference measurements](reference_speedups.json) cover all five tasks and five batch sizes on NVIDIA L4. The original four tasks span 1.48–10.17×. Allegro spans 1.72–4.70×. For the original four tasks, 18/20 behavioral comparisons pass; remaining failures: h1 / 512: reward std relative difference; h1 / 4,096: reward std relative difference. Locomotion rollouts last 1,000 controls and Franka rollouts 600, with unchanged controllers and acceptance thresholds. Every failure is retained. These results do not establish training equivalence. Lift has limited success in both engines; the thin 1×5×5 cm box has any-success frequency 7.5% with CUDA versus 15.4% native at 8,192 worlds.

The original four standalone runners have nominal resets and absolute joint actions. Their timings are a separate experiment; they do not reproduce the randomized evaluation or supply the evaluation policies. The original task harness stays local. Earlier end-to-end results are outside the scope of this physics-only scaling study.

Regenerate the tiled figure with `uv run --no-project --with matplotlib python -m environments.plot_speedups`.

## Implementation map

| File | Role |
| --- | --- |
| `environment.py` | Load MJCF into Newton, replicate worlds, set controls, reset and step either engine |
| `run.py`, `benchmark.py` | Headless rollout and scoped timing |
| `<task>/model.xml` | Authored articulation, inertial parameters and baseline collision model |
| `<task>/compact.py`, `task_specs.py` | Compile topology, ancestry, actuator maps and fixed pairs; pack varying model fields |
| `<task>/task_step.cuh` | Kinematics, mass/bias, task collision, constraints, register Cholesky and integration |
| `<task>/primitive_math.cuh`, `primitive_collision.cuh` | Task math and collision helpers |
| `<task>/direct_io.cuh`, `launch.cuh` | Read/write Newton buffers and fuse the fixed integrations |

One warp handles a world. Body Jacobians store only ancestor DOFs; each matrix row stays in a lane's registers during Cholesky. Direct actuator maps avoid scanning every actuator for each DOF. Shared scratch is reused between stages. Topology and fixed physical values compile into CUDA; varying parameters remain in an audited per-world pack. Full articulated mass and bias remain.

Go2 replaces body and limb boxes and cylinders with capsules, retaining foot spheres and all 27 plane pairs. Removing leg contacts failed the fall-count check; the all-capsule body model passes at every batch size. H1 uses two longitudinal capsules per sole and one torso capsule, five plane pairs and ten contact slots. Its 64 constraint rows cover all possible contacts plus 19 joint limits. All 25 velocity DOFs remain.

Reach retains nine DOFs and the finger equality. Capsule links contact a bounded tabletop plane; finger pads retain their flat faces because spherical pads changed success counts. Lift keeps 15 dynamic DOFs and 19 fixed pairs. Its object families are four boxes, a sphere and three capsules. Finger-pad/object box contacts use a small SAT-and-vertex routine; table contacts use bounded analytic top faces. There is no gripper/object attachment. Robot and object inertia remain the authored values.

The original four tasks remove general convex support mapping, GJK/EPA, polyhedral collision, mesh traversal, unused friction-loss constraints and the source-transform pipeline. Allegro uses cuboids only, 14 fixed pairs and six face axes with clipping; no hull data or mesh path remains. The MJCF meshes remain for the native reference and rendering; fused physics does not traverse them.

```cpp
for (int sub = 0; sub < INTERNAL_STEPS; ++sub) {
    prepare(dd, s, world, lane, v.H, v.p);
    __syncwarp();
    solve_integrate(dd, s, v, world, lane, io,
                    sub == INTERNAL_STEPS - 1);
    __syncwarp();
}
```

```text
H1 sole capsule, body coordinates:
  center = (0.055, ±0.025, -0.055)
  axis = (1, 0, 0), radius = 0.015, half-length = 0.105
```

The upstream G1 benchmark fixture has 31 capsules, two spheres and 502 explicit candidate pairs, including 14 sole-floor pairs. Its sparse ancestry, fixed pair handling and register/warp matrix solves inform these specializations; it does not use a tiny foot-only candidate set. See [the G1 model description](../fixtures/mjlab/README.md).

These forward solvers support the supplied topology. Changing the model requires revisiting the hard-coded proxies and maps. Model notices and licenses are in [licenses/](licenses/); source attributions to MuJoCo, MuJoCo Warp and NablaEngine are retained.
