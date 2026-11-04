# Newton / MuJoCo Warp environments

Three standalone scenes use task-specific fused CUDA physics: Go2 locomotion, Franka reach, and Franka lift. Each has an MJCF collision model, Newton state/control buffers, a MuJoCo Warp baseline, and the corresponding fused solver. They require Linux, an NVIDIA GPU, and a CUDA toolkit with `nvcc`.

## Run

Use a separate environment from the upstream G1 package, which pins different dependencies. From the repository root:

```bash
uv venv .venv-tasks --python 3.12
uv pip install --python .venv-tasks/bin/python -r environments/requirements.txt
export CUDA_ARCH=sm_89  # NVIDIA L4; set this to your GPU's architecture
.venv-tasks/bin/python -m environments.run --task go2 --engine fused --num-envs 128
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

Go2 accepts 12 joint angles and updates torque-limited PD control every 5 ms. Franka accepts nine absolute joint targets; position/velocity actuators and a finger equality remain active. In lift, fingers hold the object through contact and friction; there is no gripper-object attachment. The table is kinematic. Eight object families cycle across worlds: four boxes, a sphere, and three capsules. The native baseline uses their convex meshes; fused physics uses the matching analytic primitives.

Run `python -m environments.verify --task franka_lift` with the task environment's Python to check reset semantics, one-control agreement and 100 finite control steps in both engines. This is a packaging check, separate from the original behavioral evaluation.

## Benchmark

```bash
for task in go2 franka_reach franka_lift; do
  for count in 128 512 2048; do
    for engine in mjwarp fused; do
      .venv-tasks/bin/python -m environments.benchmark \
        --task "$task" --num-envs "$count" --engine "$engine"
    done
  done
done
```

CUDA events measure one captured physics control interval from a restored nominal state. Reset, JIT compilation, controller preparation and observations are excluded. Both engines refresh contacts every integration: four 5 ms integrations for Go2, eight 4.167 ms integrations for Franka. The fused solvers execute four, one and four physics launches per control interval, respectively. A separate wall-clock number includes the minimal Python control loop and CPU state observations.

The [tiled plot](../docs/assets/task_speedups.svg) and [reference ratios](reference_speedups.json) are the existing validated investigation results on an NVIDIA L4. Physics speedups are 1.15–6.57×; end-to-end speedups are 1.39–2.12×. All nine original behavioral comparisons passed. Those end-to-end runs include the original full task runtime, which stays local and is not distributed here. For Go2 they compare the original eight-integration baseline with four fused integrations; physics-only ratios use four integrations on both sides. The standalone runner has nominal resets and absolute joint actions, so its new timings are a separate experiment. It does not reproduce the original randomized evaluation or its end-to-end ratios.

Regenerate the tiled figure with `uv run --no-project --with matplotlib python -m environments.plot_speedups`.

## Implementation map

| File | Role |
| --- | --- |
| `environment.py` | Load MJCF into Newton, replicate worlds, set controls, reset and step either engine |
| `run.py`, `benchmark.py` | Headless rollout and scoped timing |
| `<task>/model.xml` | Primitive collision geometry, articulation, inertia, friction and actuators |
| `<task>/compact.py` | Specialize model constants, pack varying fields, compile and launch CUDA |
| `<task>/build_kernel.py` and transform helpers | Assemble fixed-size, lane-parallel task kernels |
| `<task>/physics_core.cuh`, `full_reach.py` | Dynamics, contact rows, constraint solve and integration |
| `<task>/direct_io.cuh`, `launch.cuh` | Read/write Newton buffers and fuse the integrations |
| `<task>/*collision*.cuh`, `analytic_proxy.cuh` | Task contact geometry and lift object primitives |

One warp handles a world, with two worlds per block in the Go2 and reach builds at larger batches. Body transforms, mass/bias terms, contacts and solver workspaces stay in shared memory during a launch. Topology and constant model fields are compiled in; varying physical fields use a compact parameter buffer. This is forward simulation only; it does not add gradients for these tasks to the upstream G1 API.

The Go2 implementation stores only free-base and same-leg Jacobian entries. Reach keeps all nine joint coordinates and fuses the eight integrations. Lift keeps full articulated mass and bias calculations, removes the kinematic table's six solver degrees of freedom, and uses the eight analytic object primitives. Fixed collision pairs and bounded constraint storage replace general scene dispatch. These kernels support the supplied topology; changing geometry or the joint tree requires revisiting that specialization.

```text
Newton state + controls
    → kinematics and mass/bias
    → contact refresh and constraint rows
    → Newton constraint solve
    → integration (repeat within the launch where fused)
    → Newton state + contact outputs
```

Model coordinates are centered per world and asset names are simplified. Collision models contain numeric inertial data and primitives, not external rendering assets. Robot model notices and licenses are in [licenses/](licenses/); the source retains its MuJoCo, MuJoCo Warp and NablaEngine attributions.
