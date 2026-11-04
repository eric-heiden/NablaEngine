"""Small Newton/MuJoCo Warp environment API; CUDA kernels live beside each MJCF."""
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo

TASKS = ('go2', 'h1', 'franka_reach', 'franka_lift')


@wp.kernel
def leg_pd(q: wp.array(dtype=float), v: wp.array(dtype=float),
           target: wp.array(dtype=float), force: wp.array(dtype=float)):
    w, j = wp.tid()
    torque = 25.0 * (target[w * 12 + j] - q[w * 19 + 7 + j]) - 0.5 * v[w * 18 + 6 + j]
    force[w * 18 + 6 + j] = wp.clamp(torque, -23.5, 23.5)


class Environment:
    """Batched fixed-topology scenes. Actions are absolute joint position targets.

    Go2 has 12 targets, H1 has 19. Franka has 9 (7 arm, 2 fingers); the finger equality
    couples their motion. The model and solver are available for direct use.
    Only cuda:0 is supported by the fused launch adapter.
    """
    def __init__(self, task='go2', num_envs=128, engine='fused'):
        if task not in TASKS or engine not in ('fused', 'mjwarp'):
            raise ValueError('Choose a documented task and fused or mjwarp engine.')
        if num_envs < 8 or num_envs % 8:
            raise ValueError('num_envs must be a positive multiple of 8 (all lift object families).')
        self.task, self.num_envs, self.engine = task, num_envs, engine
        folder = Path(__file__).parent / task
        single = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(single)
        single.add_mjcf(str(folder / 'model.xml'), parse_visuals=False, parse_sites=False)
        if task == 'franka_lift':
            # Preserve the authored table as a kinematic free body.
            single.body_flags[-1] = int(newton.BodyFlags.KINEMATIC)
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(single, num_envs)
        if task == 'franka_lift':
            # Keep all eight object meshes in order; each world uses one family.
            mesh_nodes = ET.parse(folder / 'model.xml').getroot().find('asset').findall('mesh')
            meshes = [newton.Mesh(
                vertices=np.fromstring(e.get('vertex'), sep=' ', dtype=np.float32).reshape(-1, 3),
                indices=np.fromstring(e.get('face'), sep=' ', dtype=np.int32),
            ) for e in mesh_nodes]
            object_shape = next(i for i, src in enumerate(single.shape_source) if isinstance(src, newton.Mesh))
            for w in range(num_envs):
                builder.shape_source[w * single.shape_count + object_shape] = meshes[w % 8]
        self.model = builder.finalize(device='cuda:0')
        self.state = self.model.state()
        self.control = self.model.control()
        self.nq = self.model.joint_coord_count // num_envs
        self.nv = self.model.joint_dof_count // num_envs
        self.action_size = {'go2': 12, 'h1': 19}.get(task, 9)
        self.control_dt = .02 if task in ('go2', 'h1') else 1 / 30
        self.integration_dt = {'go2': .005, 'h1': .0025}.get(task, 1 / 240)
        self.integrations = 4 if task == 'go2' else 8
        self.solver = SolverMuJoCo(self.model, use_mujoco_contacts=True, iterations=100)
        self.solver.mjw_model.opt.run_collision_detection = True
        self.initial_q = self.model.joint_q.numpy().reshape(num_envs, self.nq).copy()
        if task == 'go2':
            self.initial_q[:, 7:] = np.tile([0., .8, -1.5], 4)
            self.default_action = self.initial_q[:, 7:].copy()
        elif task == 'h1':
            self.initial_q[:, 7:] = [0., 0., -.28, .79, -.52, 0., 0., -.28, .79, -.52, 0., .28, 0., 0., .52, .28, 0., 0., .52]
            self.default_action = self.initial_q[:, 7:].copy()
        else:
            self.initial_q[:, :9] = [0., -.569, 0., -2.81, 0., 3.037, .741, .04, .04]
            if task == 'franka_lift':
                self.initial_q[:, 11] = .41
            self.default_action = self.initial_q[:, :9].copy()
        self.action = wp.array(self.default_action.ravel(), dtype=float, device='cuda:0')
        self.fused = None
        self.reset()
        if engine == 'fused':
            tag = {'go2': 'go2', 'h1': 'h1', 'franka_reach': 'reach', 'franka_lift': 'lift'}[task]
            adapter = SimpleNamespace(control=self.control, state_0=self.state,
                                      solver_dt=self.control_dt / 4, collision_pipeline=None)
            cls = import_module(f'.{task}.compact', __package__).Compact
            self.fused = cls(self.solver, tag, adapter)
        self.apply_action(self.default_action)

    def reset(self):
        self.state.joint_q.assign(self.initial_q.ravel())
        self.state.joint_qd.zero_()
        self.state.clear_forces()
        self.control.joint_f.zero_()
        self.control.joint_target_qd.zero_()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        if self.fused is not None:
            self.fused.tick_phase = 0
            self.solver.reset(self.state, flags=0)
        else:
            self.solver.reset(self.state, flags=0)
        self.apply_action(self.default_action)

    def apply_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape == (self.action_size,):
            action = np.broadcast_to(action, (self.num_envs, self.action_size))
        if action.shape != (self.num_envs, self.action_size) or not np.isfinite(action).all():
            raise ValueError(f'action must be finite with shape ({self.num_envs}, {self.action_size})')
        if self.task == 'go2':
            self.action.assign(action.ravel())
        else:
            targets = np.zeros((self.num_envs, self.control.joint_target_q.size // self.num_envs), np.float32)
            if self.task == 'h1':
                targets[:, 7:26] = action
            else:
                targets[:, :9] = action
            self.control.joint_target_q.assign(targets.ravel())

    def physics_step(self):
        """One control interval, contacts refreshed on every integration.

        Go2 PD force preparation is a controller operation; benchmark.py holds
        its force buffer fixed while timing pure physics.
        """
        ticks = 4 if self.engine == 'fused' else self.integrations
        dt = self.control_dt / ticks
        for _ in range(ticks):
            self.solver.step(self.state, self.state, self.control, None, dt)

    def prepare_control(self):
        if self.task == 'go2':
            wp.launch(leg_pd, (self.num_envs, 12),
                      inputs=[self.state.joint_q, self.state.joint_qd, self.action],
                      outputs=[self.control.joint_f], device='cuda:0')

    def step(self, action=None):
        if action is not None:
            self.apply_action(action)
        if self.task == 'go2':
            for _ in range(4):
                self.prepare_control()
                self.solver.step(self.state, self.state, self.control, None, self.integration_dt)
        else:
            self.physics_step()

    def observations(self):
        """CPU state snapshots for inspection; intentionally outside physics timing."""
        return {'joint_position': self.state.joint_q.numpy().reshape(self.num_envs, self.nq),
                'joint_velocity': self.state.joint_qd.numpy().reshape(self.num_envs, self.nv)}
