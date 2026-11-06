# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Allegro cube interaction, adapted from Newton's robot_allegro_hand example."""
from pathlib import Path
import numpy as np
import warp as wp
import newton
from newton import JointTargetMode, ModelFlags

@wp.kernel
def move_hand(
    joint_q_start: wp.array[wp.int32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    sim_time: wp.array[wp.float32],
    sim_dt: float,
    hand_rotation: wp.quat,
    # outputs
    joint_target_q: wp.array[wp.float32],
    joint_parent_xform: wp.array[wp.transform],
):
    world_id = wp.tid()
    root_joint_id = world_id * 22
    t = sim_time[world_id]

    root_dof_start = joint_q_start[root_joint_id]

    # animate the finger joints
    for i in range(20):
        di = root_dof_start + i
        target = wp.sin(t + float(i * 6) * 0.1) * 0.08 + 0.3
        joint_target_q[di] = wp.clamp(target, joint_limit_lower[di], joint_limit_upper[di])

    # animate the root joint transform
    q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.sin(t) * 0.1)
    root_xform = joint_parent_xform[root_joint_id]
    joint_parent_xform[root_joint_id] = wp.transform(root_xform.p, q * hand_rotation)

    # update the sim time
    sim_time[world_id] += sim_dt


class Environment:
    def __init__(self, num_envs=128, engine="fused"):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.solver_type = 'mujoco'
        self.world_count = num_envs
        if self.world_count is None:
            self.world_count = 100


        self.device = wp.get_device()

        self.hand_rotation = wp.normalize(wp.quat(0.21643, 0.706218, -0.648166, 0.185191))
        max_contacts_per_world = 300

        allegro_hand = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(allegro_hand)
        allegro_hand.default_shape_cfg.ke = 1.0e3
        allegro_hand.default_shape_cfg.kd = 1.0e2
        allegro_hand.default_shape_cfg.margin = 0.005
        allegro_hand.default_shape_cfg.gap = 0.015

        asset_file = str(Path(__file__).with_name("model.usda"))
        allegro_hand.add_usd(
            asset_file,
            xform=wp.transform(wp.vec3(0, 0, 0.5)),
            enable_self_collisions=False,
            ignore_paths=[".*Dummy", ".*CollisionPlane"],
            hide_collision_shapes=True,
            load_static_visual_shapes=False,
        )

        # set joint targets and joint drive gains (only on hand, not the floating-body cube)
        for i in range(allegro_hand.joint_dof_count - 6):
            allegro_hand.joint_target_ke[i] = 150
            allegro_hand.joint_target_kd[i] = 5
            allegro_hand.joint_q[i] = 0.3
            allegro_hand.joint_target_q[i] = 0.3
            if allegro_hand.joint_label[i][-2:] == "_0":
                allegro_hand.joint_q[i] = 0.6
                allegro_hand.joint_target_q[i] = 0.6
            allegro_hand.joint_target_mode[i] = int(JointTargetMode.POSITION)
            if allegro_hand.joint_type[i] == newton.JointType.REVOLUTE:
                allegro_hand.joint_armature[i] = 1e-2

        # Update root pose of the cube (free joint)
        q = np.array(allegro_hand.joint_q)
        q[-7:-4] += np.array([0.0, 0.0, 0.05])
        q[-4:] = wp.quat_rpy(0.3, 0.5, 0.1)
        allegro_hand.joint_q = q.tolist()

        builder = newton.ModelBuilder()
        builder.replicate(allegro_hand, self.world_count)

        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0, 0, -.05)),
                              hx=10., hy=10., hz=.05, label='floor')

        self.model = builder.finalize()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.initial_world_positions = self.model.body_q.numpy()[:: allegro_hand.body_count, :3].copy()

        # Find the cube body index (it's the last body in each world)
        self.cube_body_offset = allegro_hand.body_count - 1

        self.world_time = wp.zeros(self.world_count, dtype=wp.float32)

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            njmax=200,
            nconmax=max_contacts_per_world,
            impratio=20.0,
            # Preserve the example's solref-inherited grasp friction; its
            # purpose is articulation control rather than kf mapping.
            cone="pyramidal",
            iterations=100,
            ls_iterations=50,
            use_mujoco_contacts=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        self.engine_name = engine
        self.solver_dt = self.sim_dt
        self.fused = None
        self.native_pipeline = self.collision_pipeline
        if engine == 'fused':
            from .compact import Compact
            self.fused = Compact(self.solver, 'allegro', self)
        elif engine != 'mjwarp':
            raise ValueError("engine must be fused or mjwarp")

    def prepare_control(self):
        wp.launch(move_hand, dim=self.world_count,
                  inputs=[self.model.joint_q_start, self.model.joint_limit_lower,
                          self.model.joint_limit_upper, self.world_time,
                          self.sim_dt, self.hand_rotation],
                  outputs=[self.control.joint_target_q, self.model.joint_X_p])
        if self.fused is None:
            self.solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)

    def integrate(self):
        if self.fused is None:
            self.native_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.sim_dt)

    def physics_step(self):
        """Eight integrations with fixed prepared targets and root pose."""
        for _ in range(self.sim_substeps):
            self.integrate()

    def step(self):
        """The original sinusoidal fingers and moving root, updated each substep."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.prepare_control()
            self.integrate()

    def metrics(self):
        q = self.state_0.body_q.numpy().reshape(self.world_count, -1, 7)
        v = self.state_0.joint_qd.numpy()
        cubes = q[:, self.cube_body_offset]
        origin = self.initial_world_positions
        held = ((np.abs(cubes[:, :2] - origin[:, :2]) <= .5).all(1)
                & (cubes[:, 2] >= .9) & (cubes[:, 2] <= origin[:, 2] + .5))
        finite = bool(np.isfinite(q).all() and np.isfinite(v).all())
        near = bool((np.abs(q[:, :self.cube_body_offset, :3] - origin[:, None, :]) <= .5).all())
        # The fixed root must follow the controller, rather than its preceding pose.
        command = self.model.joint_X_p.numpy().reshape(self.world_count, -1, 7)[:, 0]
        root = q[:, 0]
        quat_error = np.minimum(np.linalg.norm(root[:, 3:] - command[:, 3:], axis=1),
                                np.linalg.norm(root[:, 3:] + command[:, 3:], axis=1))
        root_error = float(max(np.max(np.abs(root[:, :3] - command[:, :3])), quat_error.max()))
        flags = (self.fused.diag.numpy()[:, 2] if self.fused else self.solver.mjw_data.overflow.numpy()).astype(np.int64)
        return dict(finite=finite, hand_near_origin=near, held_fraction=float(held.mean()),
                    mean_cube_height=float(cubes[:, 2].mean()),
                    overflow_or=int(np.bitwise_or.reduce(flags.ravel())), root_pose_error=root_error,
                    root_tracks_command=bool(root_error < 1e-5),
                    example_pass=bool(finite and near and held.mean() >= .5))
