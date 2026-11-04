"""Check model import, reset, one-step agreement and nominal rollout stability."""
import argparse
import json
import numpy as np
from .environment import Environment, TASKS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--num-envs', type=int, default=128)
    args = p.parse_args()
    native = Environment(args.task, args.num_envs, 'mjwarp')
    fused = Environment(args.task, args.num_envs, 'fused')
    for env in [native, fused]:
        np.testing.assert_array_equal(env.observations()['joint_position'], env.initial_q)
        # MJCF abbreviated solref vectors must retain their damping component.
        assert np.all(env.solver.mjw_model.geom_solref.numpy()[..., 1] > 0)
        if args.task == 'franka_lift':
            assert env.solver.mj_model.nmesh == 8
        env.step()
    x, y = native.observations(), fused.observations()
    q_error = float(np.max(np.abs(x['joint_position'] - y['joint_position'])))
    v_error = float(np.max(np.abs(x['joint_velocity'] - y['joint_velocity'])))
    assert q_error < 1e-4 and v_error < 1e-3, (q_error, v_error)
    for env in [native, fused]:
        env.reset()
        np.testing.assert_array_equal(env.observations()['joint_position'], env.initial_q)
        for _ in range(100):
            env.step()
        assert all(np.isfinite(x).all() for x in env.observations().values())
    assert np.max(fused.fused.diagnostic()[:, 2]) == 0
    assert not fused.fused.parameter_audit()
    print(json.dumps(dict(task=args.task, environments=args.num_envs,
                         one_control_max_position_difference=q_error,
                         one_control_max_velocity_difference=v_error,
                         nominal_controls=100, passed=True), indent=2))


if __name__ == '__main__':
    main()
