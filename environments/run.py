"""Headless joint-target rollout. Run from the repository root with -m."""
import argparse
import json
import numpy as np
from .environment import Environment, TASKS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=TASKS, default='go2')
    p.add_argument('--engine', choices=['fused', 'mjwarp'], default='fused')
    p.add_argument('--num-envs', type=int, default=128)
    p.add_argument('--steps', type=int, default=100)
    args = p.parse_args()
    if args.steps < 1:
        p.error('--steps must be positive')
    env = Environment(args.task, args.num_envs, args.engine)
    for _ in range(args.steps):
        env.step()
    obs = env.observations()
    finite = all(np.isfinite(x).all() for x in obs.values())
    flags = int(env.fused.diagnostic()[:, 2].max()) if env.fused else 0
    print(json.dumps(dict(task=args.task, engine=args.engine, environments=args.num_envs,
                         control_steps=args.steps, finite=finite, solver_flags=flags,
                         first_world_joint_position=obs['joint_position'][0].tolist()), indent=2))
    if not finite or flags:
        raise SystemExit('Rollout failed its finite-state or solver diagnostic check.')


if __name__ == '__main__':
    main()
