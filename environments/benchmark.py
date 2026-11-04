"""CUDA-event physics benchmark; standalone control-loop wall time is separate."""
import argparse
import json
import time
import numpy as np
import warp as wp
from .environment import Environment, TASKS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--engine', choices=['fused', 'mjwarp'], required=True)
    p.add_argument('--num-envs', type=int, default=128)
    p.add_argument('--replays', type=int, default=50)
    p.add_argument('--rounds', type=int, default=5)
    args = p.parse_args()
    if args.replays < 1 or args.rounds < 1:
        p.error('rounds and replays must be positive')
    env = Environment(args.task, args.num_envs, args.engine)
    for _ in range(5):
        env.step()
    wp.synchronize()
    # Capture only physics; allocations/JIT and controller preparation stay outside.
    env.reset()
    env.prepare_control()
    with wp.ScopedCapture(device='cuda:0') as capture:
        env.physics_step()
    physics_ms = []
    for _ in range(args.rounds):
        samples = []
        for _ in range(args.replays):
            env.reset()
            env.prepare_control()
            start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
            wp.record_event(start)
            wp.capture_launch(capture.graph)
            wp.record_event(end)
            wp.synchronize_event(end)
            samples.append(wp.get_event_elapsed_time(start, end))
        physics_ms.append(float(np.mean(samples)))
    # Includes Python step dispatch, PD preparation and one CPU state observation.
    standalone_ms = []
    for _ in range(args.rounds):
        env.reset()
        wp.synchronize()
        start = time.perf_counter()
        for _ in range(args.replays):
            env.step()
            env.observations()
        wp.synchronize()
        standalone_ms.append(1000 * (time.perf_counter() - start) / args.replays)
    physics = float(np.median(physics_ms))
    standalone = float(np.median(standalone_ms))
    print(json.dumps(dict(task=args.task, engine=args.engine, environments=args.num_envs,
                         physics_ms_per_control=physics,
                         physics_environment_steps_per_second=args.num_envs * 1000 / physics,
                         standalone_loop_ms_per_control=standalone,
                         standalone_loop_environment_steps_per_second=args.num_envs * 1000 / standalone,
                         physics_round_means_ms=physics_ms,
                         standalone_loop_round_means_ms=standalone_ms,
                         note='Standalone scene timings; not a rerun of the archived task-runtime benchmark.'), indent=2))


if __name__ == '__main__':
    main()
