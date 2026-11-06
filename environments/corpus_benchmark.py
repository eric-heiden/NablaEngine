"""Paired physics-only timing from contact-rich rollout states.

benchmark.py times one control from the nominal reset, before Go2 or H1
touch the ground. This script first rolls out MuJoCo Warp with the nominal
targets, saves controls 16, 64 and 111, and times one captured control
interval of each engine from the same saved inputs. Each round averages 48
restored replays per state with equal state weighting; the result is the
median of five round means after a discarded first round.

Run from the repository root:
python -m environments.corpus_benchmark --task h1 --num-envs 4096 --output h1_4096.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from .environment import Environment, TASKS

STATES = (16, 64, 111)
DATA_FIELDS = ('qpos', 'qvel', 'qacc_warmstart', 'qacc', 'act', 'ctrl', 'time', 'qfrc_applied', 'xfrc_applied')


def arrays(env, data=True):
    out, seen = [], set()
    groups = [vars(env.state), vars(env.control)]
    if data:
        groups.append({k: getattr(env.solver.mjw_data, k, None) for k in DATA_FIELDS})
    for group in groups:
        for name, a in group.items():
            if isinstance(a, wp.array) and a.size and a.ptr not in seen:
                seen.add(a.ptr)
                out.append((name, a))
    return out


def capture(fn):
    with wp.ScopedCapture(device='cuda:0') as c:
        fn()
    return c.graph


def timing(graph, reset):
    start, warm = time.perf_counter(), 0
    while warm < 5 or time.perf_counter() - start < .5:
        wp.capture_launch(reset)
        wp.capture_launch(graph)
        wp.synchronize()
        warm += 1
    events = [(wp.Event(enable_timing=True), wp.Event(enable_timing=True)) for _ in range(48)]
    rounds = []
    for r in range(6):
        for first, last in events:
            wp.capture_launch(reset)
            wp.record_event(first)
            wp.capture_launch(graph)
            wp.record_event(last)
        wp.synchronize()
        if r:
            rounds.append(float(np.mean([wp.get_event_elapsed_time(f, l) for f, l in events])))
    return rounds


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--num-envs', type=int, default=128)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    wp.init()
    native = Environment(args.task, args.num_envs, 'mjwarp')
    fused = Environment(args.task, args.num_envs, 'fused')
    native_arrays, fused_arrays = arrays(native), arrays(fused, data=False)
    result = dict(task=args.task, environments=args.num_envs, gpu=wp.get_device('cuda:0').name,
                  sm_count=wp.get_device('cuda:0').sm_count, states=list(STATES), samples=[],
                  resources=fused.fused.metadata['resources'],
                  integrations=native.integrations, integration_dt=native.integration_dt)
    native_step = capture(native.step)
    for frame in range(1, max(STATES) + 1):
        wp.capture_launch(native_step)
        if frame not in STATES:
            continue
        wp.synchronize()
        saved = [(a, wp.clone(a)) for _, a in native_arrays]
        restore_native = capture(lambda: [wp.copy(dst, src) for dst, src in saved])
        native_ms = timing(capture(native.physics_step), restore_native)
        # The fused engine starts from the identical Newton state and MuJoCo warm start.
        by_name = dict(native_arrays)
        fused_saved = [(dst, wp.clone(by_name[name])) for name, dst in fused_arrays]
        # Lift's kinematic table keeps a MuJoCo free joint after the 15 dynamic DOFs; the fused
        # solver omits it, so take the leading DOFs of each world.
        native_warm = native.solver.mjw_data.qacc_warmstart.numpy().reshape(args.num_envs, -1)
        warm = wp.array(native_warm[:, :fused.fused.nv].copy(), dtype=wp.float32, device='cuda:0')

        def restore_fused():
            for dst, src in fused_saved:
                wp.copy(dst, src)
            wp.copy(fused.fused.warm, warm)
            fused.fused.diag.zero_()
        fused.fused.tick_phase = 0
        reset = capture(restore_fused)
        fused_graph = capture(fused.physics_step)
        fused_ms = timing(fused_graph, reset)
        wp.capture_launch(reset)
        wp.capture_launch(fused_graph)
        diag = fused.fused.diagnostic()
        q = fused.state.joint_q.numpy()
        result['samples'].append(dict(control_index=frame, native_round_means_ms=native_ms,
                                      fused_round_means_ms=fused_ms,
                                      fused_max_contacts=float(diag[:, 0].max()),
                                      fused_mean_contacts=float(diag[:, 0].mean()),
                                      fused_mean_final_iterations=float(diag[:, 3].mean()),
                                      fused_error_flags=int(diag[:, 2].max()),
                                      fused_finite=bool(np.isfinite(q).all())))
        for dst, src in saved:
            wp.copy(dst, src)
        print('MEASURED', args.task, args.num_envs, frame, flush=True)
    native_rounds = np.mean([s['native_round_means_ms'] for s in result['samples']], axis=0)
    fused_rounds = np.mean([s['fused_round_means_ms'] for s in result['samples']], axis=0)
    result['baseline_ms'] = float(np.median(native_rounds))
    result['fused_ms'] = float(np.median(fused_rounds))
    result['physics_speedup'] = result['baseline_ms'] / result['fused_ms']
    result['baseline_steps_per_second'] = args.num_envs * 1000 / result['baseline_ms']
    result['fused_steps_per_second'] = args.num_envs * 1000 / result['fused_ms']
    Path(args.output).write_text(json.dumps(result, indent=2))
    print('RESULT', json.dumps({k: result[k] for k in ('task', 'environments', 'baseline_ms', 'fused_ms', 'physics_speedup')}), flush=True)


if __name__ == '__main__':
    main()
