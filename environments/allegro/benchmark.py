"""Paired physics-only timing and the unchanged 20-second Allegro controller.

Run from the repository root: python -m environments.allegro.benchmark --num-envs 128 --output allegro.json
"""
import argparse, ctypes as ct, json, os, time
from pathlib import Path
import numpy as np
import warp as wp
from .environment import Environment
from .compact import Compact

STAGES=['import','kinematics_bias','mass_and_smooth_solve','collision','constraint_build','constraint_solve','integration_export']

def snapshot(e):
    arrays=[]; seen=set()
    objects=[('state',e.state_0,vars(e.state_0)),('control',e.control,vars(e.control)),
             ('data',e.solver.mjw_data,['qpos','qvel','qacc_warmstart','qacc','act','ctrl','time','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat','overflow']),
             ('model',e.model,['joint_X_p']),('mjmodel',e.solver.mjw_model,['body_pos','body_quat'])]
    for prefix,obj,names in objects:
        for name in names:
            x=getattr(obj,name,None)
            if isinstance(x,wp.array) and x.size and x.ptr not in seen:
                seen.add(x.ptr);arrays.append((prefix+'.'+name,x,wp.clone(x)))
    arrays.append(('world_time',e.world_time,wp.clone(e.world_time)))
    return arrays

def restore(arrays):
    for _,dst,src in arrays:wp.copy(dst,src)

def capture(fn):
    with wp.ScopedCapture() as cap:fn()
    return cap.graph

def timing(graph,reset):
    start=time.perf_counter();warm=0
    while warm<5 or time.perf_counter()-start<.5:
        wp.capture_launch(reset);wp.capture_launch(graph);wp.synchronize();warm+=1
    events=[(wp.Event(enable_timing=True),wp.Event(enable_timing=True)) for _ in range(48)]
    rounds=[]
    for r in range(6):
        for first,last in events:
            wp.capture_launch(reset);wp.record_event(first);wp.capture_launch(graph);wp.record_event(last)
        wp.synchronize()
        if r:rounds.append(float(np.mean([wp.get_event_elapsed_time(f,l) for f,l in events])))
    return dict(median_ms=float(np.median(rounds)),round_means_ms=rounds,replays_per_round=48)

def run(args):
    wp.init()
    with wp.ScopedDevice('cuda:0'):
        e=Environment(args.num_envs,'mjwarp');initial=snapshot(e)
        native={key:getattr(e.solver,key) for key in ['step','reset','notify_model_changed','update_contacts']}
        native_collision=e.solver.mjw_model.opt.run_collision_detection
        result=dict(environments=args.num_envs,control_dt=.02,integration_dt=.0025,integrations=8,
                    contact_refresh='every integration',collision_geometry='20 cuboids; identical in both engines',samples=[],validation={},gpu=wp.get_device().name)
        result['acceptance']={'example_held_minimum':.5,'held_fraction_absolute_difference_maximum':.10,'mean_cube_height_difference_maximum_m':.05,'finite':True,'overflow_or':0,'hand_near_origin':True,'root_pose_error_maximum':1e-5}
        def save():Path(args.output).write_text(json.dumps(result,indent=2))
        engine=None
        e.step();restore(initial);wp.synchronize()
        native_graph=capture(e.step)
        trace=[]
        for frame in range(1,args.steps+1):
            wp.capture_launch(native_graph)
            if frame in (16,64,111):
                state=snapshot(e)
                if args.states:np.savez_compressed(Path(args.states)/f'allegro_{args.num_envs}_{frame}.npz',**{key:src.numpy() for key,_,src in state})
                row={'control_index':frame}
                graph=capture(e.physics_step);reset=capture(lambda:restore(state))
                if not args.profile:
                    row['baseline']=timing(graph,reset)
                else:
                    for _ in range(3):wp.capture_launch(reset);wp.capture_launch(graph)
                    wp.synchronize()
                    runtime=ct.CDLL(os.environ['CUDART_LIBRARY']);runtime.cudaProfilerStart()
                    for _ in range(3):wp.capture_launch(graph)
                    wp.synchronize();runtime.cudaProfilerStop()
                restore(state)
                if engine is None:
                    engine=Compact(e.solver,'allegro',e)
                e.fused=engine
                for key in native:setattr(e.solver,key,{'step':engine.step,'update_contacts':engine.update_contacts,'notify_model_changed':lambda *a,**k:None}.get(key,native[key]))
                engine.warm.assign(e.solver.mjw_data.qacc_warmstart.numpy().copy());warm=wp.clone(engine.warm)
                def reset_fused():
                    restore(state);wp.copy(engine.warm,warm);engine.diag.zero_()
                reset2=capture(reset_fused);graph2=capture(e.physics_step)
                if not args.profile:row['fused']=timing(graph2,reset2)
                else:
                    for _ in range(3):wp.capture_launch(reset2);wp.capture_launch(graph2)
                    wp.synchronize();runtime.cudaProfilerStart()
                    for _ in range(3):wp.capture_launch(graph2)
                    wp.synchronize();runtime.cudaProfilerStop()
                reset_fused();wp.capture_launch(graph2);wp.synchronize();row['fused_state_check']=e.metrics()
                if engine.cycles.size:
                    engine.cycles.zero_()
                    for _ in range(3):wp.capture_launch(reset2);wp.capture_launch(graph2)
                    cycles=engine.cycles.numpy().sum(0).astype(float);row['stage_cycle_percent']=dict(zip(STAGES,(100*cycles/cycles.sum()).tolist()))
                row['resources']=engine.metadata['resources'];result['samples'].append(row)
                restore(state);e.fused=None
                for key,value in native.items():setattr(e.solver,key,value)
                e.solver.mjw_model.opt.run_collision_detection=native_collision
                save();print('MEASURED',args.num_envs,frame,flush=True)
            if frame%25==0 or frame==args.steps:
                trace.append(dict(frame=frame,**e.metrics()))
                if frame%100==0:print('BASELINE',frame,trace[-1],flush=True)
        result['validation']['baseline']=dict(trace=trace,final=trace[-1]);save()
        restore(initial);engine.warm.zero_();engine.diag.zero_();e.fused=engine
        e.solver.step=engine.step
        fused_graph=capture(e.step);trace=[];poses=[]
        for frame in range(1,args.steps+1):
            wp.capture_launch(fused_graph)
            if args.video and frame%2==0:poses.append(e.state_0.body_q.numpy().reshape(args.num_envs,-1,7)[0].copy())
            if frame%25==0 or frame==args.steps:
                trace.append(dict(frame=frame,**e.metrics()))
                if frame%100==0:print('FUSED',frame,trace[-1],flush=True)
        result['validation']['fused']=dict(trace=trace,final=trace[-1])
        x=result['validation']['baseline']['final'];y=result['validation']['fused']['final']
        checks={'finite':all(r['finite'] for group in result['validation'].values() for r in group['trace']),
                'zero_flags':all(r['overflow_or']==0 for group in result['validation'].values() for r in group['trace']),
                'root_tracks_command':all(r['root_tracks_command'] for group in result['validation'].values() for r in group['trace']),
                'hand_near_origin':all(r['hand_near_origin'] for group in result['validation'].values() for r in group['trace']),
                'original_example':x['example_pass'] and y['example_pass'],
                'held_fraction_difference':abs(x['held_fraction']-y['held_fraction'])<=.10,
                'cube_height_difference':abs(x['mean_cube_height']-y['mean_cube_height'])<=.05}
        result['checks']=checks;result['validation_pass']=all(checks.values())
        result['implementation']=engine.metadata
        if not args.profile:
            result['baseline_ms']=float(np.median(np.mean([r['baseline']['round_means_ms'] for r in result['samples']], axis=0)))
            result['fused_ms']=float(np.median(np.mean([r['fused']['round_means_ms'] for r in result['samples']], axis=0)))
            result['physics_speedup']=result['baseline_ms']/result['fused_ms']
            result['fused_steps_per_second']=args.num_envs*1000/result['fused_ms']
            result['baseline_steps_per_second']=args.num_envs*1000/result['baseline_ms']
        if args.video:np.savez_compressed(args.video,body_q=np.asarray(poses),dt=.04)
        save();print('RESULT',args.output,result.get('physics_speedup'),checks,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--num-envs',type=int,default=128);p.add_argument('--steps',type=int,default=1000);p.add_argument('--output',required=True);p.add_argument('--states');p.add_argument('--video');p.add_argument('--profile',action='store_true');a=p.parse_args()
    if a.num_envs<2 or a.steps<111:p.error('Use at least 2 environments and 111 control steps.')
    if a.states:Path(a.states).mkdir(parents=True,exist_ok=True)
    run(a)
if __name__=='__main__':main()
