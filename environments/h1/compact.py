"""Task-specific shared-memory CUDA physics, with no MJWarp arrays in step()."""
from pathlib import Path
import ctypes as ct, hashlib, json, subprocess, sys, re
import os, shutil
import numpy as np
import warp as wp
from .build_kernel import physics_source
ROOT = Path(__file__).parent.resolve()
FIELDS = ['body_parentid', 'body_jntadr', 'body_jntnum', 'body_pos', 'body_quat', 'body_ipos', 'body_iquat', 'body_mass', 'body_inertia', 'body_invweight0', 'body_mocapid', 'jnt_type', 'jnt_qposadr', 'jnt_dofadr', 'jnt_pos', 'jnt_axis', 'jnt_limited', 'jnt_range', 'jnt_margin', 'jnt_solref', 'jnt_solimp', 'jnt_actfrclimited', 'jnt_actfrcrange', 'dof_armature', 'dof_damping', 'dof_invweight0', 'geom_type', 'geom_bodyid', 'geom_pos', 'geom_quat', 'geom_friction', 'geom_solref', 'geom_solimp', 'geom_condim', 'geom_priority', 'geom_solmix', 'actuator_gainprm', 'actuator_biasprm', 'actuator_gear', 'actuator_forcelimited', 'actuator_forcerange', 'actuator_ctrllimited', 'actuator_ctrlrange', 'qpos0', 'timestep', 'gravity', 'tolerance', 'meaninertia', 'impratio']

def literal(v, integer=False):
    if integer:
        return str(int(v))
    return f'{float(v):.9e}f'

def const_fn(name, values):
    a = np.asarray(values).ravel()
    integer = a.dtype.kind in 'biu'
    typ = 'int' if integer else 'float'
    if len(a) == 0:
        a = np.array([0], dtype=np.int32)
        typ = 'int'
        integer = True
    return f'__device__ const {typ} T_{name}[{len(a)}]={{' + ','.join((literal(v, integer) for v in a)) + '};' + f'__device__ __forceinline__ {typ} {name}(int i){{return (unsigned)i<{len(a)}?__ldg(T_{name}+i):0;}}'

class Compact:

    def __init__(self, sol, tag, backend):
        assert tag == 'h1'
        m, d, cpu = (sol.mjw_model, sol.mjw_data, sol.mj_model)
        nm = sol.model
        n = d.nworld
        self.sol = sol
        self.n = n
        self.nv = m.nv
        self.tag = 'h1'
        self.keep = []
        self.backend = backend
        from .task_specs import SPECS
        old = SPECS['h1']
        dims = dict(NMOCAP=cpu.nmocap, NV=m.nv, NQ=m.nq, NB=m.nbody, NG=m.ngeom, NJ=cpu.njnt, NU=m.nu, NEQ=cpu.neq, NR=old['defines']['NR'], NC=(old['defines']['NR'] + 3) // 4, ITERS=m.opt.iterations, NP=len(old['pairs']), NWJ=nm.joint_count // n, NWQ=nm.joint_coord_count // n, NWV=nm.joint_dof_count // n, NWB=nm.body_count // n, NWS=nm.shape_count // n, NWT=backend.control.joint_target_q.size // n, MAX_OUT=n * ((old['defines']['NR'] + 3) // 4))
        assert (dims['NQ'], dims['NV'], dims['NB']) == (26, 25, 21)
        dims['INTERNAL_STEPS'] = 2
        self.internal_steps = dims['INTERNAL_STEPS']
        capacity = 64
        dims.update(NR=capacity, NC=10, MAX_OUT=n * 10)
        dims['WORLDS_PER_BLOCK'] = 1
        dims['NWORLDS'] = n
        dims['STAGE_PROFILE'] = int(os.getenv('COMPACT_STAGE_PROFILE', '0'))
        self.nv = dims['NV']
        self.dt = float(backend.solver_dt) / self.internal_steps
        lines = ['#include <cuda_runtime.h>', '#include <math.h>'] + [f'#define {k} {v}' for k, v in dims.items()]
        lines.append('#define STATIC_TABLE 0')
        lines.append('#define REDUCED_GRIP 0')
        lines.extend(['#if STAGE_PROFILE', '#define MARK(K) s.ticks[K]=clock64()', '#else', '#define MARK(K) ((void)0)', '#endif'])
        lines += ['#define M(F,I) field_##F(s.params,(I))', '#define P(F,I) field_nw_##F(s.params,(I))', '#define D(F,I) dd.F[(I)]']
        pack = []
        pack_sources = []
        access = []
        specs = []
        immutable = []

        def add(name, a, shape, mutable=False):
            ar = a.numpy() if isinstance(a, wp.array) else np.asarray(a)
            integer = ar.dtype.kind in 'biu'
            typ = 'int' if integer else 'float'
            stride = int(np.prod(shape))
            ar = ar.reshape(-1, max(stride, 1)) if stride else ar.reshape(1, -1)
            if not stride:
                lines.append(f'__device__ __forceinline__ {typ} field_{name}(const float*p,int i){{return 0;}}')
                return
            mask = np.any(ar != ar[:1], axis=0) if len(ar) > 1 else np.zeros(stride, bool)
            ids = np.flatnonzero(mask)
            start = len(pack)
            if len(ids):
                pack.extend(ar[0, ids].tolist())
                specs.append({'field': name, 'offset': start, 'elements': len(ids), 'source_worlds': len(ar), 'source_indices': ids.tolist()})
            if isinstance(a, wp.array) and len(ar) > 1:
                pack_sources.append((name, a, stride, start, len(ar), ids, ar[0]))
            lookup = {int(v): start + k for k, v in enumerate(ids)}
            if stride <= 3 and len(ids) == 0:
                cases = [f'case {i}:return ' + literal(v, integer) + ';' for i, v in enumerate(ar[0])]
                lines.append(f'__device__ __forceinline__ {typ} field_{name}(const float*p,int i){{switch(i){{' + ''.join(cases) + 'default:return 0;}}')
            else:
                lines.append(f'__device__ const {typ} K_{name}[{stride}]={{' + ','.join((literal(v, integer) for v in ar[0])) + '};')
                if len(ids):
                    lines.append(f'__device__ const int L_{name}[{stride}]={{' + ','.join((str(lookup.get(i, -1)) for i in range(stride))) + '};')
                    lines.append(f'__device__ __forceinline__ {typ} field_{name}(const float*p,int i){{int k=__ldg(L_{name}+i);return k<0?__ldg(K_{name}+i):({typ})p[k];}}')
                else:
                    lines.append(f'__device__ __forceinline__ {typ} field_{name}(const float*p,int i){{return __ldg(K_{name}+i);}}')
            immutable.append({'field': name, 'elements': int((~mask).sum())})
        fixed = {'body_parentid', 'body_jntadr', 'body_jntnum', 'body_mocapid', 'body_quat', 'body_iquat', 'jnt_type', 'jnt_qposadr', 'jnt_dofadr', 'jnt_pos', 'jnt_axis', 'jnt_limited', 'jnt_range', 'jnt_margin', 'jnt_solref', 'jnt_solimp', 'jnt_actfrclimited', 'jnt_actfrcrange', 'dof_solref', 'dof_solimp', 'geom_type', 'geom_bodyid', 'geom_quat', 'geom_condim', 'geom_priority', 'geom_solmix', 'mesh_vertadr', 'mesh_vertnum', 'mesh_vert', 'eq_type', 'eq_obj1id', 'eq_obj2id', 'eq_data', 'eq_solref', 'eq_solimp', 'actuator_trnid', 'actuator_gear', 'actuator_forcelimited', 'actuator_forcerange', 'actuator_ctrllimited', 'actuator_ctrlrange', 'qpos0', 'tolerance', 'impratio', 'timestep'}
        for name in FIELDS:
            obj = m.opt if name in ['timestep', 'gravity', 'tolerance', 'impratio'] else m.stat if name == 'meaninertia' else m
            a = getattr(obj, name, None)
            raw = getattr(cpu.opt if name in ['timestep', 'gravity', 'tolerance', 'impratio'] else cpu.stat if name == 'meaninertia' else cpu, name)
            if a is None:
                a = np.asarray(raw)
            if name == 'timestep':
                a = np.asarray([self.dt], dtype=np.float32)
            add(name, a, np.shape(raw), mutable=name not in fixed)
        extra = {'Xp': (nm.joint_X_p, (dims['NWJ'], 7)), 'com': (nm.body_com, (dims['NWB'], 3)), 'ref': (nm.mujoco.dof_ref, (dims['NWV'],))}
        for name, (a, shape) in extra.items():
            add('nw_' + name, a, shape, True)
        lines.append(f'#define NPARAM {len(pack)}')
        self.nparam = len(pack)
        self.params = wp.empty((n, len(pack)), dtype=wp.float32, device='cuda:0')
        self.keep.append(self.params)
        mapping = {'mjq': sol.mj_q_start.numpy(), 'mjv': sol.mj_qd_start.numpy(), 'qstart': nm.joint_q_start.numpy()[:dims['NWJ']], 'vstart': nm.joint_qd_start.numpy()[:dims['NWJ']], 'child': nm.joint_child.numpy()[:dims['NWJ']], 'jtype': nm.joint_type.numpy()[:dims['NWJ']], 'nd': nm.joint_dof_dim.numpy()[:dims['NWJ']].sum(1), 'Xc': nm.joint_X_c.numpy()[:dims['NWJ']], 'kin': (nm.body_flags.numpy()[:dims['NWB']] & 2 != 0).astype(np.int32), 'bodymap': sol.mjc_body_to_newton.numpy()[0], 'freev': sol.body_free_qd_start.numpy()[:dims['NWB']], 'shape': sol.mjc_geom_to_newton_shape.numpy()[0], 'ctrlidx': sol.mjc_actuator_to_newton_idx.numpy() if m.nu else np.zeros(0, np.int32), 'ctrlq': sol.mjc_actuator_to_newton_target_q_idx.numpy() if m.nu else np.zeros(0, np.int32)}
        mapping['shape_stride'] = sol.mjc_geom_to_newton_shape.numpy()[1] - sol.mjc_geom_to_newton_shape.numpy()[0]
        xc = nm.joint_X_c.numpy().reshape(n, dims['NWJ'], 7)
        assert np.array_equal(xc, np.broadcast_to(xc[:1], xc.shape)), 'Per-world child anchors need their own parameter pack'
        shapes = sol.mjc_geom_to_newton_shape.numpy()
        assert np.array_equal(shapes, shapes[:1] + np.arange(n)[:, None] * mapping['shape_stride'][None, :]), 'Shape mapping is not affine by world'
        for name, a in mapping.items():
            lines.append(const_fn('C_' + name, a))
        assert not m.nu or not np.any(sol.mjc_actuator_ctrl_source.numpy() != 0)
        lines.append('__device__ const int PAIRS[NP*2]={' + ','.join((str(v) for p in old['pairs'] for v in p)) + '};')
        depths = [0] * cpu.nbody
        for b in range(1, cpu.nbody):
            depths[b] = depths[int(cpu.body_parentid[b])] + 1
        lines.append('#define TREE_DEPTH ' + str(max(depths)))
        lines.append('__device__ const int BODY_DEPTH[NB]={' + ','.join(map(str, depths)) + '};')
        masks = []
        for i in range(dims['NV']):
            target = int(cpu.dof_bodyid[i])
            mask = 0
            for body in range(1, cpu.nbody):
                b = body
                while b > target:
                    b = int(cpu.body_parentid[b])
                if b == target:
                    mask |= 1 << body
            masks.append(mask)
        slots = []
        count = 0
        for body in range(cpu.nbody):
            for mask in masks:
                active = bool(mask & 1 << body)
                slots.append(count if active else -1)
                count += int(active)
        lines.append(f'#define NJAC {count}')
        lines.append('__device__ const int J_SLOT[NB*NV]={' + ','.join(map(str, slots)) + '};')
        actids = []
        actcounts = []
        joints = []
        for i in range(dims['NV']):
            aa = [a for a in range(dims['NU']) if int(cpu.jnt_dofadr[cpu.actuator_trnid[a, 0]]) == i]
            assert len(aa) <= 2
            actcounts.append(len(aa))
            actids.extend(aa + [-1] * (2 - len(aa)))
            joints.append(int(cpu.dof_jntid[i]))
        for name, vals in [('ACT_COUNT', actcounts), ('ACT_IDS', actids), ('DOF_JOINT', joints)]:
            lines.append('__device__ const int ' + name + '[' + str(len(vals)) + ']={' + ','.join(map(str, vals)) + '};')
        lines.append('__device__ const unsigned int DOF_BODIES[NV]={' + ','.join((str(x) + 'u' for x in masks)) + '};')
        names = 'qin vin qout vout bq bv jforce bforce tq tv warm params diag device_cycles param_errors count shape0 shape1 point0 point1 normal cforce'.split()
        lines.append('struct IO {' + ''.join((('unsigned long long' if x == 'device_cycles' else 'int' if x in ['param_errors', 'count', 'shape0', 'shape1'] else 'float') + '*' + x + ';' for x in names)) + '};')
        self.IO = type('IO', (ct.Structure,), {'_fields_': [(k, ct.c_void_p) for k in names]})
        lines.append('struct Data {' + '' + 'float qpos[NQ],qvel[NV],qacc_warmstart[NV],qfrc_applied[NV],xfrc_applied[NB*6],ctrl[NU+1],table_pos[STATIC_TABLE*3+1],table_quat[STATIC_TABLE*4+1],mocap_pos[NMOCAP*3+1],mocap_quat[NMOCAP*4+1];};')
        self.Refresh = type('Refresh', (ct.Structure,), {'_fields_': [(k, ct.c_void_p) for k, *_ in pack_sources]})
        refresh_decl = []
        refresh_ops = []
        for field_id, (name, a, stride, start, count, ids, initial) in enumerate(pack_sources):
            integer = a.numpy().dtype.kind in 'biu'
            refresh_decl.append(('bool' if a.numpy().dtype.kind == 'b' else 'int' if integer else 'float') + '*' + name + ';')
            lookup = {int(v): start + k for k, v in enumerate(ids)}
            body = []
            for i, v in enumerate(initial):
                expr = f'r.{name}[(w%{count})*{stride}+{i}]'
                if i in lookup:
                    body.append(f'case {i}:out[w*NPARAM+{lookup[i]}]={expr};break;')
                else:
                    body.append(f'case {i}:if({expr}!={literal(v, integer)}){{atomicOr(errors+w,2048);atomicCAS(errors+gridDim.x+w*{len(pack_sources)}+{field_id},0,{i + 1});}}break;')
            refresh_ops.append(f'for(int i=threadIdx.x;i<{stride};i+=32){{switch(i){{' + ''.join(body) + '}}')
        lines.append('struct Refresh {' + ''.join(refresh_decl) + '};')
        lines.append('__global__ void refresh_parameters(Refresh r,float*out,int*errors){int w=blockIdx.x;' + ''.join(refresh_ops) + '}')
        self.refresh_args = self.Refresh(**{name: a.ptr for name, a, *_ in pack_sources})
        source = '\n'.join(lines) + '\n' + physics_source('h1') + '\n' + (ROOT / 'launch.cuh').read_text()
        key = hashlib.sha256((source + ''.join((p.read_text() for p in sorted(ROOT.glob('*.cuh'))))).encode()).hexdigest()[:16]
        build = ROOT / 'build'
        build.mkdir(exist_ok=True)
        path = build / f'{'h1'}_{key}.cu'
        path.write_text(source)
        libpath = path.with_suffix('.so')
        cmd = [os.environ.get('NVCC') or shutil.which('nvcc') or 'nvcc', '-O3', '-lineinfo', '-arch=' + os.environ.get('CUDA_ARCH', 'sm_89'), '--shared', '-Xcompiler=-fPIC', '--use_fast_math', '-Xptxas=-v', f'-I{ROOT}', '-o', str(libpath), str(path)]
        if not libpath.exists():
            p = subprocess.run(cmd, capture_output=True, text=True)
            path.with_suffix('.log').write_text(p.stdout + p.stderr)
            if p.returncode:
                raise RuntimeError(p.stderr)
        self.lib = ct.CDLL(str(libpath))
        self.lib.launch.argtypes = [ct.c_void_p, ct.c_int, ct.POINTER(self.IO)]
        self.lib.refresh.argtypes = [ct.c_void_p, ct.c_int, ct.POINTER(self.Refresh), ct.c_void_p, ct.c_void_p]
        self.lib.reset_warm.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p]
        self.lib.resources.argtypes = [ct.c_void_p]
        self.parameter_sources = pack_sources
        self.param_errors = wp.zeros(n * (1 + len(pack_sources)), dtype=wp.int32, device='cuda:0')
        self.warm = wp.zeros((n, self.nv), dtype=wp.float32, device='cuda:0')
        self.diag = wp.zeros((n, 4), dtype=wp.float32, device='cuda:0')
        self.count = wp.zeros(1, dtype=wp.int32, device='cuda:0')
        self.outputs = {k: wp.zeros(dims['MAX_OUT'], dtype=typ, device='cuda:0') for k, typ in [('shape0', wp.int32), ('shape1', wp.int32), ('point0', wp.vec3), ('point1', wp.vec3), ('normal', wp.vec3), ('cforce', wp.spatial_vector)]}
        self.cycles = wp.zeros((n, 7) if dims['STAGE_PROFILE'] else (0,), dtype=wp.uint64, device='cuda:0')
        self.io = self.IO(device_cycles=self.cycles.ptr, param_errors=self.param_errors.ptr, warm=self.warm.ptr, params=self.params.ptr, diag=self.diag.ptr, count=self.count.ptr, **{k: v.ptr for k, v in self.outputs.items()})
        res = np.zeros(8, np.int32)
        rc = self.lib.resources(res.ctypes.data)
        if rc:
            raise RuntimeError(f'CUDA resource initialization error {rc}')
        self.metadata = {'source': str(path), 'hash': key, 'defines': dims, 'parameters': specs, 'immutable_constants': immutable, 'parameter_bytes_per_world': len(pack) * 4, 'hot_path_history_bytes_per_world': 4 * self.nv + 20, 'allocated_diagnostic_bytes_per_world': 16 + 4 * (1 + len(pack_sources)) + (56 if dims['STAGE_PROFILE'] else 0), 'resources': dict(zip(['registers', 'static_shared_bytes', 'local_bytes_per_thread', 'resident_blocks', 'dynamic_shared_bytes', 'work_bytes', 'solve_bytes', 'data_bytes'], res.tolist())), 'build_command': cmd, 'pairs': old['pairs'], 'internal_substeps': self.internal_steps, 'specialization': '25 velocities; two capsule soles per foot and one torso capsule; sparse body Jacobians, direct actuator map and register solve', 'hot_path': 'direct Newton inputs and outputs; block-shared intermediate state; no MuJoCo Warp buffers'}
        self.refresh()
        native_notify = sol.notify_model_changed

        def notify(*a, **k):
            native_notify(*a, **k)
            self.refresh()
        sol.notify_model_changed = notify

        def reset(state, world_mask=None, flags=None):
            assert flags == 0, 'These tasks reset authored Newton state outside the solver; only warm history is owned here.'
            rc = self.lib.reset_warm(wp.get_stream().cuda_stream, n, self.warm.ptr, None if world_mask is None else world_mask.ptr)
            if rc:
                raise RuntimeError(rc)
        sol.reset = reset
        sol.step = self.step
        sol.update_contacts = self.update_contacts
        m.opt.run_collision_detection = True
        backend.collision_pipeline = None

    def refresh(self):
        rc = self.lib.refresh(wp.get_stream().cuda_stream, self.n, ct.byref(self.refresh_args), self.params.ptr, self.param_errors.ptr)
        if rc:
            raise RuntimeError(rc)

    def step(self, state_in, state_out, control, contacts, dt, **kwargs):
        assert state_in is state_out, 'Task uses in-place Newton state'
        assert abs(dt - self.internal_steps * self.dt) < 1e-09, (dt, self.dt)
        assert state_out.body_qdd is None and state_out.body_parent_f is None
        fields = {'qin': state_in.joint_q, 'vin': state_in.joint_qd, 'qout': state_out.joint_q, 'vout': state_out.joint_qd, 'bq': state_out.body_q, 'bv': state_out.body_qd, 'jforce': control.joint_f, 'bforce': state_in.body_f, 'tq': control.joint_target_q, 'tv': control.joint_target_qd}
        for k, a in fields.items():
            setattr(self.io, k, a.ptr)
        rc = self.lib.launch(wp.get_stream().cuda_stream, self.n, ct.byref(self.io))
        if rc:
            raise RuntimeError(f'compact launch failed {rc}')
        self.sol._step += 1

    def update_contacts(self, contacts, state=None, **kwargs):
        for k, field in [('shape0', 'rigid_contact_shape0'), ('shape1', 'rigid_contact_shape1'), ('point0', 'rigid_contact_point0'), ('point1', 'rigid_contact_point1'), ('normal', 'rigid_contact_normal'), ('cforce', 'force')]:
            setattr(contacts, field, self.outputs[k])
        contacts.rigid_contact_count = self.count
        contacts.n_contacts = self.count
        contacts.rigid_contact_max = len(self.outputs['shape0'])

    def parameter_audit(self):
        errors = self.param_errors.numpy()[self.n:].reshape(self.n, -1)
        out = []
        for w, f in zip(*np.nonzero(errors)):
            name, a, stride, start, count, ids, initial = self.parameter_sources[f]
            i = int(errors[w, f]) - 1
            out.append({'world': int(w), 'field': name, 'index': i, 'baked': float(initial[i]), 'current': float(a.numpy().reshape(count, stride)[w % count, i])})
        return out

    def diagnostic(self):
        return self.diag.numpy()

    def actual_state(self):
        return (self.backend.state_0.joint_q.numpy(), self.backend.state_0.joint_qd.numpy())
