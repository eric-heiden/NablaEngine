"""Assemble hand-written task kernels; specialize storage and immutable model values."""
from pathlib import Path
import re
ROOT=Path(__file__).parent
def physics_source(tag):
 src=(ROOT/'physics_core.cuh').read_text()
 src=src[:src.index('__global__ void get_diag')]
 src=src.replace('#include "newton_io.cuh"','')
 # Shared storage is reused after mass/bias assembly. Collision scratch is no longer live when rows are built.
 old=' float Jv[NB][NV][3],Jw[NB][NV][3],IW[NB][9],fb[NB][3],tb[NB][3];'
 rows=' float J[NR*NV],D[NR],aref[NR],fl[NR],force[NR];'
 src=src.replace(old,' union { struct {'+old+'}; struct {'+rows+'}; float collision_scratch['+'1'+']; };').replace('\n'+rows+'\n','\n')
 src=src.replace('struct Work {', 'struct Work {\n#if STAGE_PROFILE\n unsigned long long ticks[8];\n#endif')
 src=src.replace(' int failed_pairs[NP],resolved_pairs[NP];',' int failed_pairs[NP],resolved_pairs[NP];\n float params[NPARAM];')
 src=src.replace('__global__ void k_pre(Data dd,Work* all){\n int w=blockIdx.x,l=threadIdx.x;Work&s=all[w];\n __shared__ float pre_A[NV*NV],pre_rhs[NV];','__device__ void prepare(Data &dd,Work &s,int w,int l,float*pre_A,float*pre_rhs){')
 for x in ['D(qfrc_actuator,i)=actforce;','D(qfrc_bias,i)=bias;D(qfrc_smooth,i)=s.smooth[i];','D(qacc_smooth,i)=rhs[i];']:
  src=src.replace(x,'')
 src=src.replace('for(int b=0;b<NB;b++){for(int k=0;k<3;k++){D(xpos,b*3+k)=s.x[b][k];D(xipos,b*3+k)=s.com[b][k];}for(int k=0;k<4;k++)D(xquat,b*4+k)=s.q[b][k];}','')
 src=src.replace('struct Solve {float M[NV*NV],J[NR*NV],H[NV*NV]', 'struct Solve {float H[NV*NV]')
 src=src.replace('__global__ void k_post(Data dd,Work*all){\n int w=blockIdx.x,l=threadIdx.x;Work&s=all[w];__shared__ Solve v;int n=s.nr;\n for(int i=l;i<NV*NV;i+=32)v.M[i]=s.mass[i];for(int i=l;i<n*NV;i+=32)v.J[i]=s.J[i];__syncwarp();', '__device__ void solve_integrate(Data &dd,Work &s,Solve &v,int w,int l,IO io){\n int n=s.nr;')
 src=src.replace('v.M','s.mass').replace('v.J','s.J')
 a=src.index(' // Compact contacts after');b=src.index(' __syncwarp();\n if(l==0){',a)
 src=src[:a]+''' // Contact force and geometry are final outputs, written directly for Newton sensors.
 export_contacts(s,w,l,io);
'''+src[b:]
 src=src.replace('D(time,0)+=M(timestep,0);D(overflow,0)=s.unsupported;','')
 src=src.replace('D(qacc,i)=v.a[i];','')
 src=src.replace('  #if NEWTON_IO\n  if(dd.nw_qout){kin(s,w,dd);export_newton(s,w,dd);}\n  #endif','  kin(s,w,dd);export_state(s,dd,w,io);')
 src=src.replace('D(overflow,0)=s.unsupported;','')
 src=src.replace('__device__ void kin(Work &s,int w,Data dd)','__device__ void kin(Work &s,int w,Data &dd)')
 src=src.replace('row_add(Work&s,Data dd,','row_add(Work&s,Data &dd,').replace('constraints(Work&s,Data dd,','constraints(Work&s,Data &dd,')
 if tag=='lift':
  src=src.replace('if(b==0){copy3(s.x[b],pos);', 'if(b==0||b==13){if(b==13){for(int k=0;k<3;k++)pos[k]=dd.table_pos[k];for(int k=0;k<4;k++)quat[k]=dd.table_quat[k];}copy3(s.x[b],pos);')
 # Task-specific kin tree stays a fixed-size loop; all topology accessors are constexpr switches.
 src=src.replace('__device__ void prepare(', '#include "direct_io.cuh"\n__device__ void prepare(')
 # Optional clock instrumentation is compiled out of throughput and counter runs.
 src=src.replace('if(l==0){s.unsupported=0;s.nc=0;kin(s,w,dd);}__syncwarp();', 'if(l==0){s.unsupported=0;s.nc=0;kin(s,w,dd);MARK(2);}__syncwarp();')
 src=src.replace('collisions(s,w);constraints(s,dd,w);', 'MARK(3);collisions(s,w);MARK(4);constraints(s,dd,w);MARK(5);')
 src=src.replace(' // Contact force and geometry are final outputs', ' if(l==0){MARK(6); }\n // Contact force and geometry are final outputs')
 if tag=='go2':
  # All 27 allowed Go2 pairs are plane/robot. No convex solver is reachable.
  collision=(ROOT/'collision.cuh').read_text();a=collision.index('__device__ void collide_plane');b=collision.index('__device__ void collisions',a)
  src=src.replace('#include "collision.cuh"',collision[a:b]+'''\n__device__ void collisions(Work&s,int w){
 for(int pair=0;pair<NP;pair++)collide_plane(s,w,pair,0,PAIRS[2*pair+1]);
}\n''')
 if tag=='reach':
  collision=(ROOT/'collision.cuh').read_text();a=collision.index('__device__ void collide_plane');b=collision.index('__device__ void collisions',a)
  src=src.replace('#include "collision.cuh"',collision[a:b]+(ROOT/'reach_collision.cuh').read_text())
 if tag=='go2':
  src=src.replace('float Jv[NB][NV][3],Jw[NB][NV][3],IW[NB][9]', 'float Jv[NJAC][3],IW[NB][9]')
  src=src.replace('__device__ void prepare(', '__device__ int jslot(int b,int i){return J_START[b]+(i<6?i:6+i-J_FIRST[b]);}\n__device__ void prepare(')
  old='for(int t=l;t<NB*NV;t+=32){int b=t/NV,i=t%NV;for(int k=0;k<3;k++)s.Jw[b][i][k]=(ancestor(s,b,i,w)&&s.rot[i])?s.axis[i][k]:0;jacpoint(s,w,b,s.com[b],i,s.Jv[b][i]);}__syncwarp();'
  new='for(int t=l;t<NB*NV;t+=32){int b=t/NV,i=t%NV;if(ancestor(s,b,i,w))jacpoint(s,w,b,s.com[b],i,s.Jv[jslot(b,i)]);}__syncwarp();'
  assert old in src
  src=src.replace(old,new)
  src=src.replace('float Ij[3];inertia_mul(s.IW[b],s.Jw[b][j],Ij);v+=M(body_mass,b)*dot3(s.Jv[b][i],s.Jv[b][j])+dot3(s.Jw[b][i],Ij);', 'float Ij[3],Wi[3],Wj[3];for(int k=0;k<3;k++){Wi[k]=s.rot[i]?s.axis[i][k]:0;Wj[k]=s.rot[j]?s.axis[j][k]:0;}inertia_mul(s.IW[b],Wj,Ij);v+=M(body_mass,b)*dot3(s.Jv[jslot(b,i)],s.Jv[jslot(b,j)])+dot3(Wi,Ij);')
  src=src.replace('bias+=dot3(s.Jv[b][i],s.fb[b])+dot3(s.Jw[b][i],s.tb[b]);for(int k=0;k<3;k++)force+=s.Jv[b][i][k]*D(xfrc_applied,b*6+k)+s.Jw[b][i][k]*D(xfrc_applied,b*6+3+k);', 'float Wi[3];for(int k=0;k<3;k++)Wi[k]=s.rot[i]?s.axis[i][k]:0;bias+=dot3(s.Jv[jslot(b,i)],s.fb[b])+dot3(Wi,s.tb[b]);for(int k=0;k<3;k++)force+=s.Jv[jslot(b,i)][k]*D(xfrc_applied,b*6+k)+Wi[k]*D(xfrc_applied,b*6+3+k);')
 # The second FK needs only body poses and velocities. It does not rebuild inertia, bias, COM Jacobians, or collision transforms.
 src=src.replace('__device__ void kin(', 'template<bool DYNAMICS> __device__ void kin(')
 src=src.replace('  float ip[3],iq[4],Q[4],R[9];', '  if constexpr(DYNAMICS){float ip[3],iq[4],Q[4],R[9];',1)
 src=src.replace('crossadd(s.tb[b],s.om[b],c);\n }\n for(int g=0;', 'crossadd(s.tb[b],s.om[b],c);}\n }\n if constexpr(DYNAMICS) for(int g=0;',1)
 src=src.replace('kin(s,w,dd)', 'kin<true>(s,w,dd)').replace('kin<true>(s,w,dd);export_state', 'kin<false>(s,w,dd);export_state')
 from .parallelize import transform
 from .collision_parallel import transform as collision_transform
 from .line_search import transform as line_transform
 src=line_transform(collision_transform(transform(src,tag),tag))
 from .cooperative_solver import transform as solver_transform
 src=solver_transform(src)
 src=src.replace('struct Work {','struct Work {int output_base;')
 src=src.replace('int w,int l,IO io)', 'int w,int l,IO io,bool output)')
 src=src.replace('int w,int l,const IO&io)', 'int w,int l,const IO&io,bool output)')
 src=src.replace('export_contacts(s,w,l,io);','if(output)export_contacts(s,w,l,io);')
 src=src.replace('kin<false>(s,w,dd);export_state(s,dd,w,io);','if(output){kin<false>(s,w,dd);export_state(s,dd,w,io);}')
 if tag=='reach':
  src=src.replace('for(int k=0;k<4;k++)quat[k]=M(body_quat,b*4+k);','for(int k=0;k<4;k++)quat[k]=M(body_quat,b*4+k);if(b==10||b==11)for(int k=0;k<3;k++)pos[k]+=locked_finger_offset(3*(b-10)+k);')
 if tag=='lift':
  from .parallelize import block
  a=src.index('__device__ void collisions(');b,z=block(src,a)
  src=src[:a]+(ROOT/'analytic_proxy.cuh').read_text()+src[z:]
  src=src.replace('__device__ void add_contact(', '__device__ int reduced_type(Work&s,int g){int t=M(geom_type,g);return t==7?proxy_shape(M(geom_dataid,g)):t;}\n__device__ void add_contact(')
  src=src.replace('M(geom_type,PAIRS[2*pair])','reduced_type(s,PAIRS[2*pair])').replace('M(geom_type,PAIRS[2*pair+1])','reduced_type(s,PAIRS[2*pair+1])')
 if tag=='reach':src=src.replace('if(a==0){collide_plane(s,w,pair,a,b);continue;}','')
 from .scalar_gripper import transform as grip_transform
 return grip_transform(src,tag).replace('threadIdx.x','(threadIdx.x&31)')
