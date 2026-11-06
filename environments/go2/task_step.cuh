// Full-step CUDA research implementation. All dynamics and constraints execute here.
#include "primitive_math.cuh"
struct Contact {int g1,g2,row,dim;float dist,p[3],fr[9],mu;};
struct Work {int output_base;
#if STAGE_PROFILE
 unsigned long long ticks[8];
#endif
 float x[NB][3],q[NB][4],com[NB][3],v[NB][3],om[NB][3],ab[NB][3],al[NB][3];
 float jp[NJ][3],axis[NV][3],origin[NV][3];int db[NV],rot[NV];
 union { struct { float Jv[NJAC][3],IW[NB][9],fb[NB][3],tb[NB][3];}; struct { float J[NR*NV],D[NR],aref[NR],fl[NR],force[NR];}; float collision_scratch[1]; };
 float gp[NG][3],gq[NG][4];
 float mass[NV*NV],smooth[NV],bias[NV],qsm[NV],damp[NV];
 Contact con[NC];int nc,nr,ne,nf,unsupported,niter,ever_error;

 float params[NPARAM];
};
__device__ float clampf(float x,float a,float b){return fminf(fmaxf(x,a),b);}
__device__ void copy3(float *a,const float*b){for(int k=0;k<3;k++)a[k]=b[k];}
__device__ void crossadd(float *out,const float*a,const float*b,float scale=1){float c[3];cross(a,b,c);for(int k=0;k<3;k++)out[k]+=scale*c[k];}
__device__ void inertia_mul(const float *I,const float *a,float *out){for(int i=0;i<3;i++){out[i]=0;for(int j=0;j<3;j++)out[i]+=I[3*i+j]*a[j];}}
template<bool DYNAMICS> __device__ void kin(Work &s,int w,Data &dd){
 int lane=(threadIdx.x&31);
 for(int depth=0;depth<=TREE_DEPTH;depth++){int b=lane;if(b<NB && BODY_DEPTH[b]==depth){
  int parent=M(body_parentid,b),ja=M(body_jntadr,b),nj=M(body_jntnum,b),mc=M(body_mocapid,b);
  float pos[3],quat[4],r[3];for(int k=0;k<3;k++)pos[k]=M(body_pos,b*3+k);for(int k=0;k<4;k++)quat[k]=M(body_quat,b*4+k);
  for(int k=0;k<3;k++)s.v[b][k]=s.om[b][k]=s.ab[b][k]=s.al[b][k]=0;
  if(b==0){copy3(s.x[b],pos);for(int k=0;k<4;k++)s.q[b][k]=quat[k];}
  else{
   qrot(s.q[parent],pos,r);for(int k=0;k<3;k++){s.x[b][k]=s.x[parent][k]+r[k];s.v[b][k]=s.v[parent][k];s.om[b][k]=s.om[parent][k];s.ab[b][k]=s.ab[parent][k];s.al[b][k]=s.al[parent][k];}
   qmul(s.q[parent],quat,s.q[b]);crossadd(s.v[b],s.om[parent],r);crossadd(s.ab[b],s.al[parent],r);float c[3];cross(s.om[parent],r,c);crossadd(s.ab[b],s.om[parent],c);
  }
  if(mc>=0){for(int k=0;k<3;k++)s.x[b][k]=D(mocap_pos,mc*3+k);for(int k=0;k<4;k++)s.q[b][k]=D(mocap_quat,mc*4+k);}
  for(int j=ja;j<ja+nj;j++){
   int type=M(jnt_type,j),qa=M(jnt_qposadr,j),da=M(jnt_dofadr,j);
   float ax[3],anchor[3];for(int k=0;k<3;k++){ax[k]=M(jnt_axis,3*j+k);anchor[k]=M(jnt_pos,3*j+k);}float aw[3],pre[3];qrot(s.q[b],ax,aw);qrot(s.q[b],anchor,pre);
   for(int k=0;k<3;k++)s.jp[j][k]=s.x[b][k]+pre[k];
   if(type==0){
    for(int k=0;k<3;k++){s.x[b][k]=D(qpos,qa+k);s.v[b][k]=D(qvel,da+k);s.ab[b][k]=s.al[b][k]=0;}
    for(int k=0;k<4;k++)s.q[b][k]=D(qpos,qa+3+k);
    float local[3];for(int k=0;k<3;k++)local[k]=D(qvel,da+3+k);qrot(s.q[b],local,s.om[b]);
    for(int k=0;k<6;k++){
     s.db[da+k]=b;s.rot[da+k]=(k>=3);float unit[3]={0,0,0};unit[k%3]=1;
     if(k<3)copy3(s.axis[da+k],unit);else qrot(s.q[b],unit,s.axis[da+k]);copy3(s.origin[da+k],s.x[b]);
    }
   }else if(type==3){
    float theta=D(qpos,qa)-M(qpos0,qa),qd=D(qvel,da),dq[4]={cosf(.5f*theta),ax[0]*sinf(.5f*theta),ax[1]*sinf(.5f*theta),ax[2]*sinf(.5f*theta)},qn[4];qmul(s.q[b],dq,qn);
    float post[3],omold[3],alold[3];copy3(omold,s.om[b]);copy3(alold,s.al[b]);qrot(qn,anchor,post);
    crossadd(s.v[b],omold,pre);crossadd(s.ab[b],alold,pre);float c[3];cross(omold,pre,c);crossadd(s.ab[b],omold,c);
    crossadd(s.al[b],omold,aw,qd);for(int k=0;k<3;k++)s.om[b][k]+=aw[k]*qd;
    crossadd(s.v[b],s.om[b],post,-1);crossadd(s.ab[b],s.al[b],post,-1);cross(s.om[b],post,c);crossadd(s.ab[b],s.om[b],c,-1);
    for(int k=0;k<3;k++)s.x[b][k]+=pre[k]-post[k];for(int k=0;k<4;k++)s.q[b][k]=qn[k];
    s.db[da]=b;s.rot[da]=1;copy3(s.axis[da],aw);copy3(s.origin[da],s.jp[j]);
   }else if(type==2){
    float delta=D(qpos,qa)-M(qpos0,qa),vel=D(qvel,da),off[3];for(int k=0;k<3;k++)off[k]=aw[k]*delta;
    crossadd(s.v[b],s.om[b],off);crossadd(s.ab[b],s.al[b],off);float c[3];cross(s.om[b],off,c);crossadd(s.ab[b],s.om[b],c);crossadd(s.ab[b],s.om[b],aw,2*vel);
    for(int k=0;k<3;k++){s.x[b][k]+=off[k];s.v[b][k]+=aw[k]*vel;}
    s.db[da]=b;s.rot[da]=0;copy3(s.axis[da],aw);copy3(s.origin[da],s.jp[j]);
   }else s.unsupported|=1;
  }
}
 __syncwarp();}
 for(int b=lane;b<NB;b+=32){float r[3];  if constexpr(DYNAMICS){float ip[3],iq[4],Q[4],R[9];for(int k=0;k<3;k++)ip[k]=M(body_ipos,b*3+k);for(int k=0;k<4;k++)iq[k]=M(body_iquat,b*4+k);
  qrot(s.q[b],ip,r);for(int k=0;k<3;k++)s.com[b][k]=s.x[b][k]+r[k];qmul(s.q[b],iq,Q);
  for(int k=0;k<3;k++){float u[3]={0,0,0},v[3];u[k]=1;qrot(Q,u,v);for(int a=0;a<3;a++)R[a*3+k]=v[a];}
  for(int a=0;a<3;a++)for(int c=0;c<3;c++){float z=0;for(int k=0;k<3;k++)z+=R[a*3+k]*M(body_inertia,b*3+k)*R[c*3+k];s.IW[b][a*3+c]=z;}
  float acom[3];copy3(acom,s.ab[b]);crossadd(acom,s.al[b],r);float c[3];cross(s.om[b],r,c);crossadd(acom,s.om[b],c);
  for(int k=0;k<3;k++)s.fb[b][k]=M(body_mass,b)*(acom[k]-M(gravity,k));
  inertia_mul(s.IW[b],s.al[b],s.tb[b]);inertia_mul(s.IW[b],s.om[b],c);crossadd(s.tb[b],s.om[b],c);}
 }
 __syncwarp();

 if constexpr(DYNAMICS) for(int g=(threadIdx.x&31);g<NG;g+=32){int b=M(geom_bodyid,g);float p[3],q[4],r[3];for(int k=0;k<3;k++)p[k]=M(geom_pos,g*3+k);for(int k=0;k<4;k++)q[k]=M(geom_quat,g*4+k);qrot(s.q[b],p,r);for(int k=0;k<3;k++)s.gp[g][k]=s.x[b][k]+r[k];qmul(s.q[b],q,s.gq[g]);}

__syncwarp();
}
__device__ bool ancestor(Work&s,int b,int d,int w){return (DOF_BODIES[d]&(1u<<b))!=0;}
__device__ void jacpoint(Work&s,int w,int b,const float*p,int d,float*out){
 for(int k=0;k<3;k++)out[k]=0;if(!ancestor(s,b,d,w))return;
 if(!s.rot[d])copy3(out,s.axis[d]);else{float r[3];for(int k=0;k<3;k++)r[k]=p[k]-s.origin[d][k];cross(s.axis[d],r,out);}
}

// One register row per lane, specialized to this task's NV <= 32.
// Same Cholesky factorization as upstream G1, without its second row or 35-DOF tail.
__device__ __noinline__ void chol_parallel(float*A,float*rhs,int l){
 float row[NV],inverse[NV];
 #pragma unroll
 for(int j=0;j<NV;j++)row[j]=(l<NV&&j<=l)?A[l*NV+j]:0.f;
 __syncwarp();
 #pragma unroll
 for(int k=0;k<NV;k++){
  float diag=fmaxf(__shfl_sync(0xffffffff,row[k],k),1e-14f);
  float inv=rsqrtf(diag);inv*=1.5f-.5f*diag*inv*inv;inverse[k]=inv;
  if(l>k)row[k]*=inv;else if(l==k)row[k]=diag*inv;
  #pragma unroll
  for(int j=k+1;j<NV;j++)row[j]-=row[k]*__shfl_sync(0xffffffff,row[k],j);
 }
 #pragma unroll
 for(int j=0;j<NV;j++)if(l<NV&&j<=l)A[l*NV+j]=row[j];
 float y=l<NV?rhs[l]:0.f;
 #pragma unroll
 for(int k=0;k<NV;k++){
  float v=__shfl_sync(0xffffffff,y,k)*inverse[k];
  if(l==k)y=v;else if(l>k)y-=row[k]*v;
 }
 __syncwarp();
 #pragma unroll
 for(int k=NV-1;k>=0;k--){
  float v=__shfl_sync(0xffffffff,y,k)*inverse[k];
  if(l==k)y=v;else if(l<k)y-=A[k*NV+l]*v;
 }
 if(l<NV)rhs[l]=y;__syncwarp();
}


__device__ void kbi(const float* ref,const float* imp,float pos,float h,float& k,float& b,float& impedance){
 float lo=clampf(imp[0],.0001f,.9999f),hi=clampf(imp[1],.0001f,.9999f),width=fmaxf(imp[2],1e-15f),mid=clampf(imp[3],.0001f,.9999f),power=fmaxf(imp[4],1.f);
 float x=fabsf(pos)/width,y=x<mid ? powf(x,power)/powf(mid,power-1) : 1-powf(1-x,power)/powf(1-mid,power-1);
 impedance=x>=1?hi:lo+(hi-lo)*y;
 float tc=fmaxf(ref[0],2*h),ratio=ref[1];k=ref[0]>0?1/(hi*hi*tc*tc*ratio*ratio):-ref[0]/(hi*hi);b=ref[1]>0?2/(hi*tc):-ref[1]/hi;
}
__device__ void row_add(Work&s,Data &dd,int w,const float*j,float pos,float invw,const float*ref,const float*imp,float fl=0,int fixed_row=-1){
 int r=fixed_row>=0?fixed_row:s.nr++;if(r>=NR){atomicOr(&s.unsupported,2);return;}float vel=0;for(int i=0;i<NV;i++){s.J[r*NV+i]=j[i];vel+=j[i]*D(qvel,i);}float k,b,ip;kbi(ref,imp,pos,M(timestep,0),k,b,ip);s.D[r]=1/fmaxf(invw*(1-ip)/ip,1e-15f);s.aref[r]=-k*ip*pos-b*vel;s.fl[r]=fl;
}
__device__ void add_contact(Work&s,int pair,int slot,float dist,const float*pos,const float*frame){
 if(dist>=0)return;int id=atomicAdd(&s.nc,1);if(id>=NC){atomicOr(&s.unsupported,4);return;}Contact &c=s.con[id];c.row=pair;c.g1=PAIRS[2*pair];c.g2=PAIRS[2*pair+1];c.dist=dist;for(int k=0;k<3;k++)c.p[k]=pos[k];for(int k=0;k<9;k++)c.fr[k]=frame[k]*((M(geom_type,PAIRS[2*pair])>M(geom_type,PAIRS[2*pair+1])&&(k<3||k>=6))?-1:1);
}
#include "primitive_collision.cuh"
// Plane-only robot contacts. Every cylinder and box uses capsule support; feet stay spheres.
__device__ void collisions(Work&s,int w){
 int pair=(threadIdx.x&31);if(pair>=NP)return;int g=PAIRS[2*pair+1],type=M(geom_type,g);
 float normal[3],frame[9];zaxis_rn(s.gq[0],normal);make_frame(normal,frame);

 float radius=M(geom_size,3*g),half=0,axis[3];
 if(type==5){half=fmaxf(0,M(geom_size,3*g+1)-radius);zaxis_rn(s.gq[g],axis);}
 else if(type==6){float unit[3]={1,0,0};qrot(s.gq[g],unit,axis);radius=fmaxf(M(geom_size,3*g+1),M(geom_size,3*g+2));half=fmaxf(0,M(geom_size,3*g)-radius);}
 else{axis[0]=axis[1]=0;axis[2]=1;}
 for(int e=0;e<(half>0?2:1);e++){float end[3],point[3];capsule_end(s.gp[g],axis,half,e,end);float dist=plane_point(s.gp[0],normal,end,radius,point);add_contact(s,pair,e,dist,point,frame);}
}
__device__ void constraints(Work&s,Data &dd,int w){
 if((threadIdx.x&31)==0){
 s.nr=s.ne=s.nf=0;


 for(int a=0;a<NJ;a++)if(M(jnt_limited,a)){
  int qa=M(jnt_qposadr,a),da=M(jnt_dofadr,a);float low=D(qpos,qa)-M(jnt_range,2*a),high=M(jnt_range,2*a+1)-D(qpos,qa),margin=M(jnt_margin,a),distance=fminf(low,high);if(distance>=margin)continue;
  float j[NV]={0},ref[2],imp[5];j[da]=low<high?1:-1;for(int k=0;k<2;k++)ref[k]=M(jnt_solref,2*a+k);for(int k=0;k<5;k++)imp[k]=M(jnt_solimp,5*a+k);row_add(s,dd,w,j,distance-margin,M(dof_invweight0,da),ref,imp);
 }

 for(int cidx=0;cidx<s.nc;cidx++){Contact&c=s.con[cidx];int a=c.g1,b=c.g2,pa=M(geom_priority,a),pb=M(geom_priority,b);int dim=pa==pb?max(M(geom_condim,a),M(geom_condim,b)):M(geom_condim,pa>pb?a:b);c.row=s.nr;s.nr+=(dim==1?1:4);}
 if(s.nr>NR){s.unsupported|=2;s.nr=NR;}
 }__syncwarp();
 for(int cidx=(threadIdx.x&31);cidx<s.nc;cidx+=32){
  Contact &c=s.con[cidx];int a=c.g1,b=c.g2,ba=M(geom_bodyid,a),bb=M(geom_bodyid,b),pa=M(geom_priority,a),pb=M(geom_priority,b);float weight=M(geom_solmix,a)/(M(geom_solmix,a)+M(geom_solmix,b)+1e-30f);if(pa!=pb)weight=pa>pb?1:0;
  float mu=pa==pb?fmaxf(M(geom_friction,3*a),M(geom_friction,3*b)):M(geom_friction,3*(pa>pb?a:b));mu=fmaxf(mu,1e-5f);c.mu=mu;
  float ref[2],imp[5];for(int k=0;k<2;k++){float ra=M(geom_solref,2*a+k),rb=M(geom_solref,2*b+k);ref[k]=ra>0&&rb>0?weight*ra+(1-weight)*rb:fminf(ra,rb);}for(int k=0;k<5;k++)imp[k]=weight*M(geom_solimp,5*a+k)+(1-weight)*M(geom_solimp,5*b+k);
  int dim=pa==pb?max(M(geom_condim,a),M(geom_condim,b)):M(geom_condim,pa>pb?a:b);c.dim=dim;if(dim!=1&&dim!=3){s.unsupported|=16;continue;}
  float jt[3][NV];for(int i=0;i<NV;i++){float va[3],vb[3],dv[3];jacpoint(s,w,ba,c.p,i,va);jacpoint(s,w,bb,c.p,i,vb);for(int k=0;k<3;k++)dv[k]=vb[k]-va[k];for(int k=0;k<3;k++)jt[k][i]=dot3(c.fr+3*k,dv);}
  float iw=M(body_invweight0,ba*2)+M(body_invweight0,bb*2);if(dim==3)iw*=2*mu*mu*(1+mu*mu)/M(impratio,0);
  for(int edge=0;edge<(dim==1?1:4);edge++){float j[NV];for(int i=0;i<NV;i++)j[i]=jt[0][i]+(dim==1?0:(edge%2?-mu:mu)*jt[1+edge/2][i]);row_add(s,dd,w,j,c.dist,iw,ref,imp,0,c.row+edge);}
 }

__syncwarp();
}

#include "direct_io.cuh"
__device__ int jslot(int b,int i){return J_START[b]+(i<6?i:6+i-J_FIRST[b]);}
__device__ void prepare(Data &dd,Work &s,int w,int l,float*pre_A,float*pre_rhs){
 if(l==0){s.unsupported=0;s.nc=0;}__syncwarp();kin<true>(s,w,dd);if(l==0){MARK(2);}__syncwarp();
 for(int t=l;t<NB*NV;t+=32){int b=t/NV,i=t%NV;if(ancestor(s,b,i,w))jacpoint(s,w,b,s.com[b],i,s.Jv[jslot(b,i)]);}__syncwarp();
 for(int t=l;t<NV*NV;t+=32){int i=t/NV,j=t%NV;float v=i==j?M(dof_armature,i):0;unsigned int bodies=DOF_BODIES[i]&DOF_BODIES[j];while(bodies){int b=__ffs(bodies)-1;bodies&=bodies-1;float Ij[3],Wi[3],Wj[3];for(int k=0;k<3;k++){Wi[k]=s.rot[i]?s.axis[i][k]:0;Wj[k]=s.rot[j]?s.axis[j][k]:0;}inertia_mul(s.IW[b],Wj,Ij);v+=M(body_mass,b)*dot3(s.Jv[jslot(b,i)],s.Jv[jslot(b,j)])+dot3(Wi,Ij);}s.mass[t]=v;}
 for(int i=l;i<NV;i+=32){
  float bias=0,force=D(qfrc_applied,i)-M(dof_damping,i)*D(qvel,i),damp=M(dof_damping,i);
  unsigned int bodies=DOF_BODIES[i];while(bodies){int b=__ffs(bodies)-1;bodies&=bodies-1;float Wi[3];for(int k=0;k<3;k++)Wi[k]=s.rot[i]?s.axis[i][k]:0;bias+=dot3(s.Jv[jslot(b,i)],s.fb[b])+dot3(Wi,s.tb[b]);for(int k=0;k<3;k++)force+=s.Jv[jslot(b,i)][k]*D(xfrc_applied,b*6+k)+Wi[k]*D(xfrc_applied,b*6+3+k);}
  float actforce=0;for(int ai=0;ai<ACT_COUNT[i];ai++){int a=ACT_IDS[2*i+ai];int j=DOF_JOINT[i];float gear=M(actuator_gear,a*6),ctrl=D(ctrl,a);if(M(actuator_ctrllimited,a))ctrl=clampf(ctrl,M(actuator_ctrlrange,2*a),M(actuator_ctrlrange,2*a+1));float af=M(actuator_gainprm,10*a)*ctrl+M(actuator_biasprm,10*a)+M(actuator_biasprm,10*a+1)*gear*D(qpos,M(jnt_qposadr,j))+M(actuator_biasprm,10*a+2)*gear*D(qvel,i);if(M(actuator_forcelimited,a))af=clampf(af,M(actuator_forcerange,2*a),M(actuator_forcerange,2*a+1));actforce+=gear*af;damp-=gear*gear*M(actuator_biasprm,10*a+2);}
  {int j=DOF_JOINT[i];if(M(jnt_actfrclimited,j))actforce=clampf(actforce,M(jnt_actfrcrange,2*j),M(jnt_actfrcrange,2*j+1));}force+=actforce;
  s.bias[i]=bias;s.smooth[i]=force-bias;s.damp[i]=damp;
 }__syncwarp();
 for(int i=l;i<NV*NV;i+=32)pre_A[i]=s.mass[i];if(l<NV)pre_rhs[l]=s.smooth[l];__syncwarp();chol_parallel(pre_A,pre_rhs,l);if(l<NV)s.qsm[l]=pre_rhs[l];__syncwarp();if(l==0){MARK(3);}collisions(s,w);__syncwarp();if(l==0){s.nc=min(s.nc,NC);
 for(int i=1;i<s.nc;i++){Contact v=s.con[i];int j=i;while(j>0&&s.con[j-1].row>v.row){s.con[j]=s.con[j-1];j--;}s.con[j]=v;}MARK(4);}
 __syncwarp();constraints(s,dd,w);__syncwarp();if(l==0){MARK(5);}__syncwarp();
}
// Loop exits must be warp-uniform: later shuffles and __syncwarp() use the full mask.
// Optimized code may evaluate the same float expression differently per lane, so lane 0 decides.
__device__ __forceinline__ bool uniform(bool c){return __shfl_sync(0xffffffff,(int)c,0)!=0;}
__device__ float sumwarp(float v){for(int k=16;k;k/=2)v+=__shfl_down_sync(0xffffffff,v,k);return __shfl_sync(0xffffffff,v,0);}
__device__ void row_eval(float x,float d,float fl,int r,int ne,int nf,float&f,float&diag){f=diag=0;if(r<ne){f=-d*x;diag=d;}else if(r<ne+nf){f=clampf(-d*x,-fl,fl);if(fabsf(x)<fl/d)diag=d;}else if(x<0){f=-d*x;diag=d;}}
struct Solve {float H[NV*NV],a[NV],g[NV],p[NV],x[NR],jp[NR],f[NR],diag[NR],lo,hi,gd,pmp;int stop;};
__device__ void solve_integrate(Data &dd,Work &s,Solve &v,int w,int l,const IO&io,bool output){
 int n=s.nr;
 // Choose the lower objective between the previous acceleration and the smooth solution.
 float cold_cost=0,warm_cost=0;
 for(int i=l;i<NV;i+=32){float d=D(qacc_warmstart,i)-s.qsm[i],md=0;for(int j=0;j<NV;j++)md+=s.mass[i*NV+j]*(D(qacc_warmstart,j)-s.qsm[j]);warm_cost+=.5f*d*md;}
 for(int r=l;r<n;r+=32){float xc=-s.aref[r],xw=xc;for(int i=0;i<NV;i++){xc+=s.J[r*NV+i]*s.qsm[i];xw+=s.J[r*NV+i]*D(qacc_warmstart,i);}float xs[2]={xc,xw};for(int which=0;which<2;which++){float x=xs[which],cost=0;if(r<s.ne)cost=.5f*s.D[r]*x*x;else if(r<s.ne+s.nf){float bound=s.fl[r]/s.D[r];cost=fabsf(x)<bound?.5f*s.D[r]*x*x:s.fl[r]*(fabsf(x)-.5f*bound);}else if(x<0)cost=.5f*s.D[r]*x*x;if(which==0)cold_cost+=cost;else warm_cost+=cost;}}
 cold_cost=sumwarp(cold_cost);warm_cost=sumwarp(warm_cost);
 for(int i=l;i<NV;i+=32)v.a[i]=(warm_cost<cold_cost)?D(qacc_warmstart,i):s.qsm[i];__syncwarp();
 int nit=0;
 for(int it=0;it<ITERS && n>0;it++){
  for(int r=l;r<n;r+=32){float x=-s.aref[r];for(int i=0;i<NV;i++)x+=s.J[r*NV+i]*v.a[i];v.x[r]=x;row_eval(x,s.D[r],s.fl[r],r,s.ne,s.nf,v.f[r],v.diag[r]);}__syncwarp();
  float gg=0;for(int i=l;i<NV;i+=32){float g=-s.smooth[i];for(int j=0;j<NV;j++)g+=s.mass[i*NV+j]*v.a[j];for(int r=0;r<n;r++)g-=s.J[r*NV+i]*v.f[r];v.g[i]=g;gg+=g*g;}
  float norm=sqrtf(sumwarp(gg));if(uniform(norm<M(tolerance,0)*M(meaninertia,0)*NV))break;
  for(int t=l;t<NV*NV;t+=32){int i=t/NV,j=t%NV;if(j<=i){float h=s.mass[t];for(int r=0;r<n;r++)h+=s.J[r*NV+i]*v.diag[r]*s.J[r*NV+j];v.H[t]=h;}}__syncwarp();
  if(l<NV)v.p[l]=-v.g[l];__syncwarp();chol_parallel(v.H,v.p,l);
  if(l==0){float dec=0;for(int i=0;i<NV;i++)dec-=v.g[i]*v.p[i];v.stop=norm<M(tolerance,0)*M(meaninertia,0)*NV || .5f*dec<M(tolerance,0)*M(meaninertia,0)*NV;}__syncwarp();
  if(uniform(v.stop))break;
  for(int r=l;r<n;r+=32){float jp=0;for(int i=0;i<NV;i++)jp+=s.J[r*NV+i]*v.p[i];v.jp[r]=jp;}
  float gd=0,pmp=0;for(int i=l;i<NV;i+=32){float mp=0,ma=-s.smooth[i];for(int j=0;j<NV;j++){mp+=s.mass[i*NV+j]*v.p[j];ma+=s.mass[i*NV+j]*v.a[j];}gd+=ma*v.p[i];pmp+=mp*v.p[i];}gd=sumwarp(gd);pmp=sumwarp(pmp);if(l==0){v.gd=gd;v.pmp=pmp;v.lo=0;v.hi=1;}__syncwarp();
  for(int k=0;k<16;k++){float df=0;for(int r=l;r<n;r+=32){float f,d;row_eval(v.x[r]+v.hi*v.jp[r],s.D[r],s.fl[r],r,s.ne,s.nf,f,d);df-=f*v.jp[r];}df=sumwarp(df)+v.gd+v.hi*v.pmp;if(uniform(df>=0))break;if(l==0){v.lo=v.hi;v.hi*=2;}__syncwarp();}
  float alpha=.5f*(v.lo+v.hi);
  for(int k=0;k<24;k++){
   float df=0,curv=0;for(int r=l;r<n;r+=32){float f,d;row_eval(v.x[r]+alpha*v.jp[r],s.D[r],s.fl[r],r,s.ne,s.nf,f,d);df-=f*v.jp[r];curv+=d*v.jp[r]*v.jp[r];}
   df=sumwarp(df)+v.gd+alpha*v.pmp;curv=sumwarp(curv)+v.pmp;
   if(uniform(fabsf(df)<=1e-7f*(fabsf(v.gd)+fabsf(alpha*v.pmp)+1.f)))break;
   if(l==0){if(df<0)v.lo=alpha;else v.hi=alpha;}__syncwarp();
   float next=alpha-df/fmaxf(curv,1e-30f);
   if(!(next>v.lo&&next<v.hi))next=.5f*(v.lo+v.hi);
   next=__shfl_sync(0xffffffff,next,0);if(next==alpha)break;alpha=next;
  }
  for(int i=l;i<NV;i+=32)v.a[i]+=alpha*v.p[i];__syncwarp();nit=it+1;
 }
 for(int r=l;r<n;r+=32){float x=-s.aref[r],dg;for(int i=0;i<NV;i++)x+=s.J[r*NV+i]*v.a[i];row_eval(x,s.D[r],s.fl[r],r,s.ne,s.nf,s.force[r],dg);}__syncwarp();
 if(l==0){MARK(6); }
 // Contact force and geometry are final outputs, written directly for Newton sensors.
 if(output)export_contacts(s,w,l,io);
 __syncwarp();

 float *rhs=v.p,*A=v.H;float h=M(timestep,0);
 if(l==0)s.niter=nit;
 for(int i=l;i<NV*NV;i+=32)A[i]=s.mass[i];__syncwarp();
 if(l<NV){D(qacc_warmstart,l)=v.a[l];float z=0;for(int j=0;j<NV;j++)z+=s.mass[l*NV+j]*v.a[j];rhs[l]=z;A[l*NV+l]+=h*s.damp[l];}__syncwarp();
 chol_parallel(A,rhs,l);if(l<NV)D(qvel,l)+=h*rhs[l];__syncwarp();
   for(int j=l;j<NJ;j+=32){int qa=M(jnt_qposadr,j),da=M(jnt_dofadr,j),type=M(jnt_type,j);if(type==0){for(int k=0;k<3;k++)D(qpos,qa+k)+=h*D(qvel,da+k);float vv[3];for(int k=0;k<3;k++)vv[k]=D(qvel,da+3+k);float speed=sqrtf(dot3(vv,vv)),angle=h*speed,sc=speed>1e-10f?sinf(.5f*angle)/speed:.5f*h;float dq[4]={cosf(.5f*angle),vv[0]*sc,vv[1]*sc,vv[2]*sc},q[4],out[4];for(int k=0;k<4;k++)q[k]=D(qpos,qa+3+k);qmul(q,dq,out);float norm=0;for(int k=0;k<4;k++)norm+=out[k]*out[k];for(int k=0;k<4;k++)D(qpos,qa+3+k)=out[k]*rsqrtf(norm);}else if(type==2||type==3)D(qpos,qa)+=h*D(qvel,da);}
 __syncwarp();if(output){kin<false>(s,w,dd);export_state(s,dd,w,io);}__syncwarp();
 if(l==0){for(int i=0;i<NV;i++)if(!isfinite(D(qvel,i)))s.unsupported|=1024;for(int i=0;i<NQ;i++)if(!isfinite(D(qpos,i)))s.unsupported|=1024;s.ever_error|=s.unsupported;}

}
