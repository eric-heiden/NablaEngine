// Full-step CUDA research implementation. All dynamics and constraints execute here.
#include "primitive_math.cuh"
struct Contact {int g1,g2,row,dim;float dist,p[3],fr[9],mu;};
struct Work {
 float x[NB][3],q[NB][4],com[NB][3],v[NB][3],om[NB][3],ab[NB][3],al[NB][3];
 float jp[NJ][3],axis[NV][3],origin[NV][3];int db[NV],rot[NV];
 float Jv[NB][NV][3],Jw[NB][NV][3],IW[NB][9],fb[NB][3],tb[NB][3];
 float gp[NG][3],gq[NG][4];
 float mass[NV*NV],smooth[NV],bias[NV],qsm[NV],damp[NV];
 float J[NR*NV],D[NR],aref[NR],fl[NR],force[NR];
 Contact con[NC];int nc,nr,ne,nf,unsupported,niter,ever_error;
 int failed_pairs[NP],resolved_pairs[NP];
};
__device__ float clampf(float x,float a,float b){return fminf(fmaxf(x,a),b);}
__device__ void copy3(float *a,const float*b){for(int k=0;k<3;k++)a[k]=b[k];}
__device__ void crossadd(float *out,const float*a,const float*b,float scale=1){float c[3];cross(a,b,c);for(int k=0;k<3;k++)out[k]+=scale*c[k];}
__device__ void inertia_mul(const float *I,const float *a,float *out){for(int i=0;i<3;i++){out[i]=0;for(int j=0;j<3;j++)out[i]+=I[3*i+j]*a[j];}}
__device__ void kin(Work &s,int w,Data dd){
 for(int b=0;b<NB;b++){
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
  float ip[3],iq[4],Q[4],R[9];for(int k=0;k<3;k++)ip[k]=M(body_ipos,b*3+k);for(int k=0;k<4;k++)iq[k]=M(body_iquat,b*4+k);
  qrot(s.q[b],ip,r);for(int k=0;k<3;k++)s.com[b][k]=s.x[b][k]+r[k];qmul(s.q[b],iq,Q);
  for(int k=0;k<3;k++){float u[3]={0,0,0},v[3];u[k]=1;qrot(Q,u,v);for(int a=0;a<3;a++)R[a*3+k]=v[a];}
  for(int a=0;a<3;a++)for(int c=0;c<3;c++){float z=0;for(int k=0;k<3;k++)z+=R[a*3+k]*M(body_inertia,b*3+k)*R[c*3+k];s.IW[b][a*3+c]=z;}
  float acom[3];copy3(acom,s.ab[b]);crossadd(acom,s.al[b],r);float c[3];cross(s.om[b],r,c);crossadd(acom,s.om[b],c);
  for(int k=0;k<3;k++)s.fb[b][k]=M(body_mass,b)*(acom[k]-M(gravity,k));
  inertia_mul(s.IW[b],s.al[b],s.tb[b]);inertia_mul(s.IW[b],s.om[b],c);crossadd(s.tb[b],s.om[b],c);
 }
 for(int g=0;g<NG;g++){int b=M(geom_bodyid,g);float p[3],q[4],r[3];for(int k=0;k<3;k++)p[k]=M(geom_pos,g*3+k);for(int k=0;k<4;k++)q[k]=M(geom_quat,g*4+k);qrot(s.q[b],p,r);for(int k=0;k<3;k++)s.gp[g][k]=s.x[b][k]+r[k];qmul(s.q[b],q,s.gq[g]);}
}
__device__ bool ancestor(Work&s,int b,int d,int w){return (DOF_BODIES[d]&(1u<<b))!=0;}
__device__ void jacpoint(Work&s,int w,int b,const float*p,int d,float*out){
 for(int k=0;k<3;k++)out[k]=0;if(!ancestor(s,b,d,w))return;
 if(!s.rot[d])copy3(out,s.axis[d]);else{float r[3];for(int k=0;k<3;k++)r[k]=p[k]-s.origin[d][k];cross(s.axis[d],r,out);}
}
__device__ void cholesky(float* A,float* rhs){
 for(int j=0;j<NV;j++){
  float z=A[j*NV+j];for(int k=0;k<j;k++)z-=A[j*NV+k]*A[j*NV+k];A[j*NV+j]=sqrtf(fmaxf(z,1e-14f));
  for(int i=j+1;i<NV;i++){float v=A[i*NV+j];for(int k=0;k<j;k++)v-=A[i*NV+k]*A[j*NV+k];A[i*NV+j]=v/A[j*NV+j];}
 }
 for(int i=0;i<NV;i++){float v=rhs[i];for(int j=0;j<i;j++)v-=A[i*NV+j]*rhs[j];rhs[i]=v/A[i*NV+i];}
 for(int i=NV-1;i>=0;i--){float v=rhs[i];for(int j=i+1;j<NV;j++)v-=A[j*NV+i]*rhs[j];rhs[i]=v/A[i*NV+i];}
}
__device__ void kbi(const float* ref,const float* imp,float pos,float h,float& k,float& b,float& impedance){
 float lo=clampf(imp[0],.0001f,.9999f),hi=clampf(imp[1],.0001f,.9999f),width=fmaxf(imp[2],1e-15f),mid=clampf(imp[3],.0001f,.9999f),power=fmaxf(imp[4],1.f);
 float x=fabsf(pos)/width,y=x<mid ? powf(x,power)/powf(mid,power-1) : 1-powf(1-x,power)/powf(1-mid,power-1);
 impedance=x>=1?hi:lo+(hi-lo)*y;
 float tc=fmaxf(ref[0],2*h),ratio=ref[1];k=ref[0]>0?1/(hi*hi*tc*tc*ratio*ratio):-ref[0]/(hi*hi);b=ref[1]>0?2/(hi*tc):-ref[1]/hi;
}
__device__ void row_add(Work&s,Data dd,int w,const float*j,float pos,float invw,const float*ref,const float*imp,float fl=0){
 if(s.nr>=NR){s.unsupported|=2;return;}int r=s.nr++;float vel=0;for(int i=0;i<NV;i++){s.J[r*NV+i]=j[i];vel+=j[i]*D(qvel,i);}float k,b,ip;kbi(ref,imp,pos,M(timestep,0),k,b,ip);s.D[r]=1/fmaxf(invw*(1-ip)/ip,1e-15f);s.aref[r]=-k*ip*pos-b*vel;s.fl[r]=fl;
}
__device__ void add_contact(Work&s,int pair,int slot,float dist,const float*pos,const float*frame){
 if(dist>=0)return;if(s.nc>=NC){s.unsupported|=4;return;}Contact &c=s.con[s.nc++];c.g1=PAIRS[2*pair];c.g2=PAIRS[2*pair+1];c.dist=dist;for(int k=0;k<3;k++)c.p[k]=pos[k];for(int k=0;k<9;k++)c.fr[k]=frame[k];
}
#include "primitive_collision.cuh"
#include "collision.cuh"
__device__ void constraints(Work&s,Data dd,int w){
 s.nr=s.ne=s.nf=0;
 for(int e=0;e<NEQ;e++){
  if(M(eq_type,e)!=2){s.unsupported|=8;continue;}int a=M(eq_obj1id,e),b=M(eq_obj2id,e),qa=M(jnt_qposadr,a),da=M(jnt_dofadr,a);float x=b>=0?D(qpos,M(jnt_qposadr,b))-M(qpos0,M(jnt_qposadr,b)):0;
  float poly=0,deriv=0,pow=1;for(int k=0;k<5;k++){poly+=M(eq_data,e*11+k)*pow;pow*=x;}pow=1;for(int k=1;k<5;k++){deriv+=k*M(eq_data,e*11+k)*pow;pow*=x;}
  float j[NV]={0},ref[2],imp[5];j[da]=1;float iw=M(dof_invweight0,da);if(b>=0){int db=M(jnt_dofadr,b);j[db]=-deriv;iw+=M(dof_invweight0,db);}for(int k=0;k<2;k++)ref[k]=M(eq_solref,e*2+k);for(int k=0;k<5;k++)imp[k]=M(eq_solimp,e*5+k);row_add(s,dd,w,j,D(qpos,qa)-M(qpos0,qa)-poly,iw,ref,imp);s.ne++;
 }
 for(int i=0;i<NV;i++)if(M(dof_frictionloss,i)>0){float j[NV]={0},ref[2],imp[5];j[i]=1;for(int k=0;k<2;k++)ref[k]=M(dof_solref,2*i+k);for(int k=0;k<5;k++)imp[k]=M(dof_solimp,5*i+k);row_add(s,dd,w,j,0,M(dof_invweight0,i),ref,imp,M(dof_frictionloss,i));s.nf++;}
 for(int a=0;a<NJ;a++)if(M(jnt_limited,a)){
  int qa=M(jnt_qposadr,a),da=M(jnt_dofadr,a);float low=D(qpos,qa)-M(jnt_range,2*a),high=M(jnt_range,2*a+1)-D(qpos,qa),margin=M(jnt_margin,a),distance=fminf(low,high);if(distance>=margin)continue;
  float j[NV]={0},ref[2],imp[5];j[da]=low<high?1:-1;for(int k=0;k<2;k++)ref[k]=M(jnt_solref,2*a+k);for(int k=0;k<5;k++)imp[k]=M(jnt_solimp,5*a+k);row_add(s,dd,w,j,distance-margin,M(dof_invweight0,da),ref,imp);
 }
 for(int cidx=0;cidx<s.nc;cidx++){
  Contact &c=s.con[cidx];int a=c.g1,b=c.g2,ba=M(geom_bodyid,a),bb=M(geom_bodyid,b),pa=M(geom_priority,a),pb=M(geom_priority,b);float weight=M(geom_solmix,a)/(M(geom_solmix,a)+M(geom_solmix,b)+1e-30f);if(pa!=pb)weight=pa>pb?1:0;
  float mu=pa==pb?fmaxf(M(geom_friction,3*a),M(geom_friction,3*b)):M(geom_friction,3*(pa>pb?a:b));mu=fmaxf(mu,1e-5f);c.mu=mu;
  float ref[2],imp[5];for(int k=0;k<2;k++){float ra=M(geom_solref,2*a+k),rb=M(geom_solref,2*b+k);ref[k]=ra>0&&rb>0?weight*ra+(1-weight)*rb:fminf(ra,rb);}for(int k=0;k<5;k++)imp[k]=weight*M(geom_solimp,5*a+k)+(1-weight)*M(geom_solimp,5*b+k);
  int dim=pa==pb?max(M(geom_condim,a),M(geom_condim,b)):M(geom_condim,pa>pb?a:b);c.dim=dim;c.row=s.nr;if(dim!=1&&dim!=3){s.unsupported|=16;continue;}
  float jt[3][NV];for(int i=0;i<NV;i++){float va[3],vb[3],dv[3];jacpoint(s,w,ba,c.p,i,va);jacpoint(s,w,bb,c.p,i,vb);for(int k=0;k<3;k++)dv[k]=vb[k]-va[k];for(int k=0;k<3;k++)jt[k][i]=dot3(c.fr+3*k,dv);}
  float iw=M(body_invweight0,ba*2)+M(body_invweight0,bb*2);if(dim==3)iw*=2*mu*mu*(1+mu*mu)/M(impratio,0);
  for(int edge=0;edge<(dim==1?1:4);edge++){float j[NV];for(int i=0;i<NV;i++)j[i]=jt[0][i]+(dim==1?0:(edge%2?-mu:mu)*jt[1+edge/2][i]);row_add(s,dd,w,j,c.dist,iw,ref,imp);}
 }
}
#include "newton_io.cuh"
__global__ void k_pre(Data dd,Work* all){
 int w=blockIdx.x,l=threadIdx.x;Work&s=all[w];
 __shared__ float pre_A[NV*NV],pre_rhs[NV];
 if(l==0){s.unsupported=0;s.nc=0;kin(s,w,dd);}__syncwarp();
 for(int t=l;t<NB*NV;t+=32){int b=t/NV,i=t%NV;for(int k=0;k<3;k++)s.Jw[b][i][k]=(ancestor(s,b,i,w)&&s.rot[i])?s.axis[i][k]:0;jacpoint(s,w,b,s.com[b],i,s.Jv[b][i]);}__syncwarp();
 for(int t=l;t<NV*NV;t+=32){int i=t/NV,j=t%NV;float v=i==j?M(dof_armature,i):0;unsigned int bodies=DOF_BODIES[i]&DOF_BODIES[j];while(bodies){int b=__ffs(bodies)-1;bodies&=bodies-1;float Ij[3];inertia_mul(s.IW[b],s.Jw[b][j],Ij);v+=M(body_mass,b)*dot3(s.Jv[b][i],s.Jv[b][j])+dot3(s.Jw[b][i],Ij);}s.mass[t]=v;}
 for(int i=l;i<NV;i+=32){
  float bias=0,force=D(qfrc_applied,i)-M(dof_damping,i)*D(qvel,i),damp=M(dof_damping,i);
  unsigned int bodies=DOF_BODIES[i];while(bodies){int b=__ffs(bodies)-1;bodies&=bodies-1;bias+=dot3(s.Jv[b][i],s.fb[b])+dot3(s.Jw[b][i],s.tb[b]);for(int k=0;k<3;k++)force+=s.Jv[b][i][k]*D(xfrc_applied,b*6+k)+s.Jw[b][i][k]*D(xfrc_applied,b*6+3+k);}
  float actforce=0;for(int a=0;a<NU;a++){int j=M(actuator_trnid,2*a);if(M(jnt_dofadr,j)!=i)continue;float gear=M(actuator_gear,a*6),ctrl=D(ctrl,a);if(M(actuator_ctrllimited,a))ctrl=clampf(ctrl,M(actuator_ctrlrange,2*a),M(actuator_ctrlrange,2*a+1));float af=M(actuator_gainprm,10*a)*ctrl+M(actuator_biasprm,10*a)+M(actuator_biasprm,10*a+1)*gear*D(qpos,M(jnt_qposadr,j))+M(actuator_biasprm,10*a+2)*gear*D(qvel,i);if(M(actuator_forcelimited,a))af=clampf(af,M(actuator_forcerange,2*a),M(actuator_forcerange,2*a+1));actforce+=gear*af;damp-=gear*gear*M(actuator_biasprm,10*a+2);}
  for(int j=0;j<NJ;j++)if(M(jnt_dofadr,j)==i && M(jnt_actfrclimited,j))actforce=clampf(actforce,M(jnt_actfrcrange,2*j),M(jnt_actfrcrange,2*j+1));force+=actforce;D(qfrc_actuator,i)=actforce;
  s.bias[i]=bias;s.smooth[i]=force-bias;s.damp[i]=damp;D(qfrc_bias,i)=bias;D(qfrc_smooth,i)=s.smooth[i];
 }__syncwarp();
 if(l==0){float *A=pre_A,*rhs=pre_rhs;for(int i=0;i<NV*NV;i++)A[i]=s.mass[i];for(int i=0;i<NV;i++)rhs[i]=s.smooth[i];cholesky(A,rhs);for(int i=0;i<NV;i++){s.qsm[i]=rhs[i];D(qacc_smooth,i)=rhs[i];}collisions(s,w);constraints(s,dd,w);for(int b=0;b<NB;b++){for(int k=0;k<3;k++){D(xpos,b*3+k)=s.x[b][k];D(xipos,b*3+k)=s.com[b][k];}for(int k=0;k<4;k++)D(xquat,b*4+k)=s.q[b][k];}}
}
__device__ float sumwarp(float v){for(int k=16;k;k/=2)v+=__shfl_down_sync(0xffffffff,v,k);return __shfl_sync(0xffffffff,v,0);}
__device__ void row_eval(float x,float d,float fl,int r,int ne,int nf,float&f,float&diag){f=diag=0;if(r<ne){f=-d*x;diag=d;}else if(r<ne+nf){f=clampf(-d*x,-fl,fl);if(fabsf(x)<fl/d)diag=d;}else if(x<0){f=-d*x;diag=d;}}
struct Solve {float M[NV*NV],J[NR*NV],H[NV*NV],a[NV],g[NV],p[NV],x[NR],jp[NR],f[NR],diag[NR],lo,hi,gd,pmp;int stop;};
__global__ void k_post(Data dd,Work*all){
 int w=blockIdx.x,l=threadIdx.x;Work&s=all[w];__shared__ Solve v;int n=s.nr;
 for(int i=l;i<NV*NV;i+=32)v.M[i]=s.mass[i];for(int i=l;i<n*NV;i+=32)v.J[i]=s.J[i];__syncwarp();
 // Choose the lower objective between the previous acceleration and the smooth solution.
 float cold_cost=0,warm_cost=0;
 for(int i=l;i<NV;i+=32){float d=D(qacc_warmstart,i)-s.qsm[i],md=0;for(int j=0;j<NV;j++)md+=v.M[i*NV+j]*(D(qacc_warmstart,j)-s.qsm[j]);warm_cost+=.5f*d*md;}
 for(int r=l;r<n;r+=32){float xc=-s.aref[r],xw=xc;for(int i=0;i<NV;i++){xc+=v.J[r*NV+i]*s.qsm[i];xw+=v.J[r*NV+i]*D(qacc_warmstart,i);}float xs[2]={xc,xw};for(int which=0;which<2;which++){float x=xs[which],cost=0;if(r<s.ne)cost=.5f*s.D[r]*x*x;else if(r<s.ne+s.nf){float bound=s.fl[r]/s.D[r];cost=fabsf(x)<bound?.5f*s.D[r]*x*x:s.fl[r]*(fabsf(x)-.5f*bound);}else if(x<0)cost=.5f*s.D[r]*x*x;if(which==0)cold_cost+=cost;else warm_cost+=cost;}}
 cold_cost=sumwarp(cold_cost);warm_cost=sumwarp(warm_cost);
 for(int i=l;i<NV;i+=32)v.a[i]=(warm_cost<cold_cost)?D(qacc_warmstart,i):s.qsm[i];__syncwarp();
 int nit=0;
 for(int it=0;it<ITERS;it++){
  for(int r=l;r<n;r+=32){float x=-s.aref[r];for(int i=0;i<NV;i++)x+=v.J[r*NV+i]*v.a[i];v.x[r]=x;row_eval(x,s.D[r],s.fl[r],r,s.ne,s.nf,v.f[r],v.diag[r]);}__syncwarp();
  float gg=0;for(int i=l;i<NV;i+=32){float g=-s.smooth[i];for(int j=0;j<NV;j++)g+=v.M[i*NV+j]*v.a[j];for(int r=0;r<n;r++)g-=v.J[r*NV+i]*v.f[r];v.g[i]=g;gg+=g*g;}
  float norm=sqrtf(sumwarp(gg));
  for(int t=l;t<NV*NV;t+=32){int i=t/NV,j=t%NV;float h=v.M[t];for(int r=0;r<n;r++)h+=v.J[r*NV+i]*v.diag[r]*v.J[r*NV+j];v.H[t]=h;}__syncwarp();
  for(int j=0;j<NV;j++){if(l==0){float z=v.H[j*NV+j];for(int k=0;k<j;k++)z-=v.H[j*NV+k]*v.H[j*NV+k];v.H[j*NV+j]=sqrtf(fmaxf(z,1e-14f));}__syncwarp();for(int i=j+1+l;i<NV;i+=32){float z=v.H[i*NV+j];for(int k=0;k<j;k++)z-=v.H[i*NV+k]*v.H[j*NV+k];v.H[i*NV+j]=z/v.H[j*NV+j];}__syncwarp();}
  if(l==0){for(int i=0;i<NV;i++){float z=-v.g[i];for(int j=0;j<i;j++)z-=v.H[i*NV+j]*v.p[j];v.p[i]=z/v.H[i*NV+i];}for(int i=NV-1;i>=0;i--){float z=v.p[i];for(int j=i+1;j<NV;j++)z-=v.H[j*NV+i]*v.p[j];v.p[i]=z/v.H[i*NV+i];}float dec=0;for(int i=0;i<NV;i++)dec-=v.g[i]*v.p[i];v.stop=norm<M(tolerance,0)*M(meaninertia,0)*NV || .5f*dec<M(tolerance,0)*M(meaninertia,0)*NV;}__syncwarp();
  if(it>0&&v.stop)break;
  for(int r=l;r<n;r+=32){float jp=0;for(int i=0;i<NV;i++)jp+=v.J[r*NV+i]*v.p[i];v.jp[r]=jp;}
  float gd=0,pmp=0;for(int i=l;i<NV;i+=32){float mp=0,ma=-s.smooth[i];for(int j=0;j<NV;j++){mp+=v.M[i*NV+j]*v.p[j];ma+=v.M[i*NV+j]*v.a[j];}gd+=ma*v.p[i];pmp+=mp*v.p[i];}gd=sumwarp(gd);pmp=sumwarp(pmp);if(l==0){v.gd=gd;v.pmp=pmp;v.lo=0;v.hi=1;}__syncwarp();
  for(int k=0;k<16;k++){float df=0;for(int r=l;r<n;r+=32){float f,d;row_eval(v.x[r]+v.hi*v.jp[r],s.D[r],s.fl[r],r,s.ne,s.nf,f,d);df-=f*v.jp[r];}df=sumwarp(df)+v.gd+v.hi*v.pmp;if(df>=0)break;if(l==0){v.lo=v.hi;v.hi*=2;}__syncwarp();}
  for(int k=0;k<24;k++){float alpha=.5f*(v.lo+v.hi),df=0;for(int r=l;r<n;r+=32){float f,d;row_eval(v.x[r]+alpha*v.jp[r],s.D[r],s.fl[r],r,s.ne,s.nf,f,d);df-=f*v.jp[r];}df=sumwarp(df)+v.gd+alpha*v.pmp;if(l==0){if(df<0)v.lo=alpha;else v.hi=alpha;}__syncwarp();}
  for(int i=l;i<NV;i+=32)v.a[i]+=.5f*(v.lo+v.hi)*v.p[i];__syncwarp();nit=it+1;
 }
 for(int r=l;r<n;r+=32){float x=-s.aref[r],dg;for(int i=0;i<NV;i++)x+=v.J[r*NV+i]*v.a[i];row_eval(x,s.D[r],s.fl[r],r,s.ne,s.nf,s.force[r],dg);}__syncwarp();
 // Compact contacts after every world's collision phase has completed.
 int prefix=0;for(int k=l;k<w;k+=32)prefix+=all[k].nc;prefix=(int)sumwarp((float)prefix);
 if(l==0){
  D(nefc,0)=s.nr;D(ne,0)=s.ne;D(nf,0)=s.nf;
  if(w==gridDim.x-1)dd.nacon[0]=min(prefix+s.nc,NATIVE_NC);
  if(s.nr>NATIVE_NR||prefix+s.nc>NATIVE_NC)s.unsupported|=512;
 }
 for(int r=l;r<min(s.nr,NATIVE_NR);r+=32)D(efc_force,r)=s.force[r];
 for(int ci=l;ci<s.nc;ci+=32){int id=prefix+ci;if(id>=NATIVE_NC)continue;Contact&c=s.con[ci];
  dd.con_worldid[id]=w;dd.con_dist[id]=c.dist;dd.con_dim[id]=c.dim;dd.con_geom[id*2]=c.g1;dd.con_geom[id*2+1]=c.g2;
  for(int k=0;k<3;k++)dd.con_pos[id*3+k]=c.p[k];for(int k=0;k<9;k++)dd.con_frame[id*9+k]=c.fr[k];
  for(int k=0;k<5;k++)dd.con_friction[id*5+k]=k<2?c.mu:0;
  for(int k=0;k<DS_con_efc_address;k++)dd.con_efc_address[id*DS_con_efc_address+k]=k==0?c.row:-1;
  dd.con_adhesion[id]=0;
 }
 __syncwarp();
 if(l==0){
  D(time,0)+=M(timestep,0);D(overflow,0)=s.unsupported;
  s.niter=nit;float *rhs=v.p,*A=v.H;float h=M(timestep,0);for(int i=0;i<NV*NV;i++)A[i]=v.M[i];
  for(int i=0;i<NV;i++){D(qacc,i)=v.a[i];D(qacc_warmstart,i)=v.a[i];rhs[i]=0;for(int j=0;j<NV;j++)rhs[i]+=v.M[i*NV+j]*v.a[j];A[i*NV+i]+=h*s.damp[i];}cholesky(A,rhs);
  for(int i=0;i<NV;i++)D(qvel,i)+=h*rhs[i];
  for(int j=0;j<NJ;j++){int qa=M(jnt_qposadr,j),da=M(jnt_dofadr,j),type=M(jnt_type,j);if(type==0){for(int k=0;k<3;k++)D(qpos,qa+k)+=h*D(qvel,da+k);float vv[3];for(int k=0;k<3;k++)vv[k]=D(qvel,da+3+k);float speed=sqrtf(dot3(vv,vv)),angle=h*speed,sc=speed>1e-10f?sinf(.5f*angle)/speed:.5f*h;float dq[4]={cosf(.5f*angle),vv[0]*sc,vv[1]*sc,vv[2]*sc},q[4],out[4];for(int k=0;k<4;k++)q[k]=D(qpos,qa+3+k);qmul(q,dq,out);float norm=0;for(int k=0;k<4;k++)norm+=out[k]*out[k];for(int k=0;k<4;k++)D(qpos,qa+3+k)=out[k]*rsqrtf(norm);}else if(type==2||type==3)D(qpos,qa)+=h*D(qvel,da);}
  #if NEWTON_IO
  if(dd.nw_qout){kin(s,w,dd);export_newton(s,w,dd);}
  #endif
  for(int i=0;i<NV;i++)if(!isfinite(D(qvel,i)))s.unsupported|=1024;for(int i=0;i<NQ;i++)if(!isfinite(D(qpos,i)))s.unsupported|=1024;s.ever_error|=s.unsupported;D(overflow,0)=s.unsupported;
 }
}
__global__ void get_diag(Work*s,float*out){int w=blockIdx.x;int stride=4+2*NV+NV*NV;float*p=out+w*stride;p[0]=s[w].nc;p[1]=s[w].nr;p[2]=s[w].ever_error;p[3]=s[w].niter;for(int i=0;i<NV;i++){p[4+i]=s[w].bias[i];p[4+NV+i]=s[w].smooth[i];}for(int i=0;i<NV*NV;i++)p[4+2*NV+i]=s[w].mass[i];}
extern "C" int init_model(const Model*p){return cudaMemcpyToSymbol(md,p,sizeof(Model));}
extern "C" size_t bytes_per_world(){return sizeof(Work);}
extern "C" int launch(cudaStream_t stream,int n,const Data*d,void*scratch,int mode){if(mode!=2)k_pre<<<n,32,0,stream>>>(*d,(Work*)scratch);if(mode!=1)k_post<<<n,32,0,stream>>>(*d,(Work*)scratch);return cudaGetLastError();}
extern "C" int diagnostic(cudaStream_t stream,int n,void*scratch,float*out){get_diag<<<n,1,0,stream>>>((Work*)scratch,out);return cudaGetLastError();}
__global__ void get_contacts(Work*all,float*out){int w=blockIdx.x;for(int c=0;c<all[w].nc;c++){float*p=out+(w*NC+c)*10;Contact&x=all[w].con[c];p[0]=x.g1;p[1]=x.g2;p[2]=x.dist;p[3]=x.row;for(int k=0;k<3;k++){p[4+k]=x.p[k];p[7+k]=x.fr[k];}}}
extern "C" int diagnostic_contacts(cudaStream_t stream,int n,void*scratch,float*out){get_contacts<<<n,1,0,stream>>>((Work*)scratch,out);return cudaGetLastError();}
__global__ void get_failures(Work*all,int*out){int w=blockIdx.x;for(int p=threadIdx.x;p<NP;p+=32){out[w*2*NP+p]=all[w].failed_pairs[p];out[w*2*NP+NP+p]=all[w].resolved_pairs[p];}}
extern "C" int failures(cudaStream_t stream,int n,void*scratch,int*out){get_failures<<<n,32,0,stream>>>((Work*)scratch,out);return cudaGetLastError();}

extern "C" void resources(int*p){cudaFuncAttributes a;cudaFuncGetAttributes(&a,k_pre);p[0]=a.numRegs;p[1]=a.sharedSizeBytes;p[2]=a.localSizeBytes;cudaOccupancyMaxActiveBlocksPerMultiprocessor(p+3,k_pre,32,0);cudaFuncGetAttributes(&a,k_post);p[4]=a.numRegs;p[5]=a.sharedSizeBytes;p[6]=a.localSizeBytes;cudaOccupancyMaxActiveBlocksPerMultiprocessor(p+7,k_post,32,0);}
