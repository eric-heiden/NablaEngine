// Full-step CUDA research implementation. All dynamics and constraints execute here.
#include "primitive_math.cuh"
// Row of packed lower-triangle index t (row-major, j<=i).
__device__ __forceinline__ int tri_row(int t){int i=(int)((sqrtf(8.f*t+1.f)-1.f)*.5f);if((i+1)*(i+2)/2<=t)i++;else if(i*(i+1)/2>t)i--;return i;}
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
  if(b==0||b==13){if(b==13){for(int k=0;k<3;k++)pos[k]=dd.table_pos[k];for(int k=0;k<4;k++)quat[k]=dd.table_quat[k];}copy3(s.x[b],pos);for(int k=0;k<4;k++)s.q[b][k]=quat[k];}
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
__device__ int reduced_type(Work&s,int g){int t=M(geom_type,g);return g==2?3:t==7?proxy_shape(M(geom_dataid,g)):t;}
__device__ void add_contact(Work&s,int pair,int slot,float dist,const float*pos,const float*frame){
 if(dist>=0)return;int id=atomicAdd(&s.nc,1);if(id>=NC){atomicOr(&s.unsupported,4);return;}Contact &c=s.con[id];c.row=pair;c.g1=PAIRS[2*pair];c.g2=PAIRS[2*pair+1];c.dist=dist;for(int k=0;k<3;k++)c.p[k]=pos[k];for(int k=0;k<9;k++)c.fr[k]=frame[k]*((reduced_type(s,PAIRS[2*pair])>reduced_type(s,PAIRS[2*pair+1])&&(k<3||k>=6))?-1:1);
}
#include "primitive_collision.cuh"
__device__ void emit(Work&s,int pair,float dist,const float*p,const float*n){float fr[9];make_frame(n,fr);add_contact(s,pair,0,dist,p,fr);}
struct Proxy {int type;float p[3],q[4],size[3];};
__device__ Proxy proxy_geometry(Work&s,int g){
 Proxy o;o.type=reduced_type(s,g);copy3(o.p,s.gp[g]);for(int k=0;k<4;k++)o.q[k]=s.gq[g][k];
 for(int k=0;k<3;k++)o.size[k]=M(geom_size,3*g+k);
 if(M(geom_type,g)==7){int mesh=M(geom_dataid,g);float center[3],offset[3];for(int k=0;k<3;k++)center[k]=proxy_center(3*mesh+k);qrot(o.q,center,offset);for(int k=0;k<3;k++){o.p[k]+=offset[k];o.size[k]=proxy_half(3*mesh+k);}
  if(o.type==2||o.type==3){o.size[0]=proxy_radius(mesh);o.size[1]=proxy_segment(mesh);o.size[2]=0;}
 }
 if(g==2){o.type=3;o.size[0]=.0710468f;o.size[1]=.0667092f;o.size[2]=0;}
 return o;
}
__device__ void analytic_plane(Work&s,int pair,const Proxy&a,const Proxy&b){
 float normal[3],frame[9],point[3];zaxis_rn(a.q,normal);make_frame(normal,frame);
 if(b.type==2){float dist=plane_point(a.p,normal,b.p,b.size[0],point);add_contact(s,pair,0,dist,point,frame);}
 else if(b.type==3){float axis[3];zaxis_rn(b.q,axis);plane_capsule_frame(normal,axis,frame);for(int e=0;e<2;e++){float end[3];capsule_end(b.p,axis,b.size[1],e,end);float dist=plane_point(a.p,normal,end,b.size[0],point);add_contact(s,pair,e,dist,point,frame);}}
 else if(b.type==6){float R[9];quat2mat(b.q,R);float depths[4]={1e30f,1e30f,1e30f,1e30f},points[4][3];
  for(int i=0;i<8;i++){float world[3];for(int k=0;k<3;k++){world[k]=b.p[k];for(int j=0;j<3;j++)world[k]+=R[3*k+j]*((i&(1<<j))?b.size[j]:-b.size[j]);}float dist=plane_point(a.p,normal,world,0,point);int worst=0;for(int j=1;j<4;j++)if(depths[j]>depths[worst])worst=j;if(dist<depths[worst]){depths[worst]=dist;copy3(points[worst],point);}}
  for(int i=0;i<4;i++)if(depths[i]<0)add_contact(s,pair,i,depths[i],points[i],frame);
 }else atomicOr(&s.unsupported,32);
}
// The fixed tabletop: analytic top-face contacts, bounded by its two half widths.
__device__ void table_contact(Work&s,int pair,int table,int g,const Proxy&shape){
 float normal[3],frame[9],top[3],axis[3],R[9];zaxis_rn(s.gq[table],normal);make_frame(normal,frame);
 for(int k=0;k<3;k++)top[k]=s.gp[table][k]+normal[k]*M(geom_size,3*table+2);
 int ga=PAIRS[2*pair],gb=PAIRS[2*pair+1];int first=reduced_type(s,ga)>reduced_type(s,gb)?gb:ga;
 if(first!=table)for(int k=0;k<9;k++)if(k<3||k>=6)frame[k]*=-1;
 zaxis_rn(shape.q,axis);quat2mat(shape.q,R);
 float depth[4]={1e30f,1e30f,1e30f,1e30f},points[4][3];
 int count=shape.type==6?8:shape.type==3?2:1;
 for(int e=0;e<count;e++){
  float c[3];copy3(c,shape.p);
  if(shape.type==3)for(int k=0;k<3;k++)c[k]+=(e?-1:1)*shape.size[1]*axis[k];
  if(shape.type==6)for(int k=0;k<3;k++)for(int j=0;j<3;j++)c[k]+=R[3*k+j]*((e&(1<<j))?shape.size[j]:-shape.size[j]);
  float radius=shape.type==6?0:shape.size[0],p[3],v[3],local[3];
  float dist=plane_point(top,normal,c,radius,p);for(int k=0;k<3;k++)v[k]=p[k]-s.gp[table][k];qrot_inv(s.gq[table],v,local);
  if(fabsf(local[0])>M(geom_size,3*table)||fabsf(local[1])>M(geom_size,3*table+1)||dist>=0)continue;
  int worst=0;for(int j=1;j<4;j++)if(depth[j]>depth[worst])worst=j;
  if(dist<depth[worst]){depth[worst]=dist;copy3(points[worst],p);}
 }
 for(int e=0;e<4;e++)if(depth[e]<0)add_contact(s,pair,e,depth[e],points[e],frame);
}
// Flat finger pad against one of the four small box objects. SAT plus clipped face vertices.
__device__ void pad_object(Work&s,int pair,const Proxy&a,const Proxy&b){
 float A[9],B[9],d[3],n[3],best=1e30f;quat2mat(a.q,A);quat2mat(b.q,B);
 for(int k=0;k<3;k++)d[k]=b.p[k]-a.p[k];
 for(int axis=0;axis<15;axis++){
  float u[3];if(axis<6)for(int k=0;k<3;k++)u[k]=axis<3?A[3*k+axis]:B[3*k+axis-3];
  else{float x[3],y[3];for(int k=0;k<3;k++){x[k]=A[3*k+(axis-6)/3];y[k]=B[3*k+(axis-6)%3];}cross(x,y,u);}
  float len=sqrtf(dot3(u,u));if(len<1e-6f)continue;for(int k=0;k<3;k++)u[k]/=len;
  float ra=0,rb=0;for(int j=0;j<3;j++){float da=0,db=0;for(int k=0;k<3;k++){da+=A[3*k+j]*u[k];db+=B[3*k+j]*u[k];}ra+=a.size[j]*fabsf(da);rb+=b.size[j]*fabsf(db);}
  float along=dot3(d,u),pen=ra+rb-fabsf(along);if(pen<=0)return;if(pen<best){best=pen;for(int k=0;k<3;k++)n[k]=(along<0?-1:1)*u[k];}
 }
 float frame[9];make_frame(n,frame);int emitted=0;
 // Finger corners normally lie inside the object's supporting face; the reverse covers thin objects.
 for(int side=0;side<2&&emitted<4;side++){
  const Proxy&v=side?b:a;const Proxy&o=side?a:b;const float*V=side?B:A;const float*O=side?A:B;
  float nn[3],support=0;for(int k=0;k<3;k++)nn[k]=side?-n[k]:n[k];
  for(int j=0;j<3;j++){float z=0;for(int k=0;k<3;k++)z+=V[3*k+j]*nn[k];support+=v.size[j]*fabsf(z);}
  for(int e=0;e<8&&emitted<4;e++){
   float p[3],rel[3],local[3],off[3];for(int k=0;k<3;k++){off[k]=0;for(int j=0;j<3;j++)off[k]+=V[3*k+j]*((e&(1<<j))?v.size[j]:-v.size[j]);p[k]=v.p[k]+off[k];rel[k]=p[k]-o.p[k];}
   if(dot3(off,nn)<support-.00001f)continue;mtv3(O,rel,local);
   bool inside=true;for(int j=0;j<3;j++)inside&=fabsf(local[j])<=o.size[j]+.00001f;if(!inside)continue;
   for(int k=0;k<3;k++)p[k]-=.5f*best*nn[k];add_contact(s,pair,emitted++,-best,p,frame);
  }
 }
 if(!emitted){float p[3];for(int k=0;k<3;k++)p[k]=.5f*(a.p[k]+b.p[k]);add_contact(s,pair,0,-best,p,frame);}
}
__device__ void collisions(Work&s,int w){
 for(int pair=(threadIdx.x&31);pair<NP;pair+=32){
  int ga=PAIRS[2*pair],gb=PAIRS[2*pair+1];Proxy a=proxy_geometry(s,ga),b=proxy_geometry(s,gb);
  if(gb==15){table_contact(s,pair,15,ga,a);continue;}
  if(a.type>b.type){Proxy t=a;a=b;b=t;}
  if(a.type==0){analytic_plane(s,pair,a,b);continue;}
  float delta[3];for(int k=0;k<3;k++)delta[k]=a.p[k]-b.p[k];float bound=M(geom_rbound,ga)+M(geom_rbound,gb)+.004f;if(dot3(delta,delta)>bound*bound)continue;
  float p[3],normal[3],dist;
  if(a.type==2&&b.type==2){sphere_sphere(a.p,a.size[0],b.p,b.size[0],&dist,p,normal);emit(s,pair,dist,p,normal);}
  else if(a.type==2&&b.type==3){float axis[3],v0[3],ab[3],ap[3],gam,t,pt[3];zaxis_rn(b.q,axis);segment_point(a.p,b.p,axis,b.size[1],v0,ab,ap,&gam,&t,pt);sphere_sphere(a.p,a.size[0],pt,b.size[0],&dist,p,normal);emit(s,pair,dist,p,normal);}
  else if(a.type==3&&b.type==3){CapCap cc;capcap_setup(a.p,a.q,a.size[1],b.p,b.q,b.size[1],cc);bool f1,f2;float raw,x1,x2;
   if(fabsf(cc.det)>=1e-15f){capcap_params(cc,&x1,&x2,&f1,&f2,&raw);capcap_core(a.p,b.p,cc,x1,x2,a.size[0],b.size[0],&dist,p,normal);emit(s,pair,dist,p,normal);}
   else for(int e=0;e<4;e++){capcap_candidate(cc,e,&x1,&x2,&f1,&f2,&raw);capcap_core(a.p,b.p,cc,x1,x2,a.size[0],b.size[0],&dist,p,normal);emit(s,pair,dist,p,normal);}
  }else if(a.type==2&&b.type==6){float R[9];quat2mat(b.q,R);sphere_box(a.p,a.size[0],b.p,R,b.size,&dist,p,normal);emit(s,pair,dist,p,normal);}
  else if(a.type==3&&b.type==6){float R[9],axis[3];quat2mat(b.q,R);zaxis_rn(a.q,axis);capsule_box(s,pair,a.p,axis,a.size[0],a.size[1],b.p,R,b.size);}
  else if(a.type==6&&b.type==6){pad_object(s,pair,a,b);}
  else atomicOr(&s.unsupported,32);
 }
}


__device__ void constraints(Work&s,Data &dd,int w){
 if((threadIdx.x&31)==0){
 s.nr=s.ne=s.nf=0;
 for(int e=0;e<NEQ;e++){
  if(M(eq_type,e)!=2){s.unsupported|=8;continue;}int a=M(eq_obj1id,e),b=M(eq_obj2id,e),qa=M(jnt_qposadr,a),da=M(jnt_dofadr,a);float x=b>=0?D(qpos,M(jnt_qposadr,b))-M(qpos0,M(jnt_qposadr,b)):0;
  float poly=0,deriv=0,pow=1;for(int k=0;k<5;k++){poly+=M(eq_data,e*11+k)*pow;pow*=x;}pow=1;for(int k=1;k<5;k++){deriv+=k*M(eq_data,e*11+k)*pow;pow*=x;}
  float j[NV]={0},ref[2],imp[5];j[da]=1;float iw=M(dof_invweight0,da);if(b>=0){int db=M(jnt_dofadr,b);j[db]=-deriv;iw+=M(dof_invweight0,db);}for(int k=0;k<2;k++)ref[k]=M(eq_solref,e*2+k);for(int k=0;k<5;k++)imp[k]=M(eq_solimp,e*5+k);row_add(s,dd,w,j,D(qpos,qa)-M(qpos0,qa)-poly,iw,ref,imp);s.ne++;
 }

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
__device__ int jslot(int b,int i){return J_SLOT[b*NV+i];}
__device__ void prepare(Data &dd,Work &s,int w,int l,float*pre_A,float*pre_rhs){
 if(l==0){s.unsupported=0;s.nc=0;}__syncwarp();kin<true>(s,w,dd);if(l==0){MARK(2);}__syncwarp();
 for(int t=l;t<NB*NV;t+=32){int b=t/NV,i=t%NV;if(ancestor(s,b,i,w))jacpoint(s,w,b,s.com[b],i,s.Jv[jslot(b,i)]);}__syncwarp();
 // Symmetric: assemble the lower triangle and mirror it.
 for(int t=l;t<NV*(NV+1)/2;t+=32){int i=tri_row(t),j=t-i*(i+1)/2;float v=i==j?M(dof_armature,i):0;unsigned int bodies=DOF_BODIES[i]&DOF_BODIES[j];while(bodies){int b=__ffs(bodies)-1;bodies&=bodies-1;float Ij[3],Wi[3],Wj[3];for(int k=0;k<3;k++){Wi[k]=s.rot[i]?s.axis[i][k]:0;Wj[k]=s.rot[j]?s.axis[j][k]:0;}inertia_mul(s.IW[b],Wj,Ij);v+=M(body_mass,b)*dot3(s.Jv[jslot(b,i)],s.Jv[jslot(b,j)])+dot3(Wi,Ij);}s.mass[i*NV+j]=v;s.mass[j*NV+i]=v;}
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
  // Lower triangle only; inactive rows (diag==0) contribute nothing.
  for(int t=l;t<NV*(NV+1)/2;t+=32){int i=tri_row(t),j=t-i*(i+1)/2;float h=s.mass[i*NV+j];for(int r=0;r<n;r++){float d=v.diag[r];if(d!=0.f)h+=s.J[r*NV+i]*d*s.J[r*NV+j];}v.H[i*NV+j]=h;}__syncwarp();
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
