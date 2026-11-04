"""Cooperative task-specific CUDA stages. Called only at source assembly time."""
def block(s,start):
 a=s.index('{',start);depth=1;b=a+1
 while depth:
  depth+=(s[b]=='{')-(s[b]=='}');b+=1
 return a,b

def transform(src,tag):
 # Tree dependencies are synchronized by baked depth; inertias and bias are independent per body.
 start=src.index('template<bool DYNAMICS> __device__ void kin(');a,z=block(src,start)
 f=src[a+1:z-1];ba,bz=block(f,f.index('for(int b=0;'))
 body=f[ba+1:bz-1];cut=body.index('  if constexpr(DYNAMICS){')
 pose=body[:cut];dyn=body[cut:]
 geom=f[bz:].replace('for(int g=0;g<NG;g++)','for(int g=threadIdx.x;g<NG;g+=32)')
 nf='\n int lane=threadIdx.x;\n for(int depth=0;depth<=TREE_DEPTH;depth++){int b=lane;if(b<NB && BODY_DEPTH[b]==depth){'+pose+'}\n __syncwarp();}\n for(int b=lane;b<NB;b+=32){float r[3];'+dyn+'}\n __syncwarp();\n'+geom+'\n__syncwarp();\n'
 src=src[:a+1]+nf+src[z-1:]
 old='if(l==0){s.unsupported=0;s.nc=0;kin<true>(s,w,dd);MARK(2);}__syncwarp();'
 assert old in src;src=src.replace(old,'if(l==0){s.unsupported=0;s.nc=0;}__syncwarp();kin<true>(s,w,dd);if(l==0){MARK(2);}__syncwarp();')
 # Fully cooperative dense factor/solve; each lane owns one matrix row.
 cstart=src.index('__device__ void cholesky(');ca,cz=block(src,cstart)
 src=src[:cz]+'''\n__device__ void chol_parallel(float*A,float*rhs,int l){
 for(int j=0;j<NV;j++){
  if(l==j){float z=A[j*NV+j];for(int k=0;k<j;k++)z-=A[j*NV+k]*A[j*NV+k];A[j*NV+j]=sqrtf(fmaxf(z,1e-14f));}__syncwarp();
  if(l>j && l<NV){float z=A[l*NV+j];for(int k=0;k<j;k++)z-=A[l*NV+k]*A[j*NV+k];A[l*NV+j]=z/A[j*NV+j];}__syncwarp();
 }
 for(int j=0;j<NV;j++){if(l==j)rhs[j]/=A[j*NV+j];__syncwarp();if(l>j&&l<NV)rhs[l]-=A[l*NV+j]*rhs[j];__syncwarp();}
 for(int j=NV-1;j>=0;j--){if(l==j)rhs[j]/=A[j*NV+j];__syncwarp();if(l<j)rhs[l]-=A[j*NV+l]*rhs[j];__syncwarp();}
}
'''+src[cz:]
 old='if(l==0){float *A=pre_A,*rhs=pre_rhs;for(int i=0;i<NV*NV;i++)A[i]=s.mass[i];for(int i=0;i<NV;i++)rhs[i]=s.smooth[i];cholesky(A,rhs);for(int i=0;i<NV;i++){s.qsm[i]=rhs[i];}MARK(3);collisions(s,w);MARK(4);constraints(s,dd,w);MARK(5);}'
 assert old in src
 src=src.replace(old,'''for(int i=l;i<NV*NV;i+=32)pre_A[i]=s.mass[i];if(l<NV)pre_rhs[l]=s.smooth[l];__syncwarp();chol_parallel(pre_A,pre_rhs,l);if(l<NV)s.qsm[l]=pre_rhs[l];__syncwarp();if(l==0){MARK(3);collisions(s,w);MARK(4);constraints(s,dd,w);MARK(5);}__syncwarp();''')
 # Integration and state export are now called by the full warp.
 st=src.index(' // Contact force and geometry are final outputs');ia=src.index(' if(l==0){',st);ib,iz=block(src,ia)
 old=src[ib+1:iz-1]
 js=old.index('  for(int j=0;j<NJ;j++)');ja,jz=block(old,js)
 joints=old[js:jz].replace('for(int j=0;j<NJ;j++)','for(int j=l;j<NJ;j+=32)')
 tail='''
 float *rhs=v.p,*A=v.H;float h=M(timestep,0);
 if(l==0)s.niter=nit;
 for(int i=l;i<NV*NV;i+=32)A[i]=s.mass[i];__syncwarp();
 if(l<NV){D(qacc_warmstart,l)=v.a[l];float z=0;for(int j=0;j<NV;j++)z+=s.mass[l*NV+j]*v.a[j];rhs[l]=z;A[l*NV+l]+=h*s.damp[l];}__syncwarp();
 chol_parallel(A,rhs,l);if(l<NV)D(qvel,l)+=h*rhs[l];__syncwarp();
 '''+joints+'''
 __syncwarp();kin<false>(s,w,dd);export_state(s,dd,w,io);__syncwarp();
 if(l==0){for(int i=0;i<NV;i++)if(!isfinite(D(qvel,i)))s.unsupported|=1024;for(int i=0;i<NQ;i++)if(!isfinite(D(qpos,i)))s.unsupported|=1024;s.ever_error|=s.unsupported;}
'''
 assert 'export_state' in old
 src=src[:ia]+tail+src[iz:]
 # All pointer bundles use references (the earlier instrumented by-value failure stays fixed).
 return src.replace('IO io)', 'const IO&io)')
