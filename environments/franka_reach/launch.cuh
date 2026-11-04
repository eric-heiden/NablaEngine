struct Shared {Work s;Solve v;Data d;};
__global__ void task_step(IO io){
 extern __shared__ __align__(16) unsigned char storage[];Shared &mem=((Shared*)storage)[threadIdx.x/32];Work&s=mem.s;Solve&v=mem.v;Data&dd=mem.d;int w=blockIdx.x*WORLDS_PER_BLOCK+threadIdx.x/32,l=threadIdx.x&31;if(w>=NWORLDS)return;
 if(l==0){MARK(0);}
 for(int i=l;i<NPARAM;i+=32)s.params[i]=io.params[w*NPARAM+i];
 for(int i=l;i<NP;i+=32)s.failed_pairs[i]=s.resolved_pairs[i]=0;
 __syncwarp();if(l==0)s.ever_error=(int)io.diag[w*4+2]|io.param_errors[w];import_state(s,dd,w,io);__syncwarp();
 if(l==0){MARK(1);}
 for(int sub=0;sub<INTERNAL_STEPS;sub++){
 if(sub>0&&l==0){MARK(0);MARK(1);}
 #if STATIC_TABLE
 if(sub%2==0){for(int a=l;a<NU;a+=32){int idx=C_ctrlidx(a);if(idx>=0&&io.relative_delta){int j=M(actuator_trnid,2*a);dd.ctrl[a]=dd.qpos[M(jnt_qposadr,j)]+io.relative_delta[w*9+idx];}}__syncwarp();}
 #endif

 prepare(dd,s,w,l,v.H,v.p);__syncwarp();solve_integrate(dd,s,v,w,l,io,sub==INTERNAL_STEPS-1);__syncwarp();
 if(l==0){MARK(7);
 #if STAGE_PROFILE
 for(int k=0;k<7;k++)io.device_cycles[w*7+k]+=s.ticks[k+1]-s.ticks[k];
 #endif
 }
 }
 if(l==0){
 io.diag[w*4]=s.nc;io.diag[w*4+1]=s.nr;io.diag[w*4+2]=s.ever_error;io.diag[w*4+3]=s.niter;}
}
__global__ void clear_warm(float*warm,const bool*mask){int w=blockIdx.x;if(!mask||mask[w])for(int i=threadIdx.x;i<NV;i+=32)warm[w*NV+i]=0;}
extern "C" int launch(cudaStream_t stream,int n,const IO*io){cudaMemsetAsync(io->count,0,sizeof(int),stream);task_step<<<(n+WORLDS_PER_BLOCK-1)/WORLDS_PER_BLOCK,32*WORLDS_PER_BLOCK,sizeof(Shared)*WORLDS_PER_BLOCK,stream>>>(*io);return cudaGetLastError();}
extern "C" int refresh(cudaStream_t stream,int n,const Refresh*r,float*out,int*errors){refresh_parameters<<<n,32,0,stream>>>(*r,out,errors);return cudaGetLastError();}
extern "C" int reset_warm(cudaStream_t stream,int n,float*warm,const bool*mask){clear_warm<<<n,32,0,stream>>>(warm,mask);return cudaGetLastError();}
extern "C" int resources(int*p){int rc=cudaFuncSetAttribute(task_step,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)*WORLDS_PER_BLOCK);if(rc)return rc;cudaFuncAttributes a;cudaFuncGetAttributes(&a,task_step);p[0]=a.numRegs;p[1]=a.sharedSizeBytes;p[2]=a.localSizeBytes;cudaOccupancyMaxActiveBlocksPerMultiprocessor(p+3,task_step,32*WORLDS_PER_BLOCK,sizeof(Shared)*WORLDS_PER_BLOCK);p[4]=sizeof(Shared)*WORLDS_PER_BLOCK;p[5]=sizeof(Work);p[6]=sizeof(Solve);p[7]=sizeof(Data);return 0;}
// Untimed diagnostic entry point; never called by task_step or environment stepping.
__global__ void inspect_prepare(IO io,float*out){
 extern __shared__ __align__(16) unsigned char storage[];Shared&mem=*(Shared*)storage;Work&s=mem.s;Data&dd=mem.d;int w=blockIdx.x,l=threadIdx.x;
 for(int i=l;i<NPARAM;i+=32)s.params[i]=io.params[w*NPARAM+i];__syncwarp();import_state(s,dd,w,io);__syncwarp();prepare(dd,s,w,l,mem.v.H,mem.v.p);__syncwarp();
 if(l==0){float*p=out+w*(NQ+4*NV+NU+NV*NV+2);for(int i=0;i<NQ;i++)*p++=dd.qpos[i];for(int i=0;i<NV;i++)*p++=dd.qvel[i];for(int i=0;i<NU;i++)*p++=dd.ctrl[i];for(int i=0;i<NV*NV;i++)*p++=s.mass[i];for(int i=0;i<NV;i++)*p++=s.bias[i];for(int i=0;i<NV;i++)*p++=s.smooth[i];for(int i=0;i<NV;i++)*p++=s.qsm[i];*p++=s.nc;*p++=s.nr;}
}
extern "C" int inspect(cudaStream_t stream,int n,const IO*io,float*out){cudaFuncSetAttribute(inspect_prepare,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared));inspect_prepare<<<n,32,sizeof(Shared),stream>>>(*io,out);return cudaGetLastError();}
