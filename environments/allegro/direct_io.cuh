// Direct Newton input and output. All lookup constants are task-specialized.
__device__ void tr_read(const float*t,float*p,float*q){for(int k=0;k<3;k++)p[k]=t[k];q[0]=t[6];for(int k=0;k<3;k++)q[k+1]=t[k+3];}
__device__ void import_state(Work&s,Data&dd,int w,const IO&io){
 if((threadIdx.x&31)==0){
  float pp[3],pq[4],cp[3],cq[4],ci[4],off[3];tr_read(io.rootxp+w*NWJ*7,pp,pq);
  for(int k=0;k<3;k++)cp[k]=C_Xc(k);cq[0]=C_Xc(6);for(int k=0;k<3;k++)cq[k+1]=C_Xc(3+k);
  ci[0]=cq[0];for(int k=1;k<4;k++)ci[k]=-cq[k];qmul(pq,ci,dd.root_quat);qrot(dd.root_quat,cp,off);
  for(int k=0;k<3;k++)dd.root_pos[k]=pp[k]-off[k];
 }


 for(int i=(threadIdx.x&31);i<NV;i+=32){dd.qfrc_applied[i]=0;dd.qacc_warmstart[i]=io.warm[w*NV+i];}
 __syncwarp();
 for(int j=(threadIdx.x&31);j<NWJ;j+=32){
  int qa=C_mjq(j),da=C_mjv(j);if(qa<0)continue;
  int oq=w*NWQ+C_qstart(j),ov=w*NWV+C_vstart(j),b=C_child(j);
  if(C_jtype(j)==4){
   float pp[3],pq[4],cp[3],cq[4],rp[3],rq[4],q[4],tmp[4],ci[4],p[3],off[3];
   for(int k=0;k<3;k++){pp[k]=P(Xp,j*7+k);cp[k]=C_Xc(j*7+k);rp[k]=io.qin[oq+k];}pq[0]=P(Xp,j*7+6);cq[0]=C_Xc(j*7+6);rq[0]=io.qin[oq+6];
   for(int k=0;k<3;k++){pq[k+1]=P(Xp,j*7+3+k);cq[k+1]=C_Xc(j*7+3+k);rq[k+1]=io.qin[oq+3+k];}ci[0]=cq[0];for(int k=1;k<4;k++)ci[k]=-cq[k];qmul(pq,rq,tmp);qmul(tmp,ci,q);qrot(pq,rp,p);qrot(q,cp,off);
   for(int k=0;k<3;k++)dd.qpos[qa+k]=pp[k]+p[k]-off[k];for(int k=0;k<4;k++)dd.qpos[qa+3+k]=q[k];
   float vp[3],ap[3],v[3],a[3],com[3],co[3],local[3];for(int k=0;k<3;k++){vp[k]=io.vin[ov+k];ap[k]=io.vin[ov+3+k];com[k]=P(com,b*3+k);}qrot(pq,vp,v);qrot(pq,ap,a);qrot(q,com,co);crossadd(v,a,co,-1);qrot_inv(q,a,local);
   for(int k=0;k<3;k++){dd.qvel[da+k]=v[k];dd.qvel[da+3+k]=local[k];}
  }else for(int k=0;k<C_nd(j);k++){dd.qpos[qa+k]=io.qin[oq+k]+P(ref,C_vstart(j)+k);dd.qvel[da+k]=io.vin[ov+k];if(!C_kin(b))dd.qfrc_applied[da+k]=io.jforce[ov+k];}
 }
 for(int b=(threadIdx.x&31);b<NB;b+=32){int nb=C_bodymap(b);for(int k=0;k<6;k++)dd.xfrc_applied[b*6+k]=nb<0||C_kin(nb)?0:io.bforce[(w*NWB+nb)*6+k];if(nb>=0&&!C_kin(nb)&&C_freev(nb)>=0)for(int k=0;k<6;k++)dd.xfrc_applied[b*6+k]+=io.jforce[w*NWV+C_freev(nb)+k];}
 for(int a=(threadIdx.x&31);a<NU;a+=32){int idx=C_ctrlidx(a);dd.ctrl[a]=idx>=0?io.tq[w*NWT+C_ctrlq(a)]+P(ref,idx):idx<=-2?io.tv[w*NWV-(idx+2)]:0;}
}
__device__ void export_state(Work&s,Data&dd,int w,const IO&io){

 for(int b=1+(threadIdx.x&31);b<NB;b+=32){
  int nb=C_bodymap(b);if(nb<0||C_kin(nb))continue;int id=w*NWB+nb;
  for(int k=0;k<3;k++){io.bq[id*7+k]=s.x[b][k];io.bq[id*7+3+k]=s.q[b][k+1];}io.bq[id*7+6]=s.q[b][0];
  float com[3],off[3],vc[3];for(int k=0;k<3;k++)com[k]=P(com,nb*3+k);qrot(s.q[b],com,off);copy3(vc,s.v[b]);crossadd(vc,s.om[b],off);for(int k=0;k<3;k++){io.bv[id*6+k]=vc[k];io.bv[id*6+3+k]=s.om[b][k];}
 }
 for(int j=(threadIdx.x&31);j<NWJ;j+=32){
  int qa=C_mjq(j),da=C_mjv(j);if(qa<0)continue;int oq=w*NWQ+C_qstart(j),ov=w*NWV+C_vstart(j),b=C_child(j);if(C_kin(b))continue;
  if(C_jtype(j)==4){
   float pp[3],pq[4],cp[3],cq[4],q[4],t[3],p[3],tmp[4],outq[4],ipq[4];for(int k=0;k<3;k++){pp[k]=P(Xp,j*7+k);cp[k]=C_Xc(j*7+k);}pq[0]=P(Xp,j*7+6);cq[0]=C_Xc(j*7+6);for(int k=0;k<3;k++){pq[k+1]=P(Xp,j*7+3+k);cq[k+1]=C_Xc(j*7+3+k);}ipq[0]=pq[0];for(int k=1;k<4;k++)ipq[k]=-pq[k];for(int k=0;k<4;k++)q[k]=dd.qpos[qa+3+k];qrot(q,cp,t);for(int k=0;k<3;k++)t[k]+=dd.qpos[qa+k]-pp[k];qrot_inv(pq,t,p);qmul(q,cq,tmp);qmul(ipq,tmp,outq);
   for(int k=0;k<3;k++){io.qout[oq+k]=p[k];io.qout[oq+3+k]=outq[k+1];}io.qout[oq+6]=outq[0];
   float av[3],a[3],v[3],co[3],com[3],vp[3],ap[3];for(int k=0;k<3;k++){av[k]=dd.qvel[da+3+k];v[k]=dd.qvel[da+k];com[k]=P(com,b*3+k);}qrot(q,av,a);qrot(q,com,co);crossadd(v,a,co);qrot_inv(pq,v,vp);qrot_inv(pq,a,ap);for(int k=0;k<3;k++){io.vout[ov+k]=vp[k];io.vout[ov+3+k]=ap[k];}
  }else for(int k=0;k<C_nd(j);k++){io.qout[oq+k]=dd.qpos[qa+k]-P(ref,C_vstart(j)+k);io.vout[ov+k]=dd.qvel[da+k];}
 }
 for(int i=(threadIdx.x&31);i<NV;i+=32)io.warm[w*NV+i]=dd.qacc_warmstart[i];
}
__device__ void export_contacts(Work&s,int w,int l,const IO&io){

 if(l==0)s.output_base=atomicAdd(io.count,s.nc);__syncwarp();int base=s.output_base;
 for(int cidx=l;cidx<s.nc;cidx+=32){Contact&c=s.con[cidx];if(c.row+4>NR){atomicOr(&s.unsupported,2);continue;}int id=base+cidx;if(id>=MAX_OUT){s.unsupported|=512;continue;}
  io.shape0[id]=C_shape(c.g1)<0?C_shape(c.g1):w*C_shape_stride(c.g1)+C_shape(c.g1);io.shape1[id]=C_shape(c.g2)<0?C_shape(c.g2):w*C_shape_stride(c.g2)+C_shape(c.g2);
  float force[3]={0,0,0};for(int e=0;e<4;e++){float f=s.force[c.row+e];for(int k=0;k<3;k++)force[k]-=f*(c.fr[k]+(e%2?-1.f:1.f)*c.fr[3*(1+e/2)+k]);}
  for(int k=0;k<3;k++){io.normal[id*3+k]=c.fr[k];io.cforce[id*6+k]=force[k];io.cforce[id*6+3+k]=0;}
  int bodies[2]={M(geom_bodyid,c.g1),M(geom_bodyid,c.g2)};for(int side=0;side<2;side++){float pt[3],out[3];int b=bodies[side];for(int k=0;k<3;k++)pt[k]=c.p[k]+(side?.5f:-.5f)*c.dist*c.fr[k]-s.x[b][k];qrot_inv(s.q[b],pt,out);for(int k=0;k<3;k++)(side?io.point1:io.point0)[id*3+k]=out[k];}
 }
}
