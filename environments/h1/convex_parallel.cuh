// Warp-cooperative support mapping and EPA scans. Tie order matches the serial loops.
__device__ Vec support_warp(Work&s,int w,int g,Vec dir){
 if(M(geom_type,g)!=7)return support(s,w,g,dir);
 int l=(threadIdx.x&31);float v[3]={dir.x,dir.y,dir.z},d[3];qrot_inv(s.gq[g],v,d);
 int mesh=M(geom_dataid,g),adr=M(mesh_vertadr,mesh),n=M(mesh_vertnum,mesh),best=0x7fffffff;float value=-1e30f;
 for(int i=l;i<n;i+=32){float z=0;for(int k=0;k<3;k++)z+=M(mesh_vert,3*(adr+i)+k)*d[k];if(z>value){value=z;best=i;}}
 for(int off=16;off;off/=2){float z=__shfl_down_sync(0xffffffff,value,off);int i=__shfl_down_sync(0xffffffff,best,off);if(l+off<32&&(z>value||(z==value&&i<best))){value=z;best=i;}}
 best=__shfl_sync(0xffffffff,best,0);float p[3],out[3];for(int k=0;k<3;k++)p[k]=M(mesh_vert,3*(adr+best)+k);qrot(s.gq[g],p,out);return Vec(out)+Vec(s.gp[g]);
}
__device__ Vertex sup_warp(Work&s,int w,int a,int b,Vec d){Vertex v;v.a=support_warp(s,w,a,d);v.b=support_warp(s,w,b,-d);v.p=v.a-v.b;return v;}
__device__ Vec from_leader(Vec v){return Vec(__shfl_sync(0xffffffff,v.x,0),__shfl_sync(0xffffffff,v.y,0),__shfl_sync(0xffffffff,v.z,0));}
__device__ bool convex(Work&s,int w,int a,int b,float&distance,float*pos,float*normal){
 int l=(threadIdx.x&31);distance=0;Vertex*v=(Vertex*)s.collision_scratch;Vec d=Vec(s.gp[b])-Vec(s.gp[a]);if(dot(d,d)<1e-12f)d=Vec(1,0,0);
 Vertex seed=sup_warp(s,w,a,b,d);if(l==0)v[0]=seed;__syncwarp();int n=1;d=-seed.p;bool hit=false;
 for(int it=0;it<48;it++){
  if(dot(d,d)<1e-20f)d=Vec(1,.271f,.113f);Vertex p=sup_warp(s,w,a,b,d);if(dot(p.p,d)<0)return false;
  if(l==0){v[n++]=p;hit=simplex(v,n,d);}__syncwarp();n=__shfl_sync(0xffffffff,n,0);hit=__shfl_sync(0xffffffff,(int)hit,0);d=from_leader(d);if(hit)break;
 }
 if(!hit){int result=0;if(l==0){result=gjk_fallback(s,w,a,b,v);if(result<0)s.unsupported|=64;}__syncwarp();result=__shfl_sync(0xffffffff,result,0);if(result<=0)return false;}
 Face*faces=(Face*)(v+64);int nf=4,nv=4,best=-1;
 if(l<4){int aa[4]={0,0,0,1},bb[4]={1,3,2,3},cc[4]={2,1,3,2},op[4]={3,2,1,0};face_make(faces[l],aa[l],bb[l],cc[l],v);Face&f=faces[l];if(f.live&&dot(f.n,v[op[l]].p-v[f.a].p)>0){int t=f.b;f.b=f.c;f.c=t;f.n=-f.n;f.d=-f.d;}}__syncwarp();
 for(int it=0;it<60;it++){
  float mind=1e30f;int idx=0x7fffffff;for(int f=l;f<nf;f+=32)if(faces[f].live&&faces[f].d<mind){mind=faces[f].d;idx=f;}
  for(int off=16;off;off/=2){float z=__shfl_down_sync(0xffffffff,mind,off);int i=__shfl_down_sync(0xffffffff,idx,off);if(l+off<32&&(z<mind||(z==mind&&i<idx))){mind=z;idx=i;}}
  best=__shfl_sync(0xffffffff,idx,0);if(best==0x7fffffff){if(l==0)s.unsupported|=128;return false;}
  Face chosen=faces[best];Vertex p=sup_warp(s,w,a,b,chosen.n);float delta=dot(p.p,chosen.n)-chosen.d;bool dup=false;
  for(int i=l;i<nv;i+=32)if(dot(p.p-v[i].p,p.p-v[i].p)<1e-14f)dup=true;
  if(delta<1e-5f||__any_sync(0xffffffff,dup)||nv==64)break;
  bool failed=false;
  if(l==0){int*ea=(int*)(faces+128),*eb=ea+96,ne=0;
   for(int f=0;f<nf&&!failed;f++)if(faces[f].live&&dot(faces[f].n,p.p-v[faces[f].a].p)>1e-7f){Face q=faces[f];faces[f].live=false;int aa[3]={q.a,q.b,q.c},bb[3]={q.b,q.c,q.a};for(int k=0;k<3;k++){bool found=false;for(int e=0;e<ne;e++)if(ea[e]==bb[k]&&eb[e]==aa[k]){ea[e]=ea[--ne];eb[e]=eb[ne];found=true;break;}if(!found){if(ne>=96){failed=true;break;}ea[ne]=aa[k];eb[ne++]=bb[k];}}}
   if(!failed){v[nv]=p;for(int e=0;e<ne;e++){int free=-1;for(int f=0;f<nf;f++)if(!faces[f].live){free=f;break;}if(free<0){if(nf>=128){failed=true;break;}free=nf++;}face_make(faces[free],ea[e],eb[e],nv,v);}nv++;}
   if(failed)s.unsupported|=256;
  }__syncwarp();if(__shfl_sync(0xffffffff,(int)failed,0))return false;nf=__shfl_sync(0xffffffff,nf,0);nv=__shfl_sync(0xffffffff,nv,0);
 }
 if(l==0){Face f=faces[best];Vec A=v[f.a].p,B=v[f.b].p,C=v[f.c].p,P=f.n*f.d,ab=B-A,ac=C-A,ap=P-A;float aa=dot(ab,ab),bb=dot(ab,ac),cc=dot(ac,ac),pa=dot(ap,ab),pc=dot(ap,ac),den=aa*cc-bb*bb;
  float wb=den>1e-20f?(cc*pa-bb*pc)/den:0,wc=den>1e-20f?(aa*pc-bb*pa)/den:0,wa=1-wb-wc;Vec pa0=v[f.a].a*wa+v[f.b].a*wb+v[f.c].a*wc,pb0=v[f.a].b*wa+v[f.b].b*wb+v[f.c].b*wc,p=(pa0+pb0)*.5f;
  pos[0]=p.x;pos[1]=p.y;pos[2]=p.z;normal[0]=f.n.x;normal[1]=f.n.y;normal[2]=f.n.z;distance=-f.d;
 }distance=__shfl_sync(0xffffffff,distance,0);return distance<0;
}
