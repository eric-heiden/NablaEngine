// Support-mapped convex collision, including finite cylinders and per-world meshes.
struct Vec {float x,y,z;__device__ Vec(float a=0,float b=0,float c=0):x(a),y(b),z(c){} __device__ Vec(const float*p):x(p[0]),y(p[1]),z(p[2]){}};
__device__ Vec operator+(Vec a,Vec b){return Vec(a.x+b.x,a.y+b.y,a.z+b.z);}
__device__ Vec operator-(Vec a,Vec b){return Vec(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ Vec operator-(Vec a){return Vec(-a.x,-a.y,-a.z);}
__device__ Vec operator*(Vec a,float t){return Vec(a.x*t,a.y*t,a.z*t);}
__device__ float dot(Vec a,Vec b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ Vec cr(Vec a,Vec b){return Vec(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
__device__ Vec perp(Vec a){Vec x=cr(a,Vec(1,0,0));if(dot(x,x)<1e-12f)x=cr(a,Vec(0,1,0));return x;}
__device__ Vec support(Work&s,int w,int g,Vec dir){
 float v[3]={dir.x,dir.y,dir.z},d[3],p[3]={0,0,0},out[3];qrot_inv(s.gq[g],v,d);int type=M(geom_type,g);float r=M(geom_size,g*3),h=M(geom_size,g*3+1);
 if(type==2||type==3){float len=sqrtf(dot3(d,d));if(len<1e-20f)len=1;for(int k=0;k<3;k++)p[k]=r*d[k]/len;if(type==3)p[2]+=(d[2]>=0?h:-h);}
 else if(type==5){float len=sqrtf(d[0]*d[0]+d[1]*d[1]);p[0]=len>1e-20f?r*d[0]/len:r;p[1]=len>1e-20f?r*d[1]/len:0;p[2]=d[2]>=0?h:-h;}
 else if(type==6){for(int k=0;k<3;k++)p[k]=(d[k]>=0?1:-1)*M(geom_size,3*g+k);}
 else if(type==7){int mesh=M(geom_dataid,g),adr=M(mesh_vertadr,mesh),n=M(mesh_vertnum,mesh),best=0;float value=-1e30f;for(int i=0;i<n;i++){float z=0;for(int k=0;k<3;k++)z+=M(mesh_vert,3*(adr+i)+k)*d[k];if(z>value){value=z;best=i;}}for(int k=0;k<3;k++)p[k]=M(mesh_vert,3*(adr+best)+k);}
 else s.unsupported|=32;
 qrot(s.gq[g],p,out);return Vec(out)+Vec(s.gp[g]);
}
struct Vertex {Vec p,a,b;};
__device__ Vertex sup(Work&s,int w,int a,int b,Vec d){Vertex v;v.a=support(s,w,a,d);v.b=support(s,w,b,-d);v.p=v.a-v.b;return v;}
__device__ bool simplex(Vertex*v,int&n,Vec&d){
 Vec A=v[n-1].p,AO=-A;
 if(n==2){Vec AB=v[0].p-A;if(dot(AB,AO)>0){d=cr(cr(AB,AO),AB);if(dot(d,d)<1e-20f)d=perp(AB);}else{v[0]=v[1];n=1;d=AO;}return false;}
 if(n==3){Vertex av=v[2],bv=v[1],cv=v[0];Vec AB=bv.p-A,AC=cv.p-A,ABC=cr(AB,AC);
  if(dot(cr(ABC,AC),AO)>0){if(dot(AC,AO)>0){v[0]=cv;v[1]=av;n=2;d=cr(cr(AC,AO),AC);}else{v[0]=bv;v[1]=av;n=2;return simplex(v,n,d);}}
  else if(dot(cr(AB,ABC),AO)>0){v[0]=bv;v[1]=av;n=2;return simplex(v,n,d);}
  else{if(dot(ABC,AO)>0)d=ABC;else{v[0]=bv;v[1]=cv;d=-ABC;}}
  if(dot(d,d)<1e-20f)d=perp(AB);return false;
 }
 if(n==4){Vertex a=v[3],b=v[2],c=v[1],e=v[0];Vec ab=b.p-a.p,ac=c.p-a.p,ae=e.p-a.p;Vec abc=cr(ab,ac),ace=cr(ac,ae),aeb=cr(ae,ab);
  if(dot(abc,ae)>0)abc=-abc;if(dot(ace,ab)>0)ace=-ace;if(dot(aeb,ac)>0)aeb=-aeb;
  if(dot(abc,AO)>1e-12f){v[0]=c;v[1]=b;v[2]=a;n=3;return simplex(v,n,d);}
  if(dot(ace,AO)>1e-12f){v[0]=e;v[1]=c;v[2]=a;n=3;return simplex(v,n,d);}
  if(dot(aeb,AO)>1e-12f){v[0]=b;v[1]=e;v[2]=a;n=3;return simplex(v,n,d);}return true;
 }d=AO;return false;
}
#include "gjk_fallback.cuh"
struct Face {int a,b,c;Vec n;float d;bool live;};
__device__ bool face_make(Face&f,int a,int b,int c,Vertex*v){f.a=a;f.b=b;f.c=c;Vec n=cr(v[b].p-v[a].p,v[c].p-v[a].p);float len=sqrtf(dot(n,n));if(len<1e-12f){f.live=false;return false;}n=n*(1/len);float d=dot(n,v[a].p);if(d<0){int t=f.b;f.b=f.c;f.c=t;n=-n;d=-d;}f.n=n;f.d=d;f.live=true;return true;}
__device__ bool convex(Work&s,int w,int a,int b,float&distance,float*pos,float*normal){
 Vertex *v=(Vertex*)s.collision_scratch;Vec d=Vec(s.gp[b])-Vec(s.gp[a]);if(dot(d,d)<1e-12f)d=Vec(1,0,0);v[0]=sup(s,w,a,b,d);int n=1;d=-v[0].p;bool hit=false;
 for(int it=0;it<48;it++){
  if(dot(d,d)<1e-20f)d=Vec(1,.271f,.113f);Vertex p=sup(s,w,a,b,d);if(dot(p.p,d)<0)return false;v[n++]=p;if(simplex(v,n,d)){hit=true;break;}
 }
 if(!hit){int result=gjk_fallback(s,w,a,b,v);if(result<=0){if(result<0)s.unsupported|=64;return false;}hit=true;}
 Face *faces=(Face*)(v+64);int nf=4,nv=4;face_make(faces[0],0,1,2,v);face_make(faces[1],0,3,1,v);face_make(faces[2],0,2,3,v);face_make(faces[3],1,3,2,v);
 int opposite[4]={3,2,1,0};for(int f=0;f<4;f++)if(faces[f].live&&dot(faces[f].n,v[opposite[f]].p-v[faces[f].a].p)>0){int swap=faces[f].b;faces[f].b=faces[f].c;faces[f].c=swap;faces[f].n=-faces[f].n;faces[f].d=-faces[f].d;}int best=-1;
 for(int it=0;it<60;it++){
  best=-1;float mind=1e30f;for(int f=0;f<nf;f++)if(faces[f].live&&faces[f].d<mind){mind=faces[f].d;best=f;}if(best<0){s.unsupported|=128;return false;}
  Face chosen=faces[best];Vertex p=sup(s,w,a,b,chosen.n);float delta=dot(p.p,chosen.n)-chosen.d;
  bool duplicate=false;for(int i=0;i<nv;i++)if(dot(p.p-v[i].p,p.p-v[i].p)<1e-14f)duplicate=true;
  if(delta<1e-5f||duplicate||nv==64)break;
  int *ea=(int*)(faces+128),*eb=ea+96;int ne=0;
  for(int f=0;f<nf;f++)if(faces[f].live&&dot(faces[f].n,p.p-v[faces[f].a].p)>1e-7f){Face q=faces[f];faces[f].live=false;int aa[3]={q.a,q.b,q.c},bb[3]={q.b,q.c,q.a};for(int k=0;k<3;k++){bool found=false;for(int e=0;e<ne;e++)if(ea[e]==bb[k]&&eb[e]==aa[k]){ea[e]=ea[--ne];eb[e]=eb[ne];found=true;break;}if(!found){if(ne>=96){s.unsupported|=256;return false;}ea[ne]=aa[k];eb[ne++]=bb[k];}}}
  v[nv]=p;for(int e=0;e<ne;e++){int free=-1;for(int f=0;f<nf;f++)if(!faces[f].live){free=f;break;}if(free<0){if(nf>=128){s.unsupported|=256;return false;}free=nf++;}face_make(faces[free],ea[e],eb[e],nv,v);}nv++;
 }
 if(best<0)return false;Face f=faces[best];Vec A=v[f.a].p,B=v[f.b].p,C=v[f.c].p,P=f.n*f.d,ab=B-A,ac=C-A,ap=P-A;float aa=dot(ab,ab),bb=dot(ab,ac),cc=dot(ac,ac),pa=dot(ap,ab),pc=dot(ap,ac),den=aa*cc-bb*bb;
 float wb=den>1e-20f?(cc*pa-bb*pc)/den:0,wc=den>1e-20f?(aa*pc-bb*pa)/den:0,wa=1-wb-wc;
 Vec pa0=v[f.a].a*wa+v[f.b].a*wb+v[f.c].a*wc,pb0=v[f.a].b*wa+v[f.b].b*wb+v[f.c].b*wc,p=(pa0+pb0)*.5f;
 pos[0]=p.x;pos[1]=p.y;pos[2]=p.z;normal[0]=f.n.x;normal[1]=f.n.y;normal[2]=f.n.z;distance=-f.d;return distance<0;
}
__device__ void emit(Work&s,int pair,float dist,const float*p,const float*n){float fr[9];make_frame(n,fr);add_contact(s,pair,0,dist,p,fr);}
#include "poly_collision.cuh"
#include "capsule_poly.cuh"
__device__ void collide_plane(Work&s,int w,int pair,int plane,int g){
 float normal[3],point[3],frame[9];zaxis_rn(s.gq[plane],normal);make_frame(normal,frame);int type=M(geom_type,g);float radius=M(geom_size,g*3),half=M(geom_size,g*3+1);
 if(type==2){float dist=plane_point(s.gp[plane],normal,s.gp[g],radius,point);add_contact(s,pair,0,dist,point,frame);}
 else if(type==3){float axis[3];zaxis_rn(s.gq[g],axis);plane_capsule_frame(normal,axis,frame);for(int e=0;e<2;e++){float end[3];capsule_end(s.gp[g],axis,half,e,end);float dist=plane_point(s.gp[plane],normal,end,radius,point);add_contact(s,pair,e,dist,point,frame);}}
 else if(type==6||type==7){
  int mesh=type==7?M(geom_dataid,g):0,n=type==7?M(mesh_vertnum,mesh):8,adr=type==7?M(mesh_vertadr,mesh):0;float depths[4]={1e30f,1e30f,1e30f,1e30f},points[4][3];
  for(int i=0;i<n;i++){float local[3],world[3];for(int k=0;k<3;k++)local[k]=type==7?M(mesh_vert,3*(adr+i)+k):((i&(1<<k))?1:-1)*M(geom_size,3*g+k);qrot(s.gq[g],local,world);for(int k=0;k<3;k++)world[k]+=s.gp[g][k];float p[3],dist=plane_point(s.gp[plane],normal,world,0,p);int worst=0;for(int j=1;j<4;j++)if(depths[j]>depths[worst])worst=j;if(dist<depths[worst]){depths[worst]=dist;copy3(points[worst],p);}}
  for(int i=0;i<4;i++)if(depths[i]<0)add_contact(s,pair,i,depths[i],points[i],frame);
 }else if(type==5){
  float localn[3];qrot_inv(s.gq[g],normal,localn);float lateral=sqrtf(localn[0]*localn[0]+localn[1]*localn[1]);
  // MuJoCo's plane/cylinder manifold: deepest rim point and two additional rim points.
  int count=lateral<1e-6f?3:(fabsf(localn[2])<1e-6f?2:1);
  for(int i=0;i<count;i++){float local[3];if(lateral<1e-6f){float theta=2.094395102f*i;local[0]=radius*cosf(theta);local[1]=radius*sinf(theta);}else{local[0]=-radius*localn[0]/lateral;local[1]=-radius*localn[1]/lateral;}local[2]=count==2?(i?half:-half):(localn[2]>0?-half:half);float p[3];qrot(s.gq[g],local,p);for(int k=0;k<3;k++)p[k]+=s.gp[g][k];float dist=plane_point(s.gp[plane],normal,p,0,point);add_contact(s,pair,i,dist,point,frame);}
 }else s.unsupported|=32;
}
__device__ void collisions(Work&s,int w){
 for(int pair=0;pair<NP;pair++){
  int a=PAIRS[2*pair],b=PAIRS[2*pair+1],ta=M(geom_type,a),tb=M(geom_type,b),start=s.nc,old_error=s.unsupported;bool swapped=false;
  if(ta>tb){int tmp=a;a=b;b=tmp;tmp=ta;ta=tb;tb=tmp;swapped=true;}
  if(ta!=0){Vec diff=Vec(s.gp[a])-Vec(s.gp[b]);float bound=M(geom_rbound,a)+M(geom_rbound,b);if(dot(diff,diff)>bound*bound)continue;}
  float p[3],n[3],dist;
  if(ta==0)collide_plane(s,w,pair,a,b);
  else if(ta==2&&tb==2){sphere_sphere(s.gp[a],M(geom_size,3*a),s.gp[b],M(geom_size,3*b),&dist,p,n);emit(s,pair,dist,p,n);}
  else if(ta==2&&tb==3){float axis[3],v0[3],ab[3],ap[3],gam,t,pt[3];zaxis_rn(s.gq[b],axis);segment_point(s.gp[a],s.gp[b],axis,M(geom_size,3*b+1),v0,ab,ap,&gam,&t,pt);sphere_sphere(s.gp[a],M(geom_size,3*a),pt,M(geom_size,3*b),&dist,p,n);emit(s,pair,dist,p,n);}
  else if(ta==3&&tb==3){CapCap cc;capcap_setup(s.gp[a],s.gq[a],M(geom_size,3*a+1),s.gp[b],s.gq[b],M(geom_size,3*b+1),cc);bool f1,f2;float raw,x1,x2;if(fabsf(cc.det)>=1e-15f){capcap_params(cc,&x1,&x2,&f1,&f2,&raw);capcap_core(s.gp[a],s.gp[b],cc,x1,x2,M(geom_size,3*a),M(geom_size,3*b),&dist,p,n);emit(s,pair,dist,p,n);}else for(int e=0;e<4;e++){capcap_candidate(cc,e,&x1,&x2,&f1,&f2,&raw);capcap_core(s.gp[a],s.gp[b],cc,x1,x2,M(geom_size,3*a),M(geom_size,3*b),&dist,p,n);emit(s,pair,dist,p,n);}}
  else if(ta==2&&tb==6){float R[9],sz[3];quat2mat(s.gq[b],R);for(int k=0;k<3;k++)sz[k]=M(geom_size,b*3+k);sphere_box(s.gp[a],M(geom_size,3*a),s.gp[b],R,sz,&dist,p,n);emit(s,pair,dist,p,n);}
  else if(ta==3&&tb==6){float R[9],sz[3],axis[3];quat2mat(s.gq[b],R);zaxis_rn(s.gq[a],axis);for(int k=0;k<3;k++)sz[k]=M(geom_size,b*3+k);capsule_box(s,pair,s.gp[a],axis,M(geom_size,a*3),M(geom_size,a*3+1),s.gp[b],R,sz);}
  else if(ta==6&&tb==6){float R1[9],R2[9],sz1[3],sz2[3];quat2mat(s.gq[a],R1);quat2mat(s.gq[b],R2);for(int k=0;k<3;k++){sz1[k]=M(geom_size,a*3+k);sz2[k]=M(geom_size,b*3+k);}box_box(s,pair,s.gp[a],R1,sz1,s.gp[b],R2,sz2);}
  else if((ta==6||ta==7)&&(tb==6||tb==7))poly_collision(s,w,pair,a,b);
  else{
   int before=s.unsupported;
   if(convex(s,w,a,b,dist,p,n))emit(s,pair,dist,p,n);
   else if(ta==3&&tb==7&&((s.unsupported&~before)&448)){
    if(capsule_poly_fallback(s,w,pair,a,b)){s.unsupported=(s.unsupported&~448)|before;s.resolved_pairs[pair]++;}
   }
  }
  if(swapped)for(int c=start;c<s.nc;c++)for(int k=0;k<3;k++){s.con[c].fr[k]*=-1;s.con[c].fr[6+k]*=-1;}
  if(s.unsupported&~old_error)s.failed_pairs[pair]|=s.unsupported&~old_error;
 }
}
