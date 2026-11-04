// Exact separating-axis tests and face clipping for boxes and convex meshes.
__device__ int poly_nfaces(Work&s,int w,int g){return M(geom_type,g)==6?6:MF_START[M(geom_dataid,g)+1]-MF_START[M(geom_dataid,g)];}
__device__ int poly_nedges(Work&s,int w,int g){return M(geom_type,g)==6?12:ME_START[M(geom_dataid,g)+1]-ME_START[M(geom_dataid,g)];}
__device__ Vec poly_vertex(Work&s,int w,int g,int v){float p[3],out[3];if(M(geom_type,g)==6){for(int k=0;k<3;k++)p[k]=(v&(1<<k)?1:-1)*M(geom_size,3*g+k);}else{int adr=M(mesh_vertadr,M(geom_dataid,g));for(int k=0;k<3;k++)p[k]=M(mesh_vert,3*(adr+v)+k);}qrot(s.gq[g],p,out);return Vec(out)+Vec(s.gp[g]);}
__device__ Vec poly_normal(Work&s,int w,int g,int f){float n[3]={0,0,0},out[3];if(M(geom_type,g)==6)n[f/2]=f%2?1:-1;else{int adr=MF_START[M(geom_dataid,g)]+f;for(int k=0;k<3;k++)n[k]=MF_NORMAL[3*adr+k];}qrot(s.gq[g],n,out);return Vec(out);}
__device__ int poly_face(Work&s,int w,int g,int f,Vec*verts){
 if(M(geom_type,g)==6){int axis=f/2,sign=f%2,u=(axis+1)%3,v=(axis+2)%3;int idx[4]={0,1,3,2};for(int i=0;i<4;i++){int order=sign?i:3-i,corner=(sign<<axis)|((idx[order]&1)<<u)|(((idx[order]>>1)&1)<<v);verts[i]=poly_vertex(s,w,g,corner);}return 4;}
 int adr=MF_START[M(geom_dataid,g)]+f,n=MF_VERT[17*adr];for(int i=0;i<n;i++)verts[i]=poly_vertex(s,w,g,MF_VERT[17*adr+1+i]);return n;
}
__device__ Vec poly_edge(Work&s,int w,int g,int e){int a,b;if(M(geom_type,g)==6){int axis=e/4,u=(axis+1)%3,v=(axis+2)%3,base=((e%4&1)<<u)|(((e%4>>1)&1)<<v);a=base;b=base|(1<<axis);}else{int adr=ME_START[M(geom_dataid,g)]+e;a=ME_VERT[2*adr];b=ME_VERT[2*adr+1];}return poly_vertex(s,w,g,b)-poly_vertex(s,w,g,a);}
__device__ bool sat_axis(Work&s,int w,int a,int b,Vec axis,float&best,Vec&normal){
 float n2=dot(axis,axis);if(n2<1e-12f)return true;axis=axis*rsqrtf(n2);
 float sep1=dot(support(s,w,b,-axis)-support(s,w,a,axis),axis),sep2=dot(support(s,w,a,-axis)-support(s,w,b,axis),axis);
 if(sep1>1e-7f||sep2>1e-7f)return false;
 if(sep1>best){best=sep1;normal=axis;}if(sep2>best){best=sep2;normal=-axis;}return true;
}
__device__ void closest_segments(Vec p,Vec q,Vec r,Vec t,Vec&pa,Vec&pb){
 Vec d1=q-p,d2=t-r,rr=p-r;float a=dot(d1,d1),e=dot(d2,d2),f=dot(d2,rr),u=0,v=0;
 if(a<=1e-16f){v=e>1e-16f?clampf(f/e,0,1):0;}
 else{float c=dot(d1,rr);if(e<=1e-16f)u=clampf(-c/a,0,1);else{float b=dot(d1,d2),den=a*e-b*b;if(den>1e-16f)u=clampf((b*f-c*e)/den,0,1);v=(b*u+f)/e;if(v<0){v=0;u=clampf(-c/a,0,1);}else if(v>1){v=1;u=clampf((b-c)/a,0,1);}}}
 pa=p+d1*u;pb=r+d2*v;
}
__device__ void support_segment(Work&s,int w,int g,Vec n,Vec&p,Vec&q){
 p=support(s,w,g,n);q=p;float extreme=dot(p,n),best=0;int nv=M(geom_type,g)==6?8:M(mesh_vertnum,M(geom_dataid,g));
 for(int i=0;i<nv;i++){Vec v=poly_vertex(s,w,g,i);if(extreme-dot(v,n)<2e-5f){float d=dot(v-p,v-p);if(d>best){best=d;q=v;}}}
}
__device__ void sat_contact(Work&s,int w,int pair,int a,int b,float depth,Vec n){
 Vec a0,a1,b0,b1,pa,pb;support_segment(s,w,a,n,a0,a1);support_segment(s,w,b,-n,b0,b1);closest_segments(a0,a1,b0,b1,pa,pb);Vec mid=(pa+pb)*.5f;
 float p[3]={mid.x,mid.y,mid.z},normal[3]={n.x,n.y,n.z};emit(s,pair,depth,p,normal);
}
__device__ void poly_collision(Work&s,int w,int pair,int a,int b){
 float point0[3],n0[3],best;int old_flags=s.unsupported;
 bool hit=convex(s,w,a,b,best,point0,n0);if((threadIdx.x&31)!=0)return;Vec normal(n0);
 int na=poly_nfaces(s,w,a),nb=poly_nfaces(s,w,b);
 if(!hit){
  if(s.unsupported==old_flags)return;
  // Resolve a degenerate GJK simplex independently with the exhaustive SAT test.
  best=-1e30f;
  for(int f=0;f<na;f++)if(!sat_axis(s,w,a,b,poly_normal(s,w,a,f),best,normal)){s.unsupported=old_flags;return;}
  for(int f=0;f<nb;f++)if(!sat_axis(s,w,a,b,poly_normal(s,w,b,f),best,normal)){s.unsupported=old_flags;return;}
  int ea=poly_nedges(s,w,a),eb=poly_nedges(s,w,b);
  for(int e=0;e<ea;e+=(M(geom_type,a)==6?4:1))for(int f=0;f<eb;f+=(M(geom_type,b)==6?4:1))if(!sat_axis(s,w,a,b,cr(poly_edge(s,w,a,e),poly_edge(s,w,b,f)),best,normal)){s.unsupported=old_flags;return;}
 }
 int fa=0,fb=0;float da=-2,db=-2;
 for(int f=0;f<na;f++){float d=dot(poly_normal(s,w,a,f),normal);if(d>da){da=d;fa=f;}}
 for(int f=0;f<nb;f++){float d=dot(poly_normal(s,w,b,f),-normal);if(d>db){db=d;fb=f;}}
 if(fmaxf(da,db)<.98f){if(hit)emit(s,pair,best,point0,n0);else{sat_contact(s,w,pair,a,b,best,normal);s.unsupported=old_flags;}return;}
 int ref=da>=db?a:b,inc=da>=db?b:a,rf=da>=db?fa:fb,inf=da>=db?fb:fa;Vec rn=poly_normal(s,w,ref,rf),*rv=(Vec*)s.collision_scratch,*poly=rv+32,*tmp=poly+48;int nr=poly_face(s,w,ref,rf,rv),np=poly_face(s,w,inc,inf,poly);
 for(int edge=0;edge<nr&&np;edge++){
  Vec v0=rv[edge],side=cr(rn,rv[(edge+1)%nr]-v0);int count=0;
  for(int i=0;i<np;i++){Vec p=poly[i],q=poly[(i+1)%np];float d1=dot(side,p-v0),d2=dot(side,q-v0);bool in1=d1>=-1e-8f,in2=d2>=-1e-8f;if(in1&&count<48)tmp[count++]=p;if(in1!=in2&&count<48)tmp[count++]=p+(q-p)*(d1/(d1-d2));}
  np=count;for(int i=0;i<np;i++)poly[i]=tmp[i];
 }
 Vec *pts=tmp+48;float *depth=(float*)(pts+48);int npts=0;for(int i=0;i<np;i++){float d=dot(rn,poly[i]-rv[0]);if(d<0){pts[npts]=poly[i]-rn*(.5f*d);depth[npts++]=d;}}
 if(npts==0){if(hit)emit(s,pair,best,point0,n0);else{sat_contact(s,w,pair,a,b,best,normal);s.unsupported=old_flags;}return;}
 s.unsupported=old_flags;
 // Keep at most four spatially separated manifold points.
 bool *used=(bool*)(depth+48);for(int i=0;i<48;i++)used[i]=false;int chosen[4],count=min(npts,4);float min_depth=1e30f;int first=0;for(int i=0;i<npts;i++)if(depth[i]<min_depth){min_depth=depth[i];first=i;}chosen[0]=first;used[first]=true;
 for(int j=1;j<count;j++){float maxsep=-1;int next=0;for(int i=0;i<npts;i++)if(!used[i]){float mind=1e30f;for(int k=0;k<j;k++)mind=fminf(mind,dot(pts[i]-pts[chosen[k]],pts[i]-pts[chosen[k]]));if(mind>maxsep){maxsep=mind;next=i;}}chosen[j]=next;used[next]=true;}
 float n[3]={normal.x,normal.y,normal.z};for(int j=0;j<count;j++){int i=chosen[j];float p[3]={pts[i].x,pts[i].y,pts[i].z};emit(s,pair,depth[i],p,n);}
}
