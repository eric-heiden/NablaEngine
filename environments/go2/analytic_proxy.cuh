// Task-specific authored shape families. Rounded convex meshes become analytic
// sphere/capsules; their original masses and inertias are deliberately retained.
struct Proxy {int type;float p[3],q[4],size[3];};
__device__ Proxy proxy_geometry(Work&s,int g){
 Proxy o;o.type=reduced_type(s,g);copy3(o.p,s.gp[g]);for(int k=0;k<4;k++)o.q[k]=s.gq[g][k];
 for(int k=0;k<3;k++)o.size[k]=M(geom_size,3*g+k);
 if(M(geom_type,g)==7){int mesh=M(geom_dataid,g);float center[3],offset[3];for(int k=0;k<3;k++)center[k]=proxy_center(3*mesh+k);qrot(o.q,center,offset);for(int k=0;k<3;k++){o.p[k]+=offset[k];o.size[k]=proxy_half(3*mesh+k);}
  if(o.type==2||o.type==3){o.size[0]=proxy_radius(mesh);o.size[1]=proxy_segment(mesh);o.size[2]=0;}
 }
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
__device__ void collisions(Work&s,int w){
 for(int pair=threadIdx.x;pair<NP;pair+=32){
  int ga=PAIRS[2*pair],gb=PAIRS[2*pair+1];Proxy a=proxy_geometry(s,ga),b=proxy_geometry(s,gb);
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
  else if(a.type==6&&b.type==6){float Ra[9],Rb[9];quat2mat(a.q,Ra);quat2mat(b.q,Rb);box_box(s,pair,a.p,Ra,a.size,b.p,Rb,b.size);}
  else atomicOr(&s.unsupported,32);
 }
}
