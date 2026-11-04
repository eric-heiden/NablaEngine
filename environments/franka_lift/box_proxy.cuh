// Convex meshes retain their mass, inertia, pose, friction and randomization.
// Only their collision surface becomes an oriented local bounding box.
__device__ void proxy_box(Work&s,int g,float*p,float*R,float*sz){
 quat2mat(s.gq[g],R);copy3(p,s.gp[g]);
 if(M(geom_type,g)==7){int mesh=M(geom_dataid,g);float c[3],d[3];for(int k=0;k<3;k++){c[k]=proxy_center(3*mesh+k);sz[k]=proxy_half(3*mesh+k);}qrot(s.gq[g],c,d);for(int k=0;k<3;k++)p[k]+=d[k];}
 else for(int k=0;k<3;k++)sz[k]=M(geom_size,3*g+k);
}
__device__ void proxy_plane(Work&s,int w,int pair,int plane,int g){
 if(M(geom_type,g)!=7){collide_plane(s,w,pair,plane,g);return;}
 float p[3],R[9],sz[3],normal[3],frame[9];proxy_box(s,g,p,R,sz);zaxis_rn(s.gq[plane],normal);make_frame(normal,frame);
 float depths[4]={1e30f,1e30f,1e30f,1e30f},points[4][3];
 for(int i=0;i<8;i++){float world[3];for(int k=0;k<3;k++){world[k]=p[k];for(int j=0;j<3;j++)world[k]+=R[3*k+j]*((i&(1<<j))?sz[j]:-sz[j]);}float point[3],dist=plane_point(s.gp[plane],normal,world,0,point);int worst=0;for(int j=1;j<4;j++)if(depths[j]>depths[worst])worst=j;if(dist<depths[worst]){depths[worst]=dist;copy3(points[worst],point);}}
 for(int i=0;i<4;i++)if(depths[i]<0)add_contact(s,pair,i,depths[i],points[i],frame);
}
__device__ void collisions(Work&s,int w){
 for(int pair=threadIdx.x;pair<NP;pair+=32){
  int a=PAIRS[2*pair],b=PAIRS[2*pair+1],ta=M(geom_type,a),tb=M(geom_type,b);
  if(ta>tb){int t=a;a=b;b=t;t=ta;ta=tb;tb=t;}
  if(ta==0){proxy_plane(s,w,pair,a,b);continue;}
  float d[3];for(int k=0;k<3;k++)d[k]=s.gp[a][k]-s.gp[b][k];float bound=M(geom_rbound,a)+M(geom_rbound,b);if(dot3(d,d)>bound*bound)continue;
  float pa[3],pb[3],Ra[9],Rb[9],sa[3],sb[3];proxy_box(s,a,pa,Ra,sa);proxy_box(s,b,pb,Rb,sb);
  if(ta==3){float axis[3];zaxis_rn(s.gq[a],axis);capsule_box(s,pair,pa,axis,M(geom_size,3*a),M(geom_size,3*a+1),pb,Rb,sb);}
  else box_box(s,pair,pa,Ra,sa,pb,Rb,sb);
 }
}
