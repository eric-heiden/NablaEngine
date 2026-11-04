// Franka reach has only ground/robot and static table/robot pairs.
__device__ void collisions(Work&s,int w){
 for(int pair=0;pair<NP;pair++){
  int a=PAIRS[2*pair],b=PAIRS[2*pair+1],start=s.nc;
  if(a==0){collide_plane(s,w,pair,a,b);continue;}
  float diff[3];for(int k=0;k<3;k++)diff[k]=s.gp[a][k]-s.gp[b][k];float bound=M(geom_rbound,a)+M(geom_rbound,b);if(dot3(diff,diff)>bound*bound)continue;
  float R1[9],R2[9],sz1[3],sz2[3],axis[3];quat2mat(s.gq[a],R1);quat2mat(s.gq[b],R2);for(int k=0;k<3;k++){sz1[k]=M(geom_size,a*3+k);sz2[k]=M(geom_size,b*3+k);}
  if(M(geom_type,b)==3){zaxis_rn(s.gq[b],axis);capsule_box(s,pair,s.gp[b],axis,sz2[0],sz2[1],s.gp[a],R1,sz1);for(int c=start;c<s.nc;c++)for(int k=0;k<3;k++){s.con[c].fr[k]*=-1;s.con[c].fr[6+k]*=-1;}}
  else box_box(s,pair,s.gp[a],R1,sz1,s.gp[b],R2,sz2);
 }
}
