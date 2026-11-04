// Non-iterative fallback for a capsule against a convex polyhedron.
// Outside: exact closest segment/triangle features, then subtract capsule radius.
// Intersecting centerline: SAT of the segment-extruded polyhedron, dilated by radius.
__device__ Vec triangle_closest(Vec p,Vec a,Vec b,Vec c){
 Vec ab=b-a,ac=c-a,ap=p-a;float d1=dot(ab,ap),d2=dot(ac,ap);if(d1<=0&&d2<=0)return a;
 Vec bp=p-b;float d3=dot(ab,bp),d4=dot(ac,bp);if(d3>=0&&d4<=d3)return b;
 float vc=d1*d4-d3*d2;if(vc<=0&&d1>=0&&d3<=0)return a+ab*(d1/(d1-d3));
 Vec cp=p-c;float d5=dot(ab,cp),d6=dot(ac,cp);if(d6>=0&&d5<=d6)return c;
 float vb=d5*d2-d1*d6;if(vb<=0&&d2>=0&&d6<=0)return a+ac*(d2/(d2-d6));
 float va=d3*d6-d5*d4;if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0)return b+(c-b)*((d4-d3)/((d4-d3)+(d5-d6)));
 float den=va+vb+vc;
 if(fabsf(den)<1e-24f){Vec pa,pb;closest_segments(p,p,a,b,pa,pb);Vec best=pb;float d=dot(p-best,p-best);closest_segments(p,p,b,c,pa,pb);if(dot(p-pb,p-pb)<d){best=pb;d=dot(p-best,p-best);}closest_segments(p,p,c,a,pa,pb);if(dot(p-pb,p-pb)<d)best=pb;return best;}
 return a+ab*(vb/den)+ac*(vc/den);
}
__device__ void closest_update(Vec a,Vec b,float&best,Vec&pa,Vec&pb){float d=dot(a-b,a-b);if(d<best){best=d;pa=a;pb=b;}}
__device__ bool capsule_poly_fallback(Work&s,int w,int pair,int a,int b){
 float axis0[3];zaxis_rn(s.gq[a],axis0);Vec axis(axis0),center(s.gp[a]);float half=M(geom_size,3*a+1),radius=M(geom_size,3*a);
 Vec p=center-axis*half,q=center+axis*half;int nf=poly_nfaces(s,w,b);if(nf<=0)return false;
 bool intersects=true;float enter=0,leave=1,best=1e30f;Vec near_a,near_b;
 for(int f=0;f<nf;f++){
  Vec vertices[32],normal=poly_normal(s,w,b,f);int nv=poly_face(s,w,b,f,vertices);if(nv<3)return false;
  float d0=dot(normal,p-vertices[0]),d1=dot(normal,q-vertices[0]);
  if(d0>0&&d1>0)intersects=false;
  else if((d0>0)!=(d1>0)){float t=d0/(d0-d1);if(d0>0)enter=fmaxf(enter,t);else leave=fminf(leave,t);}
  for(int j=1;j+1<nv;j++){
   Vec aa=vertices[0],bb=vertices[j],cc=vertices[j+1];closest_update(p,triangle_closest(p,aa,bb,cc),best,near_a,near_b);closest_update(q,triangle_closest(q,aa,bb,cc),best,near_a,near_b);
   Vec va,vb;closest_segments(p,q,aa,bb,va,vb);closest_update(va,vb,best,near_a,near_b);closest_segments(p,q,bb,cc,va,vb);closest_update(va,vb,best,near_a,near_b);closest_segments(p,q,cc,aa,va,vb);closest_update(va,vb,best,near_a,near_b);
  }
 }
 intersects=intersects&&enter<=leave;
 Vec normal,point;float depth;
 if(!intersects&&best>1e-16f){
  float distance=sqrtf(best);if(distance>=radius)return true;normal=(near_b-near_a)*(1/distance);depth=distance-radius;point=(near_a+normal*radius+near_b)*.5f;
 }else{
  depth=-1e30f;
  for(int f=0;f<nf;f++)if(!sat_axis(s,w,a,b,poly_normal(s,w,b,f),depth,normal))return true;
  for(int e=0;e<poly_nedges(s,w,b);e++)if(!sat_axis(s,w,a,b,cr(axis,poly_edge(s,w,b,e)),depth,normal))return true;
  Vec b0,b1,ca,cb;support_segment(s,w,b,-normal,b0,b1);closest_segments(p,q,b0,b1,ca,cb);
  float projection=dot(axis,normal);if(projection>1e-6f)ca=q;else if(projection<-1e-6f)ca=p;
  point=ca+normal*(radius+.5f*depth);
 }
 if(!isfinite(depth)||!isfinite(point.x)||!isfinite(point.y)||!isfinite(point.z))return false;
 float outp[3]={point.x,point.y,point.z},outn[3]={normal.x,normal.y,normal.z};emit(s,pair,depth,outp,outn);return true;
}
