// Closest-simplex GJK fallback for near-degenerate support directions.
// Double precision is confined to the tiny barycentric systems in this fallback.
__device__ Vec closest_simplex(Vertex*v,int&n){
 double best=1e100;int maskbest=1;double bestw[4]={1,0,0,0};
 for(int mask=1;mask<(1<<n);mask++){
  int ids[4],k=0;for(int i=0;i<n;i++)if(mask&(1<<i))ids[k++]=i;
  double weights[4]={1,0,0,0};Vec base=v[ids[0]].p;int dim=k-1;double A[3][4]={{0}};bool valid=true;
  for(int i=0;i<dim;i++){
   Vec ei=v[ids[i+1]].p-base;
   for(int j=0;j<dim;j++){Vec ej=v[ids[j+1]].p-base;A[i][j]=(double)ei.x*ej.x+(double)ei.y*ej.y+(double)ei.z*ej.z;}
   A[i][dim]=-((double)ei.x*base.x+(double)ei.y*base.y+(double)ei.z*base.z);
  }
  for(int col=0;col<dim;col++){
   int pivot=col;for(int r=col+1;r<dim;r++)if(fabs(A[r][col])>fabs(A[pivot][col]))pivot=r;
   if(fabs(A[pivot][col])<1e-24){valid=false;break;}
   for(int j=col;j<=dim;j++){double tmp=A[col][j];A[col][j]=A[pivot][j];A[pivot][j]=tmp;}
   double divisor=A[col][col];for(int j=col;j<=dim;j++)A[col][j]/=divisor;
   for(int r=0;r<dim;r++)if(r!=col){double mul=A[r][col];for(int j=col;j<=dim;j++)A[r][j]-=mul*A[col][j];}
  }
  if(!valid)continue;
  for(int i=0;i<dim;i++){weights[i+1]=A[i][dim];weights[0]-=weights[i+1];}
  double x=0,y=0,z=0;for(int i=0;i<k;i++){if(weights[i]<-1e-10)valid=false;x+=weights[i]*v[ids[i]].p.x;y+=weights[i]*v[ids[i]].p.y;z+=weights[i]*v[ids[i]].p.z;}
  double norm=x*x+y*y+z*z;
  if(valid&&norm<best){best=norm;maskbest=mask;for(int i=0;i<4;i++)bestw[i]=0;for(int i=0;i<k;i++)bestw[ids[i]]=weights[i];}
 }
 Vertex out[4];int k=0;Vec closest;for(int i=0;i<n;i++)if(maskbest&(1<<i)){out[k++]=v[i];closest=closest+v[i].p*(float)bestw[i];}n=k;for(int i=0;i<n;i++)v[i]=out[i];return closest;
}
__device__ int gjk_fallback(Work&s,int w,int a,int b,Vertex*v){
 Vec direction=Vec(s.gp[b])-Vec(s.gp[a]);if(dot(direction,direction)<1e-12f)direction=Vec(1,0,0);
 v[0]=sup(s,w,a,b,direction);int n=1;Vec closest=v[0].p;
 for(int iteration=0;iteration<96;iteration++){
  float dist2=dot(closest,closest);
  if(dist2<1e-16f){
   if(n==4)return 1;
   direction=n==1?Vec(1,0,0):n==2?perp(v[1].p-v[0].p):cr(v[1].p-v[0].p,v[2].p-v[0].p);
   if(dot(direction,direction)<1e-20f)direction=Vec(0,1,0);
   Vertex p=sup(s,w,a,b,direction);
   if(n==3&&fabsf(dot(direction,p.p-v[0].p))<1e-15f)p=sup(s,w,a,b,-direction);
   v[n++]=p;continue;
  }
  direction=-closest;Vertex p=sup(s,w,a,b,direction);
  if(dot(p.p,direction)<0)return 0;
  float gain=dot(p.p-closest,direction);if(gain<=1e-12f*sqrtf(dist2))return 0;
  bool duplicate=false;for(int i=0;i<n;i++)if(dot(p.p-v[i].p,p.p-v[i].p)<1e-20f)duplicate=true;if(duplicate)return 0;
  if(n>=4)return -1;
  v[n++]=p;closest=closest_simplex(v,n);
 }
 return -1;
}
