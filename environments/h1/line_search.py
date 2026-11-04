"""Safeguarded Newton root of the existing piecewise-linear line derivative."""
def transform(src):
 old='''  for(int k=0;k<24;k++){float alpha=.5f*(v.lo+v.hi),df=0;for(int r=l;r<n;r+=32){float f,d;row_eval(v.x[r]+alpha*v.jp[r],s.D[r],s.fl[r],r,s.ne,s.nf,f,d);df-=f*v.jp[r];}df=sumwarp(df)+v.gd+alpha*v.pmp;if(l==0){if(df<0)v.lo=alpha;else v.hi=alpha;}__syncwarp();}
  for(int i=l;i<NV;i+=32)v.a[i]+=.5f*(v.lo+v.hi)*v.p[i];'''
 new='''  float alpha=.5f*(v.lo+v.hi);
  for(int k=0;k<24;k++){
   float df=0,curv=0;for(int r=l;r<n;r+=32){float f,d;row_eval(v.x[r]+alpha*v.jp[r],s.D[r],s.fl[r],r,s.ne,s.nf,f,d);df-=f*v.jp[r];curv+=d*v.jp[r]*v.jp[r];}
   df=sumwarp(df)+v.gd+alpha*v.pmp;curv=sumwarp(curv)+v.pmp;
   if(fabsf(df)<=1e-7f*(fabsf(v.gd)+fabsf(alpha*v.pmp)+1.f))break;
   if(l==0){if(df<0)v.lo=alpha;else v.hi=alpha;}__syncwarp();
   float next=alpha-df/fmaxf(curv,1e-30f);
   if(!(next>v.lo&&next<v.hi))next=.5f*(v.lo+v.hi);
   if(next==alpha)break;alpha=next;
  }
  for(int i=l;i<NV;i+=32)v.a[i]+=alpha*v.p[i];'''
 assert old in src
 return src.replace(old,new).replace('for(int it=0;it<ITERS;it++){','for(int it=0;it<ITERS && n>0;it++){').replace('if(it>0&&v.stop)break;','if(v.stop)break;')
