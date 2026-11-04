"""Cooperative Newton linear solve and lower-triangular Hessian assembly."""
def transform(src):
 a=src.index('  for(int j=0;j<NV;j++){if(l==0){float z=v.H[')
 b=src.index('float dec=0;',a)
 src=src[:a]+'  if(l<NV)v.p[l]=-v.g[l];__syncwarp();chol_parallel(v.H,v.p,l);\n  if(l==0){'+src[b:]
 old='float norm=sqrtf(sumwarp(gg));'
 src=src.replace(old,old+'if(norm<M(tolerance,0)*M(meaninertia,0)*NV)break;')
 old='int i=t/NV,j=t%NV;float h=s.mass[t];for(int r=0;r<n;r++)h+=s.J[r*NV+i]*v.diag[r]*s.J[r*NV+j];v.H[t]=h;'
 new='int i=t/NV,j=t%NV;if(j<=i){float h=s.mass[t];for(int r=0;r<n;r++)h+=s.J[r*NV+i]*v.diag[r]*s.J[r*NV+j];v.H[t]=h;}'
 assert old in src
 return src.replace(old,new)
