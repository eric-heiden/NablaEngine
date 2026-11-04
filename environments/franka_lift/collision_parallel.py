from .parallelize import block
from pathlib import Path
R=Path(__file__).parent

def transform(src,tag):
 # Concurrent primitive emissions reserve a slot, then sort by original pair order.
 old='if(dist>=0)return;if(s.nc>=NC){s.unsupported|=4;return;}Contact &c=s.con[s.nc++];'
 assert old in src
 src=src.replace(old,'if(dist>=0)return;int id=atomicAdd(&s.nc,1);if(id>=NC){atomicOr(&s.unsupported,4);return;}Contact &c=s.con[id];c.row=pair;')
 src=src.replace('for(int k=0;k<9;k++)c.fr[k]=frame[k];','for(int k=0;k<9;k++)c.fr[k]=frame[k]*((M(geom_type,PAIRS[2*pair])>M(geom_type,PAIRS[2*pair+1])&&(k<3||k>=6))?-1:1);')
 if tag=='go2':
  src=src.replace('for(int pair=0;pair<NP;pair++)collide_plane','for(int pair=threadIdx.x;pair<NP;pair+=32)collide_plane')
 elif tag=='reach':
  src=src.replace('for(int pair=0;pair<NP;pair++){','for(int pair=threadIdx.x;pair<NP;pair+=32){')
  old='for(int c=start;c<s.nc;c++)for(int k=0;k<3;k++){s.con[c].fr[k]*=-1;s.con[c].fr[6+k]*=-1;}'
  assert old in src;src=src.replace(old,'')
 else:
  collision=(R/'collision.cuh').read_text()
  a=collision.index('__device__ bool convex(');b,z=block(collision,a)
  collision=collision[:a]+(R/'convex_parallel.cuh').read_text()+collision[z:]
  # Primitive pairs execute lane-parallel; convex pairs then use the whole warp.
  a=collision.index('__device__ void collisions(');b,z=block(collision,a)
  f=collision[b+1:z-1]
  loopstart=f.index(' for(int pair=0;');lb,lz=block(f,loopstart);body=f[lb+1:lz-1]
  tailstart=body.index('  if(swapped)');body=body[:tailstart]
  split=body.index('  else if((ta==6||ta==7)')
  prim=body[:split]+'\n'
  prim=prim.replace('  float p[3],n[3],dist;', '  if(ta!=0&&tb==7)continue;float p[3],n[3],dist;')
  complexbody=body[:body.index('  float p[3],n[3],dist;')]+'''  if(ta==0||tb!=7)continue;
  float p[3],n[3],dist;
  if(ta==6||ta==7)poly_collision(s,w,pair,a,b);
  else{int before=s.unsupported;bool hit=convex(s,w,a,b,dist,p,n);if(threadIdx.x==0){
   if(hit)emit(s,pair,dist,p,n);
   else if(ta==3&&((s.unsupported&~before)&448)){if(capsule_poly_fallback(s,w,pair,a,b)){s.unsupported=(s.unsupported&~448)|before;s.resolved_pairs[pair]++;}}
  }}
  __syncwarp();if(threadIdx.x==0&&s.unsupported&~old_error)s.failed_pairs[pair]|=s.unsupported&~old_error;
  __syncwarp();
'''
  new='\n for(int pair=threadIdx.x;pair<NP;pair+=32){'+prim+'}\n __syncwarp();\n for(int pair=0;pair<NP;pair++){'+complexbody+'}\n'
  collision=collision[:b+1]+new+collision[z-1:]
  src=src.replace('#include "collision.cuh"',collision)
 # prepare() calls collision and constraint stages collectively.
 old='if(l==0){MARK(3);collisions(s,w);MARK(4);constraints(s,dd,w);MARK(5);}__syncwarp();'
 new='''if(l==0){MARK(3);}collisions(s,w);__syncwarp();if(l==0){s.nc=min(s.nc,NC);
 for(int i=1;i<s.nc;i++){Contact v=s.con[i];int j=i;while(j>0&&s.con[j-1].row>v.row){s.con[j]=s.con[j-1];j--;}s.con[j]=v;}MARK(4);}
 __syncwarp();constraints(s,dd,w);__syncwarp();if(l==0){MARK(5);}__syncwarp();'''
 assert old in src;src=src.replace(old,new)
 # Equality/friction/limit rows retain their prefix. Contacts own disjoint row spans.
 src=src.replace('float fl=0){','float fl=0,int fixed_row=-1){')
 src=src.replace('if(s.nr>=NR){s.unsupported|=2;return;}int r=s.nr++;','int r=fixed_row>=0?fixed_row:s.nr++;if(r>=NR){atomicOr(&s.unsupported,2);return;}')
 a=src.index('__device__ void constraints(');b,z=block(src,a);f=src[b+1:z-1];cut=f.index(' for(int cidx=0;')
 prefix=f[:cut];contacts=f[cut:]
 contacts=contacts.replace('for(int cidx=0;cidx<s.nc;cidx++)','for(int cidx=threadIdx.x;cidx<s.nc;cidx+=32)').replace('c.row=s.nr;','')
 contacts=contacts.replace('row_add(s,dd,w,j,c.dist,iw,ref,imp);','row_add(s,dd,w,j,c.dist,iw,ref,imp,0,c.row+edge);')
 f='\n if(threadIdx.x==0){'+prefix+'''
 for(int cidx=0;cidx<s.nc;cidx++){Contact&c=s.con[cidx];int a=c.g1,b=c.g2,pa=M(geom_priority,a),pb=M(geom_priority,b);int dim=pa==pb?max(M(geom_condim,a),M(geom_condim,b)):M(geom_condim,pa>pb?a:b);c.row=s.nr;s.nr+=(dim==1?1:4);}
 if(s.nr>NR){s.unsupported|=2;s.nr=NR;}
 }__syncwarp();
'''+contacts+'\n__syncwarp();\n'
 src=src[:b+1]+f+src[z-1:]
 return src
