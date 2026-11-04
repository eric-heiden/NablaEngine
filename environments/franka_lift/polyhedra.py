"""Compile the actual convex-mesh topology, retaining live vertex coordinates."""
import numpy as np
from scipy.spatial import ConvexHull
def declarations(cpu):
 normals=[];indices=[];offsets=[0];mesh_faces=[];edge_pairs=[];edge_offsets=[0]
 for mesh in range(cpu.nmesh):
  adr=int(cpu.mesh_vertadr[mesh]);verts=cpu.mesh_vert[adr:adr+cpu.mesh_vertnum[mesh]]
  hull=ConvexHull(verts);groups=[]
  for face,eq in zip(hull.simplices,hull.equations):
   for g in groups:
    if np.linalg.norm(eq-g[0])<1e-5:g[1].update(face.tolist());break
   else:groups.append([eq,set(face.tolist())])
  mesh_faces.append(len(groups));edges=set()
  for eq,ids in groups:
   ids=list(ids);normal=eq[:3];center=verts[ids].mean(0);u=verts[ids[0]]-center;u/=np.linalg.norm(u);v=np.cross(normal,u)
   ids=[ids[i] for i in np.argsort(np.arctan2((verts[ids]-center)@v,(verts[ids]-center)@u))]
   assert len(ids)<=16
   normals.extend(normal.tolist());indices.extend([len(ids)]+ids+[-1]*(16-len(ids)))
   for i in range(len(ids)):edges.add(tuple(sorted((ids[i],ids[(i+1)%len(ids)]))))
  offsets.append(offsets[-1]+len(groups));edge_pairs.extend(v for e in sorted(edges) for v in e);edge_offsets.append(edge_offsets[-1]+len(edges))
 def array(typ,name,values):
  vals=','.join((f'{float(x):.9e}f' if typ=='float' else str(int(x))) for x in values) or '0'
  return f'__device__ __constant__ {typ} {name}[{max(1,len(values))}]={{'+vals+'};'
 return '\n'.join([array('int','MF_START',offsets),array('float','MF_NORMAL',normals),array('int','MF_VERT',indices),array('int','ME_START',edge_offsets),array('int','ME_VERT',edge_pairs)])
