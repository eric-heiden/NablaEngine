from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
root = Path(__file__).resolve().parents[1]
rows = json.loads((root / 'environments/reference_speedups.json').read_text())['results']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,3,figsize=(11,6.5),sharex=True,layout='constrained')
colors=['#206da0','#16806a','#b26225']
for col,name in enumerate(['Go2 locomotion','Franka reach','Franka lift']):
 vals=[r for r in rows if r['task']==name]
 for row,key in enumerate(['physics_speedup','end_to_end_speedup']):
  ax=axs[row,col];x=[r['environments'] for r in vals];y=[r[key] for r in vals]
  ax.plot(x,y,'o-',color=colors[col],lw=2.5,ms=7)
  for xx,yy in zip(x,y):ax.annotate(f'{yy:.2f}×',(xx,yy),xytext=(0,10),textcoords='offset points',ha='center',fontsize=11)
  ax.axhline(1,color='#8a94a0',ls='--',lw=1);ax.grid(alpha=.18)
  ax.set_xscale('log',base=2);ax.set_xticks([128,512,2048],['128','512','2,048']);ax.set_xlim(90,2900)
  ax.set_ylim(.7,7.6 if row==0 else 2.65)
  if col==0:ax.set_ylabel(('Physics only' if row==0 else 'End-to-end')+'\nSpeedup over MuJoCo Warp')
  if row==0:ax.set_title(name,fontweight='bold',pad=12)
  else:ax.set_xlabel('Parallel environments')
fig.suptitle('Fused CUDA physics — final validated implementations',fontsize=16)
fig.supxlabel('NVIDIA L4 · End-to-end includes the original task runtime · Dashed line: equal speed',fontsize=10)
for ext in ['svg','png']:fig.savefig(root/f'docs/assets/task_speedups.{ext}',dpi=180)
svg = root / 'docs/assets/task_speedups.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
