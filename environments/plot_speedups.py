"""Plot the recorded physics-only scaling sweep; no benchmark is run here."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
rows = json.loads((root / 'environments/reference_speedups.json').read_text())['results']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig, axs = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
for ax, name, color in zip(axs.flat, ['Go2 locomotion','H1 locomotion','Franka reach','Franka lift'], ['#206da0','#9b4380','#16806a','#b26225']):
    vals = [r for r in rows if r['task'] == name]
    x, y = [r['environments'] for r in vals], [r['physics_speedup'] for r in vals]
    ax.plot(x, y, 'o-', color=color, lw=2.5, ms=7)
    for xx, yy in zip(x, y):
        ax.annotate(f'{yy:.2f}×', (xx, yy), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10)
    ax.axhline(1, color='#8a94a0', ls='--', lw=1)
    ax.grid(alpha=.18)
    ax.set_xscale('log', base=2)
    ax.set_xticks([128,512,2048,4096,8192], ['128','512','2,048','4,096','8,192'])
    ax.set_xlim(85,12000)
    ax.set_ylim(0,7.5)
    ax.set_ylabel('Speedup over MuJoCo Warp')
    ax.set_xlabel('Parallel environments')
    ax.set_title(name, fontweight='bold', pad=10)
fig.suptitle('Physics-only speedups on NVIDIA L4', fontsize=16)
fig.supxlabel('Matched integrations and contact refresh · Dashed line: equal speed', fontsize=10)
for ext in ['svg','png']:
    fig.savefig(root / f'docs/assets/task_speedups.{ext}', dpi=180)
svg = root / 'docs/assets/task_speedups.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
