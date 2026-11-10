"""Raincloud-style plot: detailed VSP 0-3 states only, other VSP bins unsplit."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-vsp-raincloud')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/cache-vsp-raincloud')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parent
D=pd.read_csv(ROOT/'pems_combined_vsp_conditional_labeled_samples.csv')
OUT=ROOT/'pems_vsp_conditional_raincloud'
# The final validated library has conditional corrections only within VSP 0-3.
low_order=[
 'VSP 0–3｜怠速/停车｜下坡（<-1°）','VSP 0–3｜怠速/停车｜平坡（−1°至1°）','VSP 0–3｜怠速/停车｜上坡（>1°）',
 'VSP 0–3｜匀速｜下坡（<-1°）','VSP 0–3｜匀速｜平坡（−1°至1°）','VSP 0–3｜制动/滑行｜平坡（−1°至1°）',
 'VSP 0–3｜起步/加速｜平坡（−1°至1°）','VSP 0–3｜起步/加速｜下坡（<-1°）','VSP 0–3',
]
other_order=['VSP 负负荷（<0）','VSP 3–6','VSP 6–9','VSP 9–12','VSP 12–18','VSP ≥18']
# Fine leaves that did not satisfy n>=50 have already been returned to the
# parent low-VSP class by the validated-factor selection script.
D['绘图工况']=D['最终工况']
order=[x for x in ['VSP 负负荷（<0）']+low_order+['VSP 3–6','VSP 6–9','VSP 9–12','VSP 12–18','VSP ≥18'] if x in set(D['绘图工况'])]
labels={
 'VSP 0–3｜怠速/停车｜下坡（<-1°）':'怠速/停车\n下坡', 'VSP 0–3｜怠速/停车｜平坡（−1°至1°）':'怠速/停车\n平坡','VSP 0–3｜怠速/停车｜上坡（>1°）':'怠速/停车\n上坡',
 'VSP 0–3｜匀速｜下坡（<-1°）':'匀速\n下坡','VSP 0–3｜匀速｜平坡（−1°至1°）':'匀速\n平坡','VSP 0–3｜制动/滑行｜平坡（−1°至1°）':'制动/滑行\n平坡',
 'VSP 0–3｜起步/加速｜平坡（−1°至1°）':'起步/加速\n平坡','VSP 0–3｜起步/加速｜下坡（<-1°）':'起步/加速\n下坡','VSP 0–3':'其他\n低负荷状态',
 'VSP 负负荷（<0）':'VSP<0','VSP 3–6':'VSP 3–6','VSP 6–9':'VSP 6–9','VSP 9–12':'VSP 9–12','VSP 12–18':'VSP 12–18','VSP ≥18':'VSP≥18'}
colors=['#8FB4CC','#8FB4CC','#8FB4CC','#76A5C5','#76A5C5','#A6A6C8','#C88991','#C88991','#B8C5D2','#7D9BB3','#9BB58B','#6FA7C8','#C77F87','#CE9A7B','#B894C6']
font='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
if Path(font).exists(): plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'font.size':12})
rng=np.random.default_rng(20261110)
fig,ax=plt.subplots(figsize=(21,9),dpi=240)
for i,(state,color) in enumerate(zip(order,colors),start=1):
    y=D.loc[D['绘图工况']==state,'碳排放强度'].to_numpy()
    # Violin is placed right of the box; the box and its mean dot remain visible.
    vio=ax.violinplot([y],positions=[i+.20],widths=.55,showmeans=False,showmedians=False,showextrema=False)
    for body in vio['bodies']:
        body.set_facecolor(color);body.set_edgecolor('none');body.set_alpha(.30)
    bp=ax.boxplot([y],positions=[i-.12],widths=.34,patch_artist=True,showfliers=False,
                  medianprops={'color':'white','linewidth':1.8},boxprops={'edgecolor':color,'linewidth':1.2},
                  whiskerprops={'color':color,'linewidth':1.4},capprops={'color':color,'linewidth':1.4})
    bp['boxes'][0].set_facecolor(color);bp['boxes'][0].set_alpha(.85)
    # Draw up to 180 deterministic observations to avoid visual saturation.
    sample=y if len(y)<=180 else rng.choice(y,180,replace=False)
    jitter=i+.18+rng.normal(0,.045,len(sample))
    ax.scatter(jitter,sample,s=13,color=color,alpha=.48,edgecolors='none',zorder=3)
    ax.scatter(i-.12,np.mean(y),s=52,color='white',edgecolors=color,linewidths=1.2,zorder=5)
# Ordered strictly by VSP interval; no explanatory header or legend.
total=len(order)
ax.set_xlim(.45,total+.65);ax.set_ylim(0,6.2)
ax.set_xticks(range(1,total+1),[labels[x] for x in order],fontsize=10.5)
ax.set_ylabel('瞬时 CO$_2$ 排放强度（g·s$^{-1}$）',fontsize=16)
ax.set_xlabel('运行工况',fontsize=15,labelpad=12)
ax.set_title('不同 VSP 工况下的 CO$_2$ 排放分布',fontsize=22,weight='bold',pad=18)
ax.grid(axis='y',color='#DCE6F1',linewidth=.9);ax.set_axisbelow(True)
ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
fig.subplots_adjust(top=.90,bottom=.19,left=.08,right=.985)
fig.savefig(OUT.with_suffix('.png'),bbox_inches='tight');fig.savefig(OUT.with_suffix('.pdf'),bbox_inches='tight')
# Explicit plotting specification for the thesis methods appendix.
pd.DataFrame({'分区':['VSP 0–3 kW/t（细分）' if x in low_order else '其余VSP（未细分）' for x in order], '绘图工况':order,'横坐标标签':[labels[x].replace('\n',' ') for x in order], '样本数':[int((D['绘图工况']==x).sum()) for x in order]}).to_csv(OUT.with_name(OUT.name+'_groups.csv'),index=False,encoding='utf-8-sig')
print(OUT.with_suffix('.png'))
