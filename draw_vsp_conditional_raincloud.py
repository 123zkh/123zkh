"""Ordered raincloud plot for the validated VSP conditional factor library."""
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

ROOT=Path(__file__).resolve().parent
D=pd.read_csv(ROOT/'pems_combined_vsp_conditional_labeled_samples.csv')
OUT=ROOT/'pems_vsp_conditional_raincloud'
low_states=[
 'VSP 0–3｜怠速/停车｜下坡（<-1°）','VSP 0–3｜怠速/停车｜平坡（−1°至1°）','VSP 0–3｜怠速/停车｜上坡（>1°）',
 'VSP 0–3｜匀速｜下坡（<-1°）','VSP 0–3｜匀速｜平坡（−1°至1°）','VSP 0–3｜制动/滑行｜平坡（−1°至1°）',
 'VSP 0–3｜起步/加速｜平坡（−1°至1°）','VSP 0–3｜起步/加速｜下坡（<-1°）','VSP 0–3',
]
base_order=['VSP 负负荷（<0）','VSP 3–6','VSP 6–9','VSP 9–12','VSP 12–18','VSP ≥18']
base_labels={'VSP 负负荷（<0）':'[−∞, 0)','VSP 3–6':'[3, 6)','VSP 6–9':'[6, 9)','VSP 9–12':'[9, 12)','VSP 12–18':'[12, 18)','VSP ≥18':'[18, +∞)'}
base_colors={'VSP 负负荷（<0）':'#91A8B7','VSP 0–3':'#C88991','VSP 3–6':'#D5AD6D','VSP 6–9':'#9BB58B','VSP 9–12':'#7DA8C8','VSP 12–18':'#A993C7','VSP ≥18':'#B49E7A'}
D['绘图工况']=D['最终工况']
available=set(D['绘图工况'])
low=[x for x in low_states if x in available]
# Within VSP 0-3, sort states by observed median CO2 intensity.
median=D[D['绘图工况'].isin(low)].groupby('绘图工况')['碳排放强度'].median()
low=sorted(low,key=lambda x:median[x])
order=['VSP 负负荷（<0）']+low+base_order[1:]
order=[x for x in order if x in available]
def base_of(state):
    return 'VSP 0–3' if state.startswith('VSP 0–3') else state

def detail_label(state):
    if state=='VSP 0–3':return '其他低负荷状态'
    part=state.split('｜')
    if len(part)==3:
        return part[1]+'\n'+part[2].split('（')[0]
    return ''
font='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
if Path(font).exists(): plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'font.size':36})
rng=np.random.default_rng(20261110)
fig,ax=plt.subplots(figsize=(42,20),dpi=180)
positions={state:i for i,state in enumerate(order,start=1)}
for state,i in positions.items():
    y=D.loc[D['绘图工况']==state,'碳排放强度'].to_numpy(); color=base_colors[base_of(state)]
    vio=ax.violinplot([y],positions=[i+.20],widths=.55,showmeans=False,showmedians=False,showextrema=False)
    for body in vio['bodies']:
        body.set_facecolor(color);body.set_edgecolor('none');body.set_alpha(.28)
    bp=ax.boxplot([y],positions=[i-.12],widths=.34,patch_artist=True,showfliers=False,
                  medianprops={'color':'white','linewidth':1.8},boxprops={'edgecolor':color,'linewidth':1.2},
                  whiskerprops={'color':color,'linewidth':1.4},capprops={'color':color,'linewidth':1.4})
    bp['boxes'][0].set_facecolor(color);bp['boxes'][0].set_alpha(.85)
    sample=y if len(y)<=180 else rng.choice(y,180,replace=False)
    ax.scatter(i+.18+rng.normal(0,.045,len(sample)),sample,s=13,color=color,alpha=.45,edgecolors='none',zorder=3)
    ax.scatter(i-.12,np.mean(y),s=52,color='white',edgecolors=color,linewidths=1.2,zorder=5)
    # Labels are printed above the detailed VSP 0-3 boxes; no n is printed.
    if state.startswith('VSP 0–3'):
        # Detailed conditions are arranged in a single row above the VSP 0–3 group.
        ax.text(i, 6.86, detail_label(state), ha='center', va='top', fontsize=26.4, fontweight='bold', color='#4A4A4A', linespacing=1.06)
# Dashed separators at every VSP-bin boundary.
groups=[['VSP 负负荷（<0）'],low,[x for x in base_order[1:] if x in available]]
# The six unsegmented VSP bins each form an individual section.
groups=groups[:2]+[[x] for x in groups[2]]
for g in groups[:-1]:
    last=positions[g[-1]]
    ax.axvline(last+.5,color='#8EA9C1',lw=1.5,ls=(0,(6,5)),zorder=0)
# VSP classifications shown on the x axis; VSP0-3 is the parent group label.
centers=[];xlabels=[]
for g in groups:
    centers.append(np.mean([positions[x] for x in g]))
    xlabels.append('[0, 3)' if g==low else base_labels.get(g[0],g[0]))
ax.set_xticks(centers,xlabels,fontsize=39)
ax.tick_params(axis='x',pad=22); ax.tick_params(axis='y',labelsize=39)
ax.set_xlim(.45,len(order)+.65);ax.set_ylim(0,7.20)
ax.set_ylabel('瞬时 CO$_2$ 排放强度（g·s$^{-1}$）',fontsize=48)
ax.set_xlabel('VSP 区间（kW/t）',fontsize=45,labelpad=30)
ax.set_title('不同 VSP 工况下的 CO$_2$ 排放分布',fontsize=66,weight='bold',pad=40)
ax.grid(axis='y',color='#DCE6F1',linewidth=.9);ax.set_axisbelow(True)
ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
fig.subplots_adjust(top=.86,bottom=.20,left=.105,right=.985)
fig.savefig(OUT.with_suffix('.png'),bbox_inches='tight');fig.savefig(OUT.with_suffix('.pdf'),bbox_inches='tight')
pd.DataFrame({'排列序号':range(1,len(order)+1),'VSP区间':[base_of(x) for x in order],'绘图工况':order,'中位排放强度（g/s）':[median.get(x,D.loc[D['绘图工况']==x,'碳排放强度'].median()) for x in order]}).to_csv(OUT.with_name(OUT.name+'_groups.csv'),index=False,encoding='utf-8-sig')
