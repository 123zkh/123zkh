"""Descriptive all-VSP fine-condition boxplots; leaves with n<50 are omitted."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-all-vsp-box')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/cache-all-vsp-box')
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parent
D=pd.read_csv(ROOT/'pems_combined_vsp_conditional_labeled_samples.csv')
OUT=ROOT/'pems_combined_all_vsp_fine_conditions'
ORDER=['VSP 负负荷（<0）','VSP 0–3','VSP 3–6','VSP 6–9','VSP 9–12','VSP 12–18','VSP ≥18']
MIN_N=50
D['细分工况']=D['VSP骨架']+'｜'+D['阶段修正']+'｜'+D['坡度区间']
cnt=D.groupby('细分工况').size();D['样本数']=D['细分工况'].map(cnt)
S=D[D['样本数']>=MIN_N].copy()
rows=[]
for state,g in S.groupby('细分工况'):
    rows.append({'VSP骨架':g['VSP骨架'].iloc[0],'细分工况':state,'样本数':len(g),'CO2均值（g/s）':g['碳排放强度'].mean(),'CO2中位数（g/s）':g['碳排放强度'].median(),'标准差（g/s）':g['碳排放强度'].std(),'Q1（g/s）':g['碳排放强度'].quantile(.25),'Q3（g/s）':g['碳排放强度'].quantile(.75)})
summary=pd.DataFrame(rows);summary['四分位距（g/s）']=summary['Q3（g/s）']-summary['Q1（g/s）'];summary.to_csv(OUT.with_name(OUT.name+'_summary.csv'),index=False,encoding='utf-8-sig')
font='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
if Path(font).exists():plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'font.size':11})
fig,axes=plt.subplots(2,4,figsize=(21,11),dpi=220,sharey=True);axes=axes.ravel()
colors={'怠速/停车':'#9BB58B','制动/滑行':'#A6A6C8','匀速':'#7AA6C2','起步/加速':'#C88991'}
for ax,base in zip(axes,ORDER):
    states=summary[summary['VSP骨架']==base].sort_values(['CO2中位数（g/s）','样本数']).copy()
    vals=[];labs=[];boxcolors=[]
    for _,r in states.iterrows():
        state=r['细分工况'];part=state.split('｜');phase=part[1];grade=part[2]
        vals.append(S.loc[S['细分工况']==state,'碳排放强度'].to_numpy());labs.append(f'{phase}\n{grade}\nn={int(r["样本数"])}');boxcolors.append(colors[phase])
    bp=ax.boxplot(vals,patch_artist=True,showfliers=False,widths=.62,medianprops={'color':'#263746','linewidth':1.5},boxprops={'edgecolor':'#657786'},whiskerprops={'color':'#657786'},capprops={'color':'#657786'})
    for b,c in zip(bp['boxes'],boxcolors):b.set_facecolor(c);b.set_alpha(.82)
    ax.set_title(base,fontsize=15,weight='bold',pad=7);ax.set_xticklabels(labs,fontsize=8.2);ax.grid(axis='y',color='#DCE6F1');ax.set_axisbelow(True);ax.set_ylim(0,6.2)
    if ax in axes[::4]:ax.set_ylabel('瞬时 CO$_2$ 排放强度（g·s$^{-1}$）',fontsize=12)
axes[-1].axis('off')
fig.suptitle('不同 VSP 区间内细分运行工况的 CO$_2$ 排放分布',fontsize=20,weight='bold',y=.99)
fig.text(.5,.015,'仅展示样本数不少于 50 秒的细分工况；箱体为第 1 至第 3 四分位数，中线为中位数，须线为 1.5 倍四分位距范围。',ha='center',fontsize=11)
fig.tight_layout(rect=(0,.04,1,.965));fig.savefig(OUT.with_suffix('.png'),bbox_inches='tight');fig.savefig(OUT.with_suffix('.pdf'),bbox_inches='tight')
report=f'''# 全 VSP 区间细分工况箱线图\n\n在每个 VSP 区间内均按“运动阶段＋坡度”生成候选子工况，并仅展示样本数不少于 {MIN_N} 秒的组合。该图用于描述不同细分工况的排放分布；它不替代经连续区段留出验证筛选后的最终因子库。\n\n共展示 {len(summary)} 个细分工况，覆盖 {len(S)}/{len(D)} 条记录（{len(S)/len(D)*100:.1f}%）。\n'''
OUT.with_name(OUT.name+'_report.md').write_text(report,encoding='utf-8')
print(summary.sort_values(['VSP骨架','CO2中位数（g/s）']).to_string(index=False))
