from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parent
m=pd.read_csv(ROOT/'pems_fair_model_comparison_random_split.csv')
p=pd.read_csv(ROOT/'pems_fair_test_predictions_random_split.csv')
# Keep the four requested tree models and arrange by intended comparison order
models=['梯度提升树','极端随机树','XGBoost','随机森林']
colors={'梯度提升树':'#74bbde','极端随机树':'#a7d0eb','XGBoost':'#f8d0cc','随机森林':'#c5cad8'}
fp='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'; font_manager.fontManager.addfont(fp)
font_name=font_manager.FontProperties(fname=fp).get_name(); plt.rcParams['font.family']=font_name; plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(2,2,figsize=(15,13),dpi=220)
for ax,name in zip(axes.flat,models):
 y=p['实际值'].to_numpy(); z=p[name].to_numpy(); lo=min(y.min(),z.min()); hi=max(y.max(),z.max());
 ax.scatter(y,z,s=25,alpha=.42,color=colors[name],edgecolors='none'); ax.plot([lo,hi],[lo,hi],ls='--',lw=2,color='#555555')
 row=m[m['模型']==name].iloc[0]
 ax.text(.045,.95,f'$R^2$ = {row["R²"]:.4f}\nRMSE = {row["RMSE"]:.4f}',transform=ax.transAxes,va='top',fontsize=17,bbox=dict(facecolor='white',edgecolor='#888',alpha=.9,pad=7))
 ax.set_xlabel('实测碳排放强度（g/s$^{-1}$）',fontsize=17,labelpad=8); ax.set_ylabel('预测碳排放强度（g/s$^{-1}$）',fontsize=17,labelpad=8)
 ax.tick_params(labelsize=15); ax.grid(alpha=.25); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
 # Put subplot name beneath the axes, as requested.
 ax.set_title(name,fontsize=23,pad=18,y=-.34)
fig.suptitle('四种机器学习模型碳排放预测效果对比',fontsize=27,y=.985)
fig.subplots_adjust(left=.09,right=.98,bottom=.12,top=.93,wspace=.28,hspace=.70)
fig.savefig(ROOT/'pems_fair_four_ml_scatter_large.png',bbox_inches='tight'); fig.savefig(ROOT/'pems_fair_four_ml_scatter_large.pdf',bbox_inches='tight'); plt.close(fig)
# Chinese metrics table
m[m['模型'].isin(models)].set_index('模型').loc[models].to_csv(ROOT/'pems_fair_four_ml_metrics_chinese.csv',encoding='utf-8-sig')
