import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

ROOT=Path(__file__).resolve().parent
D=pd.read_csv(ROOT/'pems_fair_common_dataset.csv')
# Recompute calibrated COPERT Tier 3 prediction for every cleaned observation.
# July 2026 COPERT EC parameters: Passenger Car / Petrol / Medium / Euro 6 a/b/c.
v_kmh=D['速度']*3.6
acc=D['加速度']
actual=D['碳排放强度'].astype(float)
alpha,beta,gamma,delta,eps,zeta,eta,rf = 0.000132,0.005486,2.619195,1.727746e-9,-0.000085,0.023584,0.344298,0.231967
# CO2 EF (g/km) -> fuel g/km -> CO2 g/km -> g/s; use calibration from prior PEMS fit.
ef_ec=((alpha*v_kmh**2+beta*v_kmh+gamma+delta/v_kmh)/(eps*v_kmh**2+zeta*v_kmh+eta))*(1-rf)
pred_raw=ef_ec*(3.168/43.774)*1000*v_kmh/3600
pred=pred_raw*0.6853814454410899
pred=np.where(np.isfinite(pred),pred,0.0)
# common operational-state definitions. Acceleration/deceleration take priority over idle/cruise.
state=np.where(acc>0.5,'加速',np.where(acc<-0.5,'减速',np.where(v_kmh<=5,'怠速','匀速')))
D['运行状态']=state
D['实测_CO2_g_s']=actual; D['COPERT_CO2_g_s']=pred; D['速度_km_h']=v_kmh
# g/km is defined for moving observations; remove near-zero speed from factor aggregation
valid=v_kmh>5
D['实测_CO2_g_km']=np.where(valid,actual*3600/v_kmh,np.nan)
D['COPERT_CO2_g_km']=np.where(valid,pred*3600/v_kmh,np.nan)
rows=[]
order=['怠速','匀速','加速','减速']
for s in order:
    q=D[D['运行状态']==s]
    qk=q[q['速度_km_h']>5]
    # report intensity in g/s for all state samples; error in g/km when enough moving samples
    a=q['实测_CO2_g_s'].mean(); m=q['COPERT_CO2_g_s'].mean()
    ak=qk['实测_CO2_g_km'].mean() if len(qk) else np.nan
    mk=qk['COPERT_CO2_g_km'].mean() if len(qk) else np.nan
    alpha=abs(ak-mk)/ak if np.isfinite(ak) and ak!=0 else np.nan
    rows.append({'运行状态':s,'样本数':len(q),'实测平均_CO2_g_s':a,'COPERT平均_CO2_g_s':m,'实测平均_CO2_g_km':ak,'COPERT平均_CO2_g_km':mk,'误差率_alpha':alpha})
# overall all data, moving g/km
qk=D[D['速度_km_h']>5]
rows.append({'运行状态':'全部（行驶样本）','样本数':len(qk),'实测平均_CO2_g_s':qk['实测_CO2_g_s'].mean(),'COPERT平均_CO2_g_s':qk['COPERT_CO2_g_s'].mean(),'实测平均_CO2_g_km':qk['实测_CO2_g_km'].mean(),'COPERT平均_CO2_g_km':qk['COPERT_CO2_g_km'].mean(),'误差率_alpha':abs(qk['实测_CO2_g_km'].mean()-qk['COPERT_CO2_g_km'].mean())/qk['实测_CO2_g_km'].mean()})
out=pd.DataFrame(rows)
out.to_csv(ROOT/'pems_copert_state_comparison.csv',index=False,encoding='utf-8-sig')
# per-observation error summary, all valid moving observations
err=D.loc[valid,['实测_CO2_g_km','COPERT_CO2_g_km','运行状态']].copy(); err['误差率_alpha']=abs(err['实测_CO2_g_km']-err['COPERT_CO2_g_km'])/err['实测_CO2_g_km'].replace(0,np.nan)
err.groupby('运行状态').agg(样本数=('误差率_alpha','count'),平均误差率=('误差率_alpha','mean'),误差率中位数=('误差率_alpha','median'),超过20比例=('误差率_alpha',lambda x:(x>0.2).mean())).reindex(order).to_csv(ROOT/'pems_copert_state_error_summary.csv',encoding='utf-8-sig')
# chart 1: reference-like grouped bars, measured/COPERT CO2 intensity g/s
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']='Noto Sans CJK SC'
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(10.5,6.5),dpi=180)
x=np.arange(4); w=.34
ax.bar(x-w/2,out.iloc[:4]['实测平均_CO2_g_s'],w,label='实测值',color='#e74c3c')
ax.bar(x+w/2,out.iloc[:4]['COPERT平均_CO2_g_s'],w,label='COPERT计算值',color='#9b59b6')
ax.set_xticks(x,order); ax.set_ylabel('二氧化碳排放强度（克/秒）'); ax.set_xlabel('车辆运行状态')
ax.set_title('不同运行状态下 COPERT 与 PEMS 实测二氧化碳排放')
ax.grid(axis='y',alpha=.25); ax.legend(frameon=True); fig.tight_layout()
fig.savefig(ROOT/'pems_copert_state_comparison.png',bbox_inches='tight'); plt.close(fig)
# chart 2: error rate by state for moving states only
fig,ax=plt.subplots(figsize=(9.5,5.8),dpi=180)
ss=out.iloc[:4]; vals=ss['误差率_alpha'].fillna(0)*100
ax.bar(order,vals,color='#4c8eda',alpha=.9)
for i,v in enumerate(vals):
    if np.isfinite(v) and v>0: ax.text(i,v+max(vals)*.02,f'{v:.1f}%',ha='center',fontsize=10)
ax.set_ylabel('相对误差率 α（%）'); ax.set_xlabel('车辆运行状态'); ax.set_title('COPERT 计算值与 PEMS 实测值的状态误差')
ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(ROOT/'pems_copert_state_error.png',bbox_inches='tight'); plt.close(fig)
# markdown report
with open(ROOT/'pems_copert_state_report.md','w',encoding='utf-8') as f:
 f.write('# COPERT 按运行状态对比\n\n')
 f.write('采用全部 7312 条统一清洗后的 PEMS 样本。COPERT Tier 3 能源消耗函数本身按连续车速计算，再乘以 PEMS 校准系数 0.685381；运行状态用于汇总展示。\n\n')
 f.write('状态定义：加速 `a>0.5 m/s²`，减速 `a<-0.5 m/s²`，其余速度不超过 5 km/h 为怠速，其余为匀速。误差率按 α=|实测−模型|/实测；g/km 统计剔除速度≤5 km/h 的样本。\n\n')
 f.write(out.to_csv(index=False,float_format='%.4f'))
 f.write('\n\n说明：COPERT 的连续速度函数不是把速度简单分成四档；本次四种状态仅用于与示例图一致地汇总。\n')
print(out.to_string(index=False))
