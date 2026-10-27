from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parent
q=pd.read_csv(ROOT/'pems_co2_speed_acceleration_cleaned_series.csv')
coef=pd.read_csv(ROOT/'pems_co2_speed_acceleration_cleaned_coefficients.csv')
# Refit on the identical cleaned sample to obtain confidence band
v=q['v'].to_numpy(); a=q['a'].to_numpy(); y=q['y'].to_numpy()
X=np.c_[np.ones(len(q)),v,v**2,a,a**2,v*a]; b=np.linalg.lstsq(X,y,rcond=None)[0]
r=y-X@b; n=len(y); p=X.shape[1]; s2=(r@r)/(n-p); cov=s2*np.linalg.pinv(X.T@X)
r2=1-(r@r)/((y-y.mean())@(y-y.mean())); adj=1-(1-r2)*(n-1)/(n-p); rmse=np.sqrt(np.mean(r*r)); mae=np.mean(np.abs(r))
seg=q.iloc[:120].copy(); xs=np.c_[np.ones(len(seg)),seg.v,seg.v**2,seg.a,seg.a**2,seg.v*seg.a]; band=1.96*np.sqrt(np.einsum('ij,jk,ik->i',xs,cov,xs)+s2)
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']='Noto Sans CJK JP'; plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(14,7),dpi=220)
x=np.arange(1,len(seg)+1)
ax.fill_between(x,seg['predicted_g_s'].to_numpy()-band,seg['predicted_g_s'].to_numpy()+band,color='#f3b6b6',alpha=.42,label='95%置信区间')
ax.plot(x,seg['y'],color='#6f9fc2',lw=2.4,label='实测值')
ax.plot(x,seg['predicted_g_s'],'--',color='#c97984',lw=2.2,label='预测值')
ax.set_xlabel('时间步长（秒）',fontsize=20,labelpad=10); ax.set_ylabel('瞬时二氧化碳排放强度（克/秒）',fontsize=20,labelpad=12)
ax.tick_params(axis='both',labelsize=16); ax.set_xlim(1,120); ax.grid(alpha=.25)
ax.legend(fontsize=16,loc='upper right',frameon=True); fig.tight_layout()
fig.savefig(ROOT/'pems_co2_speed_acceleration_cleaned_fit_large.png',bbox_inches='tight'); fig.savefig(ROOT/'pems_co2_speed_acceleration_cleaned_fit_large.pdf',bbox_inches='tight'); plt.close(fig)
# Chinese coefficient/effect tables
names=['常数项 f₁','速度项 f₂','速度平方项 f₃','加速度项 f₄','加速度平方项 f₅','速度×加速度项 f₆']
co=pd.DataFrame({'参数项':names,'拟合系数':b,'标准误':np.sqrt(np.diag(cov)),'95%置信区间下限':b-1.96*np.sqrt(np.diag(cov)),'95%置信区间上限':b+1.96*np.sqrt(np.diag(cov))})
co.to_csv(ROOT/'pems_co2_speed_acceleration_polynomial_coefficients_chinese.csv',index=False,encoding='utf-8-sig')
met=pd.DataFrame([{'样本数':n,'决定系数R²':r2,'调整决定系数':adj,'均方根误差RMSE（克/秒）':rmse,'平均绝对误差MAE（克/秒）':mae}])
met.to_csv(ROOT/'pems_co2_speed_acceleration_polynomial_fit_metrics.csv',index=False,encoding='utf-8-sig')
(ROOT/'pems_co2_speed_acceleration_polynomial_fit_report.md').write_text(f'''# 速度—加速度—二氧化碳排放多项式拟合结果\n\n采用与前次分析完全相同的清洗数据，共 {n} 个有效样本。图中仅展示前 120 个时间步长，阴影为预测均值的 95% 置信区间。\n\n## 拟合效果\n\n- 决定系数 R²：{r2:.4f}\n- 调整决定系数：{adj:.4f}\n- 均方根误差 RMSE：{rmse:.4f} 克/秒\n- 平均绝对误差 MAE：{mae:.4f} 克/秒\n\n## 拟合参数\n\n模型形式为 `E=f₁+f₂v+f₃v²+f₄a+f₅a²+f₆va`，具体参数见 `pems_co2_speed_acceleration_polynomial_coefficients_chinese.csv`。\n''',encoding='utf-8')
print('n',n,'r2',r2,'adj',adj,'rmse',rmse,'mae',mae); print(co.to_string(index=False))
