from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import medfilt
import matplotlib.pyplot as plt
p=Path('/workspace/attachments/a3ae6598-90a7-41f1-913e-9e0afb51c700/PP_2025-04-07 16-27-44_M1-M4.xlsx'); out=Path('/workspace/123zkh')
d=pd.read_excel(p,header=0,skiprows=[1,2],usecols=['DATE','TIME','GPS Ground Speed','Instantaneous Mass CO2'])
d['timestamp']=pd.to_datetime(d.DATE.astype(str).str.strip()+' '+d.TIME.astype(str).str.strip(),errors='coerce'); d['v']=pd.to_numeric(d['GPS Ground Speed'],errors='coerce')/3.6; d['y']=pd.to_numeric(d['Instantaneous Mass CO2'],errors='coerce'); d=d.sort_values('timestamp').reset_index(drop=True); t=(d.timestamp-d.timestamp.iloc[0]).dt.total_seconds().to_numpy(); v=d.v.interpolate(limit_direction='both').to_numpy(); d['speed_m_s']=medfilt(v,5); d['accel_m_s2']=np.gradient(d.speed_m_s.to_numpy(),t)
z=d[['timestamp','speed_m_s','accel_m_s2','y']].replace([np.inf,-np.inf],np.nan).dropna().copy(); z.columns=['timestamp','v','a','y']; m=np.ones(len(z),bool)
for c in ['v','a','y']:
 q=z[c]; m &= q.between(q.mean()-3*q.std(),q.mean()+3*q.std()).to_numpy()
for _ in range(3):
 q=z.loc[m]; X=np.c_[np.ones(len(q)),q.v,q.v**2,q.a,q.a**2,q.v*q.a]; b=np.linalg.lstsq(X,q.y,rcond=None)[0]; rr=q.y-X@b; keep=np.abs(rr-rr.mean())<=3*rr.std(); idx=np.where(m)[0]; m[idx[~keep]]=False
q=z.loc[m].copy(); X=np.c_[np.ones(len(q)),q.v,q.v**2,q.a,q.a**2,q.v*q.a]; b=np.linalg.lstsq(X,q.y,rcond=None)[0]; rr=q.y-X@b; n=len(q); rss=rr@rr; r2=1-rss/((q.y-q.y.mean())@(q.y-q.y.mean())); rmse=np.sqrt(np.mean(rr**2)); mae=np.mean(np.abs(rr)); adj=1-(1-r2)*(n-1)/(n-6); cov=(rss/(n-6))*np.linalg.pinv(X.T@X); se=np.sqrt(np.diag(cov)); ci=1.96*se
terms=['f1','f2(v)','f3(v²)','f4(a)','f5(a²)','f6(va)']; pd.DataFrame({'term':terms,'coefficient':b,'std_error':se,'ci95_low':b-ci,'ci95_high':b+ci}).to_csv(out/'pems_co2_speed_acceleration_cleaned_coefficients.csv',index=False)
q['predicted_g_s']=X@b; q.to_csv(out/'pems_co2_speed_acceleration_cleaned_series.csv',index=False)
seg=q.iloc[:120]; XS=np.c_[np.ones(len(seg)),seg.v,seg.v**2,seg.a,seg.a**2,seg.v*seg.a]; band=1.96*np.sqrt(np.einsum('ij,jk,ik->i',XS,cov,XS)); fig,ax=plt.subplots(figsize=(13,5.8),dpi=180); ax.fill_between(seg.timestamp,seg.predicted_g_s-band,seg.predicted_g_s+band,color='#f4a6a6',alpha=.4,label='95% CI'); ax.plot(seg.timestamp,seg.y,color='#2f78b7',label='Measured'); ax.plot(seg.timestamp,seg.predicted_g_s,'--',color='#b64b5a',label='Predicted'); ax.set(xlabel='Time',ylabel='Instantaneous CO$_2$ mass (g s$^{-1}$)',title='Cleaned PEMS CO$_2$ speed-acceleration fit (120 s)'); ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(out/'pems_co2_speed_acceleration_cleaned_fit.png',bbox_inches='tight'); plt.close(fig)
formula=' + '.join([f'{b[0]:.6g}',f'{b[1]:+.6g}v',f'{b[2]:+.6g}v²',f'{b[3]:+.6g}a',f'{b[4]:+.6g}a²',f'{b[5]:+.6g}va'])
(out/'pems_co2_speed_acceleration_cleaned_report.md').write_text(f'''# 清洗后 CO₂-速度-加速度拟合\n\n按缺失值剔除、5点中值滤波、各变量均值±3σ异常值剔除，并对残差迭代3次±3σ筛选。原始有效样本7590，最终样本{n}。速度单位m/s，加速度单位m/s²，排放单位g/s。\n\n公式：`E = {formula}`\n\nR²={r2:.4f}；调整R²={adj:.4f}；RMSE={rmse:.4f} g/s；MAE={mae:.4f} g/s。\n''',encoding='utf-8'); print(n,r2,adj,rmse,mae,b)
