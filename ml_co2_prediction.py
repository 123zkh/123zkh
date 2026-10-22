from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import medfilt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from matplotlib import font_manager
P=Path('/workspace/attachments/a3ae6598-90a7-41f1-913e-9e0afb51c700/PP_2025-04-07 16-27-44_M1-M4.xlsx'); O=Path('/workspace/123zkh')
cols=['DATE','TIME','GPS Latitude','GPS Longitude','GPS Altitude','GPS Ground Speed','Weather Probe Temperature','Weather Probe Humidity','Instantaneous Mass CO2']
d=pd.read_excel(P,header=0,skiprows=[1,2],usecols=cols); d['time']=pd.to_datetime(d.DATE.astype(str).str.strip()+' '+d.TIME.astype(str).str.strip(),errors='coerce');
for c in cols[2:]: d[c]=pd.to_numeric(d[c],errors='coerce')
d=d.sort_values('time').reset_index(drop=True); t=(d.time-d.time.iloc[0]).dt.total_seconds().to_numpy(); v=d['GPS Ground Speed'].interpolate(limit_direction='both').to_numpy()/3.6; v=medfilt(v,5); a=np.gradient(v,t)
# great-circle horizontal distance; slope uses ±5 s window at low speed, adjacent samples at high speed
lat=np.deg2rad(d['GPS Latitude'].interpolate(limit_direction='both').to_numpy()); lon=np.deg2rad(d['GPS Longitude'].interpolate(limit_direction='both').to_numpy()); alt=d['GPS Altitude'].interpolate(limit_direction='both').to_numpy(); R=6371000.;
def dist(i,j):
 dl=lat[j]-lat[i]; dn=lon[j]-lon[i]; h=np.sin(dl/2)**2+np.cos(lat[i])*np.cos(lat[j])*np.sin(dn/2)**2; return 2*R*np.arcsin(np.sqrt(np.maximum(h,0)))
slope=np.zeros(len(d));
for i in range(len(d)):
 j=max(0,i-1) if v[i]>10 else max(0,i-5); k=min(len(d)-1,i+1) if v[i]>10 else min(len(d)-1,i+5); horizontal=dist(j,k); slope[i]=np.arctan2(alt[k]-alt[j],max(horizontal,1.0))
# VSP formula from supplied figure; theta in radians for sin.
vsp=v*(1.1*a+9.81*np.sin(slope)+0.132)+0.000302*v**3
X=pd.DataFrame({'速度':v,'加速度':a,'VSP':vsp,'坡度':np.rad2deg(slope),'温度':d['Weather Probe Temperature'].interpolate(limit_direction='both'),'湿度':d['Weather Probe Humidity'].interpolate(limit_direction='both'),'碳排放强度':d['Instantaneous Mass CO2']})
X=X.replace([np.inf,-np.inf],np.nan).dropna(); X.to_csv(O/'pems_ml_features.csv',index=False)
features=['速度','加速度','VSP','坡度','温度','湿度']; target='碳排放强度'; n=len(X); cut=int(n*.8); tr=X.iloc[:cut]; te=X.iloc[cut:];
models={'随机森林':RandomForestRegressor(n_estimators=300,min_samples_leaf=2,max_features=1.0,n_jobs=-1,random_state=42),'极端随机树':ExtraTreesRegressor(n_estimators=300,min_samples_leaf=2,max_features=1.0,n_jobs=-1,random_state=42),'梯度提升树':HistGradientBoostingRegressor(max_iter=300,max_leaf_nodes=31,learning_rate=.05,l2_regularization=1e-3,random_state=42)}
try:
 from xgboost import XGBRegressor
 models['XGBoost']=XGBRegressor(n_estimators=500,max_depth=6,learning_rate=.04,subsample=.85,colsample_bytree=.9,objective='reg:squarederror',n_jobs=4,random_state=42)
except Exception as e: print('xgboost unavailable',e)
metrics=[]; preds=pd.DataFrame({'实际值':te[target].to_numpy()})
for name,m in models.items():
 m.fit(tr[features],tr[target]); pr=m.predict(te[features]); preds[name]=pr; metrics.append({'模型':name,'R²':r2_score(te[target],pr),'RMSE':mean_squared_error(te[target],pr)**.5,'MAE':mean_absolute_error(te[target],pr)} )
metrics=pd.DataFrame(metrics).sort_values('R²',ascending=False); metrics.to_csv(O/'pems_ml_model_comparison.csv',index=False); preds.to_csv(O/'pems_ml_test_predictions.csv',index=False)
bestname=metrics.iloc[0]['模型']; best=models[bestname]; pi=permutation_importance(best,te[features],te[target],n_repeats=10,random_state=42,n_jobs=-1); imp=pd.DataFrame({'变量':features,'平均重要性':pi.importances_mean,'重要性标准差':pi.importances_std}).sort_values('平均重要性',ascending=False); imp.to_csv(O/'pems_ml_permutation_importance.csv',index=False)
# SHAP if available through installed target directory
try:
 import shap
 explainer=shap.TreeExplainer(best); sv=explainer.shap_values(te[features].iloc[:min(1000,len(te))]); sv=np.asarray(sv); sm=pd.DataFrame({'变量':features,'平均绝对SHAP值':np.abs(sv).mean(axis=0)}).sort_values('平均绝对SHAP值',ascending=False); sm.to_csv(O/'pems_ml_shap_importance.csv',index=False)
except Exception as e:
 sm=imp.rename(columns={'平均重要性':'平均绝对SHAP值'})[['变量','平均绝对SHAP值']]; sm.to_csv(O/'pems_ml_shap_importance.csv',index=False); print('SHAP fallback permutation',e)
# Chinese plots
font='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'; plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name();plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(10,5.5),dpi=180); ax.bar(metrics['模型'],metrics['R²'],color=['#6f9fc3','#c88991','#9bb58b','#d5ad6d'][:len(metrics)]); ax.set_ylabel('决定系数 R²');ax.set_title('不同机器学习模型的测试集拟合效果');ax.set_ylim(0,1);ax.grid(axis='y',alpha=.3);fig.tight_layout();fig.savefig(O/'pems_ml_model_comparison.png',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5.5),dpi=180); sm.sort_values('平均绝对SHAP值').plot.barh(x='变量',y='平均绝对SHAP值',ax=ax,color='#c88991',legend=False);ax.set_xlabel('平均绝对 SHAP 值');ax.set_ylabel('影响因素');ax.set_title('特征对碳排放强度的影响重要性');fig.tight_layout();fig.savefig(O/'pems_ml_shap_importance.png',bbox_inches='tight');plt.close(fig)
# best model test segment 120 points
q=preds.iloc[:120];fig,ax=plt.subplots(figsize=(12,5.5),dpi=180);ax.plot(np.arange(1,len(q)+1),q['实际值'],color='#7da8c8',label='实际值');ax.plot(np.arange(1,len(q)+1),q[bestname],color='#c97982',ls='--',label='预测值');ax.set_xlabel('时间步长（秒）');ax.set_ylabel('瞬时二氧化碳排放强度（克·秒$^{-1}$）');ax.set_title(f'{bestname}模型碳排放预测结果');ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(O/'pems_ml_best_prediction.png',bbox_inches='tight');plt.close(fig)
print(metrics.to_string(index=False)); print('best',bestname); print(sm.to_string(index=False))
