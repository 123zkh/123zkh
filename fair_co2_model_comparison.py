from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import font_manager
O=Path('/workspace/123zkh'); X=pd.read_csv(O/'pems_ml_features.csv'); fs=['速度','加速度','VSP','坡度','温度','湿度']; y='碳排放强度'
# One common cleaned dataset for every model.
mask=np.ones(len(X),bool)
for c in fs+[y]:
 q=X[c]; mask &= q.between(q.mean()-3*q.std(),q.mean()+3*q.std()).to_numpy()
D=X.loc[mask].reset_index(drop=True); D.to_csv(O/'pems_fair_common_dataset.csv',index=False)
tr,te=train_test_split(D,test_size=.2,random_state=42)
def design(z): return np.c_[np.ones(len(z)),z['速度'],z['速度']**2,z['加速度'],z['加速度']**2,z['速度']*z['加速度']]
models={
 '多项式回归':None,
 '随机森林':RandomForestRegressor(n_estimators=500,min_samples_leaf=2,max_features=1.0,n_jobs=-1,random_state=42),
 '极端随机树':ExtraTreesRegressor(n_estimators=500,min_samples_leaf=2,max_features=1.0,n_jobs=-1,random_state=42),
 '梯度提升树':HistGradientBoostingRegressor(max_iter=500,max_leaf_nodes=31,learning_rate=.04,l2_regularization=1e-3,random_state=42)}
try:
 from xgboost import XGBRegressor
 models['XGBoost']=XGBRegressor(n_estimators=900,max_depth=4,learning_rate=.03,subsample=.9,colsample_bytree=.9,min_child_weight=3,reg_lambda=1.0,objective='reg:squarederror',n_jobs=4,random_state=42)
except Exception as e: print('xgboost unavailable:',e)
res=[]; pred=pd.DataFrame({'实际值':te[y].to_numpy()}); fitted={}
for name,model in models.items():
 if name=='多项式回归':
  b=np.linalg.lstsq(design(tr),tr[y],rcond=None)[0]; p=design(te)@b; fitted[name]=(b,None)
 else:
  model.fit(tr[fs],tr[y]); p=model.predict(te[fs]); fitted[name]=(model,model)
 pred[name]=p; res.append({'模型':name,'R²':r2_score(te[y],p),'RMSE':mean_squared_error(te[y],p)**.5,'MAE':mean_absolute_error(te[y],p)})
metrics=pd.DataFrame(res).sort_values('R²',ascending=False); metrics.to_csv(O/'pems_fair_model_comparison_random_split.csv',index=False); pred.to_csv(O/'pems_fair_test_predictions_random_split.csv',index=False)
best=metrics.iloc[0]['模型']; model=fitted[best][0]
# SHAP-compatible model explanation: permutation for all models and TreeSHAP for best tree model when available.
if best!='多项式回归':
 pi=permutation_importance(model,te[fs],te[y],n_repeats=10,random_state=42,n_jobs=-1); pd.DataFrame({'变量':fs,'平均置换重要性':pi.importances_mean,'标准差':pi.importances_std}).sort_values('平均置换重要性',ascending=False).to_csv(O/'pems_fair_permutation_importance.csv',index=False)
try:
 import shap
 if best!='多项式回归':
  sv=shap.TreeExplainer(model).shap_values(te[fs].iloc[:min(1000,len(te))]); imp=pd.DataFrame({'变量':fs,'平均绝对SHAP值':np.abs(np.asarray(sv)).mean(axis=0)}).sort_values('平均绝对SHAP值',ascending=False)
  imp.to_csv(O/'pems_fair_shap_importance.csv',index=False)
except Exception as e: print('SHAP unavailable:',e)
font='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'; plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name();plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(10,5.8),dpi=180); ax.bar(metrics['模型'],metrics['R²'],color=['#c88991','#6f9fc3','#9bb58b','#d5ad6d','#a993c7'][:len(metrics)]); ax.set_ylabel('测试集决定系数 R²');ax.set_title('相同数据处理下不同模型的碳排放预测效果');ax.set_ylim(0,1);ax.grid(axis='y',alpha=.3);fig.tight_layout();fig.savefig(O/'pems_fair_model_comparison_random_split.png',bbox_inches='tight');plt.close(fig)
q=pred.iloc[:120];fig,ax=plt.subplots(figsize=(12,5.5),dpi=180);ax.plot(np.arange(1,121),q['实际值'],color='#7da8c8',lw=1.8,label='实际值');ax.plot(np.arange(1,121),q[best],color='#c97982',lw=1.8,ls='--',label='预测值');ax.set_xlabel('时间步长（秒）');ax.set_ylabel('瞬时二氧化碳排放强度（克·秒$^{-1}$）');ax.set_title(f'最优模型{best}测试集预测结果');ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(O/'pems_fair_best_prediction_random_split.png',bbox_inches='tight');plt.close(fig)
metrics.to_string(index=False)
print('共同样本数',len(D),'训练',len(tr),'测试',len(te));print(metrics.to_string(index=False));print('最优模型',best)
