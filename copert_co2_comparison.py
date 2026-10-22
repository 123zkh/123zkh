from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
O=Path('/workspace/123zkh'); D=pd.read_csv(O/'pems_fair_common_dataset.csv'); tr,te=train_test_split(D,test_size=.2,random_state=42)
# Parameters transcribed from supplied figure (fuel-consumption factor).
a,b,c,d,e=217.0,9.6e-2,0.253,-4.12e-4,9.65e-3
v=te['速度'].to_numpy()*3.6; v_eval=np.maximum(v,5.0); fc=(a+c*v_eval+e*v_eval**2)/(1+b*v_eval+d*v_eval**2); ef_co2=3.18*fc; pred=ef_co2*v/3600; pred[v<=0]=0
metrics={'模型':'COPERT IV','R²':r2_score(te['碳排放强度'],pred),'RMSE':mean_squared_error(te['碳排放强度'],pred)**.5,'MAE':mean_absolute_error(te['碳排放强度'],pred)}
pd.DataFrame([metrics]).to_csv(O/'pems_copert_metrics.csv',index=False); out=te.copy();out['COPERT燃油消耗因子_g每km']=fc;out['COPERT二氧化碳因子_g每km']=ef_co2;out['COPERT预测_g每s']=pred;out.to_csv(O/'pems_copert_test_predictions.csv',index=False)
# same test-set comparison with existing models
m=pd.read_csv(O/'pems_fair_model_comparison_random_split.csv'); m=pd.concat([m,pd.DataFrame([metrics])],ignore_index=True);m.to_csv(O/'pems_all_models_including_copert.csv',index=False)
font='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc';plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name();plt.rcParams['axes.unicode_minus']=False
# scatter
fig,ax=plt.subplots(figsize=(7,7),dpi=200); y=te['碳排放强度'];lo=min(y.min(),pred.min());hi=max(y.max(),pred.max());ax.scatter(y,pred,s=13,alpha=.35,color='#c88991');ax.plot([lo,hi],[lo,hi],'--',color='#555',label='理想预测线');ax.set_xlabel('实测二氧化碳排放强度（克·秒$^{-1}$）');ax.set_ylabel('COPERT预测二氧化碳排放强度（克·秒$^{-1}$）');ax.set_title('COPERT模型预测效果');ax.text(.04,.95,f'R² = {metrics["R²"]:.4f}\nRMSE = {metrics["RMSE"]:.4f}',transform=ax.transAxes,va='top',bbox=dict(fc='white',alpha=.9));ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(O/'pems_copert_scatter.png',bbox_inches='tight');plt.close(fig)
# first 120 test points
q=pd.DataFrame({'实际值':y.iloc[:120].to_numpy(),'预测值':pred[:120]});fig,ax=plt.subplots(figsize=(12,5.5),dpi=200);x=np.arange(1,121);ax.plot(x,q['实际值'],color='#7da8c8',label='实际值');ax.plot(x,q['预测值'],color='#c97982',ls='--',label='COPERT预测值');ax.set_xlabel('时间步长（秒）');ax.set_ylabel('瞬时二氧化碳排放强度（克·秒$^{-1}$）');ax.set_title('COPERT模型碳排放预测结果');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(O/'pems_copert_prediction_curve.png',bbox_inches='tight');plt.close(fig)
report=f'''# COPERT IV二氧化碳计算结果\n\n根据提供图片，燃油消耗因子参数为 a=217，b=0.096，c=0.253，d=-0.000412，e=0.00965，计算式为 `EF_FC=(a+cV+eV²)/(1+bV+dV²)`，CO₂因子为 `EF_CO2=3.18×EF_FC`。速度单位为km/h，最终换算为g/s。\n\n所有方法使用同一份7312条清洗数据和同一个固定随机测试集（1463条）。\n\nCOPERT测试集：R²={metrics['R²']:.4f}，RMSE={metrics['RMSE']:.4f} g/s，MAE={metrics['MAE']:.4f} g/s。\n\nCOPERT结果明显高估当前车辆的实测瞬时CO₂，说明图中参数对应的车辆类别、燃料或排放标准与本车可能不匹配。该结果是未校准的直接COPERT计算，未使用PEMS数据反推参数。\n''';(O/'pems_copert_report.md').write_text(report,encoding='utf8');print(metrics)
