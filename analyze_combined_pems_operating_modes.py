"""Combined PEMS operating-mode analysis for the April and May 2025 test runs."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-combined-pems')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/cache-combined-pems')
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent
FILES=[
 (Path('/workspace/attachments/a3ae6598-90a7-41f1-913e-9e0afb51c700/PP_2025-04-07 16-27-44_M1-M4.xlsx'),'2025年4月行程'),
 (Path('/workspace/attachments/38a44027-2ffc-4d88-a0da-9cfea2b7fe9a/PP_2025-05-20 12-50-14_M01-M02.xlsx'),'2025年5月行程'),
]
P='pems_combined_vsp_conditional'
MIN_N=50

COLS=['DATE','TIME','GPS Ground Speed','Instantaneous Mass CO2','GPS Altitude','GPS Latitude','GPS Longitude','Weather Probe Temperature','Weather Probe Humidity']
def horizontal_dist(lat1,lon1,lat2,lon2):
    # locally accurate equirectangular distance for short 1 Hz steps
    r=6371000.; lat1=np.deg2rad(lat1); lat2=np.deg2rad(lat2); dl=np.deg2rad(lon2-lon1)
    return r*np.sqrt((lat2-lat1)**2+(np.cos((lat1+lat2)/2)*dl)**2)

def calc_grade(lat,lon,alt):
    alt=medfilt(np.asarray(alt,float),5)
    seg=np.r_[0,horizontal_dist(lat[:-1],lon[:-1],lat[1:],lon[1:])]
    cum=np.cumsum(seg); n=len(cum); out=np.zeros(n)
    # Following the supplied reference principle: calculate slope over a path
    # containing at least 10 m (about 5 m before and after the point).
    for i in range(n):
        left=np.searchsorted(cum,cum[i]-5,side='left')
        right=np.searchsorted(cum,cum[i]+5,side='left')
        left=max(0,left);right=min(n-1,right)
        dx=cum[right]-cum[left]
        if dx>=10: out[i]=np.degrees(np.arctan2(alt[right]-alt[left],dx))
    return out

def read_trial(path,name):
    d=pd.read_excel(path,skiprows=[1,2],usecols=COLS)
    d['timestamp']=pd.to_datetime(d['DATE'].astype(str).str.strip()+' '+d['TIME'].astype(str).str.strip(),errors='coerce')
    for c in COLS[2:]: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.sort_values('timestamp').reset_index(drop=True)
    d=d.dropna(subset=['timestamp','GPS Ground Speed','Instantaneous Mass CO2','GPS Altitude','GPS Latitude','GPS Longitude']).copy()
    d=d[d['Instantaneous Mass CO2']>=0].reset_index(drop=True)
    t=(d['timestamp']-d['timestamp'].iloc[0]).dt.total_seconds().to_numpy()
    v=medfilt((d['GPS Ground Speed'].to_numpy()/3.6),5)
    d['速度']=v
    d['加速度']=np.gradient(v,t)
    d['坡度']=calc_grade(d['GPS Latitude'].to_numpy(),d['GPS Longitude'].to_numpy(),d['GPS Altitude'].to_numpy())
    theta=np.deg2rad(d['坡度'].to_numpy())
    d['VSP']=v*(1.1*d['加速度'].to_numpy()+9.81*np.sin(theta)+.132)+.000302*v**3
    d['温度']=d['Weather Probe Temperature'].interpolate(limit_direction='both')
    d['湿度']=d['Weather Probe Humidity'].interpolate(limit_direction='both')
    d['碳排放强度']=d['Instantaneous Mass CO2']
    d['行程']=name
    return d[['行程','timestamp','速度','加速度','VSP','坡度','温度','湿度','碳排放强度']]

def phase(d):
    v=d['速度']*3.6;a=d['加速度']
    x=np.select([v<1,a<=-.447,(a>-.447)&(a<-.20),(a>=.20)&(v<15),a>=.20],['怠速/停车','制动','滑行','起步加速','加速'],default='匀速/轻微波动')
    return pd.Series(x,index=d.index)
def phase_correct(d):
    return phase(d).map({'怠速/停车':'怠速/停车','制动':'制动/滑行','滑行':'制动/滑行','起步加速':'起步/加速','加速':'起步/加速','匀速/轻微波动':'匀速'})
def weighted_sd(y,g):
    q=pd.DataFrame({'y':y,'g':g}).groupby('g')['y'].agg(['size','std'])
    return np.sqrt(np.average(q['std'].fillna(0)**2,weights=q['size']))
def fallback_labels(train,test,cand,parent):
    n=train.groupby(cand).size(); ok=n[n>=MIN_N].index
    return cand.where(cand.isin(ok),parent).astype(str),test[cand.name].where(test[cand.name].isin(ok),test[parent.name]).astype(str)
def block_cv(d,cand,parent):
    se_c=se_b=n=0
    # preserve test-trip chronology: five contiguous test blocks within each run
    for _,g in d.groupby('行程',sort=False):
        for ids in np.array_split(g.index.to_numpy(),5):
            test=d.loc[ids]; train=d.drop(index=ids)
            tr,te=fallback_labels(train,test,train[cand],train[parent])
            mc=train.groupby(tr)['碳排放强度'].mean(); mb=train.groupby(train[parent])['碳排放强度'].mean()
            y=test['碳排放强度'].to_numpy(); pc=te.map(mc).fillna(train['碳排放强度'].mean()).to_numpy(); pb=test[parent].map(mb).fillna(train['碳排放强度'].mean()).to_numpy()
            se_c+=((y-pc)**2).sum();se_b+=((y-pb)**2).sum();n+=len(y)
    return (se_c/n)**.5,(se_b/n)**.5

def main():
    raw=pd.concat([read_trial(p,n) for p,n in FILES],ignore_index=True)
    # Same joint mean±3σ filter used in the prior common-model data; applied
    # after combining, so neither trip is given preferential treatment.
    fields=['速度','加速度','VSP','坡度','温度','湿度','碳排放强度']
    mask=np.ones(len(raw),dtype=bool)
    for c in fields:
        x=raw[c]; mask &= x.between(x.mean()-3*x.std(),x.mean()+3*x.std()).to_numpy()
    d=raw.loc[mask].copy().reset_index(drop=True)
    d['VSP骨架']=pd.cut(d['VSP'],[-np.inf,0,3,6,9,12,18,np.inf],labels=['VSP 负负荷（<0）','VSP 0–3','VSP 3–6','VSP 6–9','VSP 9–12','VSP 12–18','VSP ≥18'],right=False).astype(str)
    d['运动阶段']=phase(d);d['阶段修正']=phase_correct(d)
    d['坡度区间']=pd.cut(d['坡度'],[-np.inf,-1,1,np.inf],labels=['下坡（<-1°）','平坡（−1°至1°）','上坡（>1°）'],right=False).astype(str)
    rows=[]; choice={}
    for base,g in d.groupby('VSP骨架',sort=False):
        candidates={'VSP骨架':g['VSP骨架'], '运动阶段修正':g['VSP骨架']+'｜'+g['阶段修正'], '运动阶段＋坡度修正':g['VSP骨架']+'｜'+g['阶段修正']+'｜'+g['坡度区间']}
        tmp=[]
        baseline_sd=weighted_sd(g['碳排放强度'],g['VSP骨架'])
        for name,label in candidates.items():
            x=g.copy();x['_cand']=label.astype(str);x['_parent']=g['VSP骨架'].astype(str)
            counts=x.groupby('_cand').size();coverage=counts[counts>=MIN_N].sum()/len(x)
            final=x['_cand'].where(x['_cand'].map(counts)>=MIN_N,x['_parent'])
            ws=weighted_sd(x['碳排放强度'],final)
            if name=='VSP骨架': cv=np.nan;base_rmse=np.nan
            else: cv,base_rmse=block_cv(x,'_cand','_parent')
            tmp.append({'VSP骨架':base,'候选修正':name,'样本数':len(x),'有效叶节点覆盖率':coverage,'组内加权标准差':ws,'连续区段RMSE':cv,'VSP基线RMSE':base_rmse,'离散度下降比例':(baseline_sd-ws)/baseline_sd,'RMSE改善比例':(base_rmse-cv)/base_rmse if name!='VSP骨架' else np.nan})
        t=pd.DataFrame(tmp); eligible=t[(t['候选修正']!='VSP骨架')&(t['有效叶节点覆盖率']>=.8)&(t['离散度下降比例']>=.05)&(t['RMSE改善比例']>=.01)]
        chosen=eligible.sort_values('连续区段RMSE').iloc[0]['候选修正'] if len(eligible) else 'VSP骨架';t['是否入选']=t['候选修正'].eq(chosen);choice[base]=chosen;rows.append(t)
    tests=pd.concat(rows,ignore_index=True)
    def final(r):
        b=r['VSP骨架'];x=choice[b]
        if x=='运动阶段修正':return b+'｜'+r['阶段修正']
        if x=='运动阶段＋坡度修正':return b+'｜'+r['阶段修正']+'｜'+r['坡度区间']
        return b
    d['最终工况']=d.apply(final,axis=1)
    n=d.groupby('最终工况').size();d['最终工况']=d['最终工况'].where(d['最终工况'].map(n)>=MIN_N,d['VSP骨架'])
    f=d.groupby('最终工况')['碳排放强度'].agg(样本数='size',均值='mean',中位数='median',标准差='std',Q1=lambda x:x.quantile(.25),Q3=lambda x:x.quantile(.75)).reset_index();f['四分位距']=f.Q3-f.Q1;f['变异系数']=f['标准差']/f['均值'];f=f.sort_values('均值',ascending=False)
    # New-vs-old factor comparison uses the earlier selected library when labels overlap.
    old=pd.read_csv(ROOT/'pems_vsp_conditional_target_library_final_factor_library.csv')[['最终工况','均值']].rename(columns={'均值':'原4月均值'})
    compare=f[['最终工况','均值','样本数']].merge(old,on='最终工况',how='outer').rename(columns={'均值':'合并后均值','样本数':'合并后样本数'})
    # Target conditions: high VSP; stop-go if selected low-load start factor exceeds steady.
    target=[]
    for state in ['VSP ≥18','VSP 12–18','VSP 9–12']:
        r=f[f['最终工况']==state]
        if len(r):
            r=r.iloc[0]; scene='快速路/高速公路高负荷行驶、长上坡、加速并线和匝道汇入段' if state=='VSP ≥18' else '快速路或主干路的中高负荷巡航、上坡和加速并线段'
            target.append({'靶点类型':'高负荷行驶靶点','工况':state,'样本数':r['样本数'],'平均CO2（g/s）':r['均值'],'候选空间区域':scene})
    st=f[f['最终工况'].str.contains('起步/加速',regex=False)]; steady=f[f['最终工况'].str.contains('｜匀速',regex=False)]
    for _,r in st.iterrows():
        base=r['最终工况'].split('｜')[0];q=steady[steady['最终工况'].str.startswith(base+'｜')]
        if len(q):
            ref=q.iloc[0]; inc=(r['均值']/ref['均值']-1)*100
            if inc>0:target.append({'靶点类型':'频繁启停靶点','工况':r['最终工况'],'样本数':r['样本数'],'平均CO2（g/s）':r['均值'],'候选空间区域':f'较同VSP匀速高{inc:.1f}%；信号交叉口放行段、排队消散段、匝道汇入段'})
    targets=pd.DataFrame(target)
    # files
    raw.to_csv(ROOT/(P+'_all_raw_processed_samples.csv'),index=False,encoding='utf-8-sig');d.to_csv(ROOT/(P+'_labeled_samples.csv'),index=False,encoding='utf-8-sig');tests.to_csv(ROOT/(P+'_selection_tests.csv'),index=False,encoding='utf-8-sig');f.to_csv(ROOT/(P+'_final_factor_library.csv'),index=False,encoding='utf-8-sig');compare.to_csv(ROOT/(P+'_april_vs_combined_comparison.csv'),index=False,encoding='utf-8-sig');targets.to_csv(ROOT/(P+'_candidate_targets.csv'),index=False,encoding='utf-8-sig')
    # figure
    font='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
    if Path(font).exists():plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':11})
    show=f.sort_values('中位数');high=set(['VSP ≥18','VSP 12–18','VSP 9–12']);colors=['#BF6F72' if x in high else '#7AA6C2' for x in show['最终工况']]
    fig,ax=plt.subplots(figsize=(12.5,12.5),dpi=220);ax.barh(range(len(show)),show['中位数'],xerr=[show['中位数']-show.Q1,show.Q3-show['中位数']],color=colors,alpha=.86,ecolor='#526A7A',capsize=3)
    ax.set_yticks(range(len(show)),[x.replace('｜','  |  ') for x in show['最终工况']],fontsize=9.5);ax.set_xlabel('瞬时 CO$_2$ 排放强度（g·s$^{-1}$，柱为中位数）',fontsize=13);ax.set_title('合并两次 PEMS 行程后的 VSP 条件修正工况因子',fontsize=16,weight='bold');ax.grid(axis='x',color='#DCE6F1');ax.set_axisbelow(True);ax.text(.99,.02,'红色：高负荷靶点工况\n误差线：第1至第3四分位数',transform=ax.transAxes,ha='right',va='bottom',bbox={'facecolor':'white','edgecolor':'#9AA5AE','alpha':.92});fig.tight_layout();fig.savefig(ROOT/(P+'.png'),bbox_inches='tight');fig.savefig(ROOT/(P+'.pdf'),bbox_inches='tight');plt.close(fig)
    report=f'''# 合并两次 PEMS 行程的细分运行工况研究\n\n本轮将2025年4月与2025年5月两次实测行程以相同方法处理。速度经5点中值滤波；加速度由逐秒速度求导；坡度依据平滑高程与至少10 m的水平距离计算；VSP采用 `v(1.1a+9.81sinθ+0.132)+0.000302v³`。随后在合并数据上，对速度、加速度、VSP、坡度、温度、湿度及CO₂统一执行均值±3σ筛选。\n\n原始有效记录：4月 {len(raw[raw.行程=='2025年4月行程'])} 条，5月 {len(raw[raw.行程=='2025年5月行程'])} 条；合并清洗后 {len(d)} 条。\n\n工况选择规则：以7个样本充足的VSP骨架为基础；仅当运动阶段或运动阶段＋坡度修正满足叶节点≥{MIN_N}秒、覆盖率≥80%、组内离散度下降≥5%且分行程连续留出RMSE改善≥1%时保留。结果见 `*_selection_tests.csv`，因子见 `*_final_factor_library.csv`。\n\n靶点含义：表中靶点为高排放运行场景，尚非地图上已定位热点；应与视频轨迹、道路ID、流量及信号相位匹配后计算路段总排放。\n'''
    (ROOT/(P+'_report.md')).write_text(report,encoding='utf-8')
    print('RAW/FINAL',raw.groupby('行程').size().to_dict(),len(d));print('\nSELECTED\n',tests[tests.是否入选].to_string(index=False));print('\nFACTORS\n',f.to_string(index=False));print('\nTARGETS\n',targets.to_string(index=False));print('\nCOMPARE\n',compare.to_string(index=False))
if __name__=='__main__':main()
