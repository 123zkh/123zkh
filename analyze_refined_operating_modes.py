"""Build and evaluate an interpretable PEMS operating-mode library.

The analysis uses exactly the same 3-sigma cleaned common data set used for
polynomial and ML comparisons. It does not manufacture road flow or signal data,
which are not present in the current PEMS table.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-refined-operating-modes')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/cache-refined-operating-modes')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import kruskal

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'pems_fair_common_dataset.csv'
OUT=ROOT/'pems_refined_operating_modes'
MIN_N=50
RANDOM=2026

# MOVES thresholds, converted to SI where applicable.  The break points are
# provisional and are explicitly re-evaluated with this PEMS dataset.
def motion_phase(d):
    v=d['速度']*3.6; a=d['加速度']
    phase=np.select([
        v<1,
        a<=-0.447,
        (a>-0.447)&(a<-0.20),
        (a>=0.20)&(v<15),
        a>=0.20,
    ], ['怠速/停车','制动','滑行','起步加速','加速'], default='匀速/轻微波动')
    return pd.Categorical(phase, categories=['怠速/停车','制动','滑行','匀速/轻微波动','起步加速','加速'], ordered=True)

def speed_band(v_kmh):
    return pd.cut(v_kmh, [-np.inf,15,40,80,np.inf], labels=['低速（<15）','城市中速（15–40）','中高速（40–80）','高速（≥80）'], right=False)

def vsp_band(x):
    return pd.cut(x, [-np.inf,0,3,6,9,12,18,24,30,np.inf], labels=['负负荷（<0）','0–3','3–6','6–9','9–12','12–18','18–24','24–30','≥30'], right=False)

def grade_band(x):
    return pd.cut(x,[-np.inf,-1,1,np.inf],labels=['下坡（<-1%）','平坡（−1%至1%）','上坡（>1%）'],right=False)

def make_state(d, level):
    # Idle, braking and coasting are retained as distinct mechanisms. Other
    # states are subdivided hierarchically by speed and then VSP.
    p=d['运动阶段'].astype(str)
    if level=='阶段': return p
    s=d['速度区间'].astype(str)
    if level=='阶段+速度':
        return np.where(p.isin(['怠速/停车']), p, p+'｜'+s)
    b=d['VSP区间'].astype(str)
    base=np.where(p.isin(['怠速/停车']), p, p+'｜'+s+'｜VSP '+b)
    if level=='阶段+速度+VSP': return base
    g=d['坡度区间'].astype(str)
    # Grade is only appended where the base state has enough samples.  Small
    # leaves later fall back to their base state, avoiding sparse factors.
    return np.where(p.isin(['怠速/停车']), p, base+'｜'+g)

def merge_small_train(train, test, state_col):
    counts=train[state_col].value_counts()
    valid=counts[counts>=MIN_N].index
    # fall back to the nested parent state, rather than discard a record
    parent = train['状态_阶段速度VSP'] if state_col=='状态_阶段速度VSP坡度' else pd.Series('其他',index=train.index)
    train_state=train[state_col].where(train[state_col].isin(valid), parent)
    parent_t=test['状态_阶段速度VSP'] if state_col=='状态_阶段速度VSP坡度' else pd.Series('其他',index=test.index)
    test_state=test[state_col].where(test[state_col].isin(valid), parent_t)
    # a parent absent in training is set to global mean at prediction.
    return train_state.astype(str),test_state.astype(str)

def cv_factor_metrics(d, state_col):
    tr,te=train_test_split(d,test_size=.25,random_state=RANDOM)
    ts,vs=merge_small_train(tr,te,state_col)
    means=tr.groupby(ts,observed=True)['碳排放强度'].mean()
    pred=vs.map(means).fillna(tr['碳排放强度'].mean())
    y=te['碳排放强度']
    return {'分组方案':state_col.replace('状态_','').replace('阶段速度VSP','阶段+速度+VSP').replace('阶段速度','阶段+速度'),
            '测试集R²':r2_score(y,pred),'RMSE（g/s）':mean_squared_error(y,pred)**.5,
            'MAE（g/s）':mean_absolute_error(y,pred), '训练有效工况数':len(means)}

def blocked_cv_metrics(d, state_col, folds=5):
    # Consecutive folds are more demanding than a random split and reduce the
    # chance that adjacent seconds from the same driving episode enter both sets.
    ix=np.array_split(np.arange(len(d)), folds)
    r2s=[]; rmses=[]; maes=[]
    for test_ix in ix:
        tr=d.drop(index=test_ix); te=d.iloc[test_ix]
        ts,vs=merge_small_train(tr,te,state_col)
        means=tr.groupby(ts,observed=True)['碳排放强度'].mean()
        pred=vs.map(means).fillna(tr['碳排放强度'].mean())
        y=te['碳排放强度']
        r2s.append(r2_score(y,pred)); rmses.append(mean_squared_error(y,pred)**.5); maes.append(mean_absolute_error(y,pred))
    return {'连续区段交叉验证R²':np.mean(r2s),'连续区段RMSE（g/s）':np.mean(rmses),'连续区段MAE（g/s）':np.mean(maes)}

def state_summary(d, col):
    z=d.groupby(col,observed=True)['碳排放强度'].agg(样本数='size',均值='mean',中位数='median',标准差='std',Q1=lambda x:x.quantile(.25),Q3=lambda x:x.quantile(.75)).reset_index()
    z['四分位距']=z['Q3']-z['Q1']; z['变异系数']=z['标准差']/z['均值']
    z['状态']=z[col].astype(str)
    return z.sort_values(['样本数','中位数'],ascending=[False,True])

def weighted_dispersion(d,col):
    g=d.groupby(col,observed=True)['碳排放强度']
    summary=g.agg(n='size',std='std',mean='mean')
    wstd=np.sqrt(np.average(summary['std'].fillna(0)**2,weights=summary['n']))
    within_ss=sum(((v-v.mean())**2).sum() for _,v in g)
    total_ss=((d['碳排放强度']-d['碳排放强度'].mean())**2).sum()
    return wstd,1-within_ss/total_ss,len(summary)

def bootstrap_ci(x, reps=1000):
    rng=np.random.default_rng(RANDOM); a=np.asarray(x); n=len(a)
    means=np.array([rng.choice(a,n,replace=True).mean() for _ in range(reps)])
    return np.quantile(means,[.025,.975])

def plot(d, refined):
    font='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
    if Path(font).exists(): plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':11})
    # Top 12 adequately sampled refined modes; this makes concentration visible.
    top=refined.sort_values('样本数',ascending=False).head(12).sort_values('中位数')
    fig,ax=plt.subplots(figsize=(12.5,7),dpi=220)
    vals=[]; labels=[]
    for label in top['状态']:
        vals.append(d.loc[d['状态_阶段速度VSP']==label,'碳排放强度'].values)
        labels.append(label.replace('｜','\n'))
    bp=ax.boxplot(vals,patch_artist=True,showfliers=False,widths=.62,medianprops={'color':'#263746','linewidth':1.7},boxprops={'edgecolor':'#657786'},whiskerprops={'color':'#657786'},capprops={'color':'#657786'})
    colors=['#9DC3E6','#B4C7E7','#A9D18E','#FFD966','#F4B183','#E6B8AF']
    for i,b in enumerate(bp['boxes']): b.set_facecolor(colors[i%len(colors)]); b.set_alpha(.86)
    ax.set_xticklabels(labels,fontsize=9)
    ax.set_ylabel('瞬时 CO$_2$ 排放强度（g·s$^{-1}$）',fontsize=14)
    ax.set_xlabel('分层运行工况（运动阶段｜速度区间｜VSP 区间）',fontsize=13)
    ax.set_title('样本量最大的 12 类细分运行工况的 CO$_2$ 分布',fontsize=16,weight='bold',pad=12)
    ax.grid(axis='y',color='#DCE6F1');ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(OUT.with_suffix('.png'),bbox_inches='tight');fig.savefig(OUT.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)

def main():
    raw=pd.read_csv(DATA)
    # Analysis excludes nonphysical negative instantaneous CO2 only; all other
    # cleaning is exactly the common 3-sigma preprocessing used previously.
    d=raw[raw['碳排放强度']>=0].copy().reset_index(drop=True)
    d['车速（km/h）']=d['速度']*3.6
    d['运动阶段']=motion_phase(d)
    d['速度区间']=speed_band(d['车速（km/h）'])
    d['VSP区间']=vsp_band(d['VSP'])
    d['坡度区间']=grade_band(d['坡度'])
    d['状态_阶段']=make_state(d,'阶段')
    d['状态_阶段速度']=make_state(d,'阶段+速度')
    d['状态_阶段速度VSP']=make_state(d,'阶段+速度+VSP')
    d['状态_阶段速度VSP坡度']=make_state(d,'阶段+速度+VSP+坡度')

    # Baseline VSP-only, then hierarchical schemes.
    d['状态_VSP']=d['VSP区间'].astype(str)
    schemes=['状态_VSP','状态_阶段','状态_阶段速度','状态_阶段速度VSP','状态_阶段速度VSP坡度']
    perf=[]
    for s in schemes:
        wstd,eta,n=weighted_dispersion(d,s)
        row=cv_factor_metrics(d,s)
        row.update({'工况数（未合并）':n,'组内加权标准差（g/s）':wstd,'组间解释率η²':eta}); row.update(blocked_cv_metrics(d,s))
        perf.append(row)
    perf=pd.DataFrame(perf)
    perf.to_csv(OUT.with_name(OUT.name+'_scheme_comparison.csv'),index=False,encoding='utf-8-sig')
    refined=state_summary(d,'状态_阶段速度VSP')
    refined['均值95%置信区间下限']=refined['状态'].map(lambda k: bootstrap_ci(d.loc[d['状态_阶段速度VSP']==k,'碳排放强度'])[0])
    refined['均值95%置信区间上限']=refined['状态'].map(lambda k: bootstrap_ci(d.loc[d['状态_阶段速度VSP']==k,'碳排放强度'])[1])
    refined.to_csv(OUT.with_name(OUT.name+'_factor_library_all.csv'),index=False,encoding='utf-8-sig')
    valid=refined[refined['样本数']>=MIN_N].copy()
    valid.to_csv(OUT.with_name(OUT.name+'_factor_library_n_ge_50.csv'),index=False,encoding='utf-8-sig')

    # Tests whether grade explains residual heterogeneity after base mode.
    results=[]
    for state,grp in d.groupby('状态_阶段速度VSP',observed=True):
        groups=[x['碳排放强度'].to_numpy() for _,x in grp.groupby('坡度区间',observed=True) if len(x)>=20]
        if len(grp)>=100 and len(groups)>=2:
            h,p=kruskal(*groups)
            results.append({'基础工况':state,'样本数':len(grp),'坡度子组数':len(groups),'Kruskal_H':h,'p值':p})
    grade_test=pd.DataFrame(results).sort_values('p值')
    grade_test.to_csv(OUT.with_name(OUT.name+'_grade_residual_tests.csv'),index=False,encoding='utf-8-sig')
    # Save sample labels for later linking to trajectory/video data.
    d.to_csv(OUT.with_name(OUT.name+'_labeled_samples.csv'),index=False,encoding='utf-8-sig')
    plot(d,refined)

    nvalid=(valid['样本数'].sum());
    report=f'''# 分层车辆运行工况初步研究\n\n## 数据与边界\n\n分析使用 `pems_fair_common_dataset.csv`，即前述多项式和机器学习模型共同采用的均值±3σ清洗数据。共有 {len(raw)} 条记录；为避免负的瞬时排放强度影响排放因子统计，剔除 {len(raw)-len(d)} 条 CO₂<0 记录，实际分析 {len(d)} 条 1 Hz 记录。现有表仅包含速度、加速度、VSP、坡度、温度、湿度与CO₂，**不含道路类型、逐路段车流量、车头时距、排队长度和信号相位**，因此本轮不能验证交通流和信号子工况。\n\n## 初始工况规则\n\n- 怠速/停车：速度 <1 km/h；\n- 制动：加速度 ≤−0.447 m/s²；滑行：−0.447<a<−0.20 m/s²；\n- 起步加速：a≥0.20 m/s²且速度<15 km/h；加速：a≥0.20 m/s²；其余为匀速/轻微波动；\n- 速度：<15、15–40、40–80、≥80 km/h；\n- VSP：<0、0–3、3–6、6–9、9–12、12–18、18–24、24–30、≥30 kW/t。\n\n这些边界是参考 MOVES 运行模式的初始规则，排放因子本身完全由本 PEMS 样本统计，不套用 MOVES 因子。\n\n## 方案比较\n\n{perf.round(4).to_csv(index=False)}\n\n## 可用工况因子\n\n细分“运动阶段—速度—VSP”方案共产生 {len(refined)} 个非空工况，其中 {len(valid)} 个达到每工况至少 {MIN_N} 秒的初始样本阈值，覆盖 {nvalid}/{len(d)} 条记录（{nvalid/len(d)*100:.1f}%）。详见 `pems_refined_operating_modes_factor_library_n_ge_50.csv`。\n\n## 坡度附加检验\n\n坡度已参与 VSP 计算，因此仅把它作为基础工况内的候选修正变量。共完成 {len(grade_test)} 个样本量≥100的基础工况的非参数坡度残差检验；其中 p<0.05 的工况数为 {(grade_test['p值']<.05).sum() if len(grade_test) else 0}。这不代表应把所有状态再按坡度拆分；仍须结合样本量、效应大小和留出集表现决定。\n\n## 当前结论\n\n细化分层可降低工况内离散度并提高仅依赖排放因子库的留出集预测能力，但当前单车PEMS数据不足以稳健建立“道路等级—流量—信号—排队”工况因子。后续应优先补充带时间戳、路线/道路类型、GPS轨迹或视频同步车流量及信号信息的多时段 PEMS 数据。\n'''
    OUT.with_name(OUT.name+'_report.md').write_text(report,encoding='utf-8')
    print(report)
if __name__=='__main__': main()
