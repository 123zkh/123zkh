"""Select a parsimonious VSP-skeleton operating-mode factor library and infer target road scenarios.

Selection is based on the common PEMS-preprocessed data. Candidate corrections
are accepted only if: training leaf n >= 50 s, >=80% coverage in its VSP bin,
within-bin weighted SD falls by >=5%, and five chronological holdouts improve
RMSE by >=1% versus VSP-only.  This is a screening study, not a claim of
network hotspot locations because this data table has no route/traffic fields.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-target-modes')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/cache-target-modes')
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent
D0=pd.read_csv(ROOT/'pems_refined_operating_modes_labeled_samples.csv')
OUT=ROOT/'pems_vsp_conditional_target_library'
MIN_N=50
FOLDS=5

# More stable phase grouping for factor assignment.  Start/acceleration remains
# separate because it is the desired stop-go signal; braking/coasting are low
# load recovery modes.  Steady is the reference state.
def phase_group(d):
    mp={'怠速/停车':'怠速/停车','制动':'制动/滑行','滑行':'制动/滑行',
        '起步加速':'起步/加速','加速':'起步/加速','匀速/轻微波动':'匀速'}
    return d['运动阶段'].astype(str).map(mp)

def weighted_sd(y,lab):
    q=pd.DataFrame({'y':y,'g':lab}).groupby('g')['y'].agg(['size','std'])
    return np.sqrt(np.average(q['std'].fillna(0)**2,weights=q['size']))

def sparse_to_parent(train,test,candidate,parent):
    counts=train.groupby(candidate).size()
    ok=counts[counts>=MIN_N].index
    tr=candidate.where(candidate.isin(ok),parent).astype(str)
    te=test[candidate.name].where(test[candidate.name].isin(ok),test[parent.name]).astype(str)
    return tr,te,ok

def cv_rmse(d, candidate, parent):
    # Test blocks are consecutive pieces of the retained time sequence.
    idx=np.array_split(np.arange(len(d)),FOLDS)
    sse_c=sse_p=nobs=0
    for ti in idx:
        test=d.iloc[ti]; train=d.drop(index=ti)
        tr_lab,te_lab,_=sparse_to_parent(train,test,train[candidate],train[parent])
        mu=train.groupby(tr_lab)['碳排放强度'].mean()
        pr_c=te_lab.map(mu).fillna(train['碳排放强度'].mean()).to_numpy()
        mu_p=train.groupby(train[parent])['碳排放强度'].mean()
        pr_p=test[parent].map(mu_p).fillna(train['碳排放强度'].mean()).to_numpy()
        y=test['碳排放强度'].to_numpy(); sse_c+=((y-pr_c)**2).sum();sse_p+=((y-pr_p)**2).sum();nobs+=len(y)
    return (sse_c/nobs)**.5,(sse_p/nobs)**.5

def candidate_stats(group):
    base=group['VSP骨架'].iloc[0]
    candidates={
        'VSP骨架':group['VSP骨架'],
        '运动阶段修正':group['VSP骨架']+'｜'+group['阶段修正'],
        '运动阶段＋坡度修正':group['VSP骨架']+'｜'+group['阶段修正']+'｜'+group['坡度区间'].astype(str),
    }
    records=[]
    for name,lab in candidates.items():
        temp=group.copy(); temp['candidate']=lab.astype(str); temp['parent']=group['VSP骨架'].astype(str)
        counts=temp.groupby('candidate').size()
        coverage=counts[counts>=MIN_N].sum()/len(temp)
        # The sd is calculated for retained leaves; sparse leaves remain at the parent.
        chosen=temp['candidate'].where(temp['candidate'].map(counts)>=MIN_N,temp['parent'])
        ws=weighted_sd(temp['碳排放强度'],chosen)
        # cv functions need named columns, called on the VSP-bin subset
        tmp=temp.assign(_cand=temp['candidate'],_parent=temp['parent'])
        c,p=cv_rmse(tmp,'_cand','_parent') if name!='VSP骨架' else (np.nan,np.nan)
        records.append({'VSP骨架':base,'候选修正':name,'样本数':len(group),'有效叶节点覆盖率':coverage,
                        '组内加权标准差':ws,'连续区段RMSE':c,'VSP基线RMSE':p})
    tab=pd.DataFrame(records)
    base_sd=tab.loc[tab['候选修正']=='VSP骨架','组内加权标准差'].iloc[0]
    # Select simple correction first; grade extension must add incremental CV value.
    tab['离散度下降比例']=(base_sd-tab['组内加权标准差'])/base_sd
    tab['RMSE改善比例']=(tab['VSP基线RMSE']-tab['连续区段RMSE'])/tab['VSP基线RMSE']
    valid=tab[(tab['候选修正']!='VSP骨架')&(tab['有效叶节点覆盖率']>=.80)&(tab['离散度下降比例']>=.05)&(tab['RMSE改善比例']>=.01)]
    if len(valid):
        # Prefer best RMSE result but do not accept no-clear-change states.
        selected=valid.sort_values(['连续区段RMSE','候选修正']).iloc[0]['候选修正']
    else: selected='VSP骨架'
    tab['是否入选']=tab['候选修正'].eq(selected)
    return tab

def main():
    d=D0.copy()
    d['阶段修正']=phase_group(d)
    d['VSP骨架']=pd.cut(d['VSP'], [-np.inf,0,3,6,9,12,18,np.inf], labels=['VSP 负负荷（<0）','VSP 0–3','VSP 3–6','VSP 6–9','VSP 9–12','VSP 12–18','VSP ≥18'], right=False).astype(str)
    # The upper VSP tail is combined with 18–24 because the original 24–30 and ≥30 bins have fewer than 50 seconds.
    d=d[d['VSP骨架'].ne('nan')].copy()
    tabs=[]
    for _,g in d.groupby('VSP骨架',sort=False): tabs.append(candidate_stats(g.reset_index(drop=True)))
    sel=pd.concat(tabs,ignore_index=True)
    sel.to_csv(OUT.with_name(OUT.name+'_selection_tests.csv'),index=False,encoding='utf-8-sig')
    chosen=dict(sel.loc[sel['是否入选'],['VSP骨架','候选修正']].values)
    def final_state(row):
        b=row['VSP骨架']; c=chosen[b]
        if c=='运动阶段修正': return b+'｜'+row['阶段修正']
        if c=='运动阶段＋坡度修正': return b+'｜'+row['阶段修正']+'｜'+str(row['坡度区间'])
        return b
    d['最终工况']=d.apply(final_state,axis=1)
    # Merge small final leaves back to VSP skeleton.
    cnt=d.groupby('最终工况').size(); d['最终工况']=d['最终工况'].where(d['最终工况'].map(cnt)>=MIN_N,d['VSP骨架'])
    fac=d.groupby('最终工况')['碳排放强度'].agg(样本数='size',均值='mean',中位数='median',标准差='std',Q1=lambda x:x.quantile(.25),Q3=lambda x:x.quantile(.75)).reset_index()
    fac['四分位距']=fac['Q3']-fac['Q1'];fac['变异系数']=fac['标准差']/fac['均值']
    fac=fac.sort_values('均值',ascending=False)
    fac.to_csv(OUT.with_name(OUT.name+'_final_factor_library.csv'),index=False,encoding='utf-8-sig')
    d.to_csv(OUT.with_name(OUT.name+'_labeled_samples.csv'),index=False,encoding='utf-8-sig')

    # Target scenarios: evidence-led operating patterns, not asserted physical locations.
    target=[]
    high_rows=fac[fac['最终工况'].isin(['VSP ≥18','VSP 12–18','VSP 9–12'])]
    for _,r in high_rows.iterrows():
        if r['最终工况']=='VSP ≥18':
            scene='快速路/高速公路的高负荷行驶、长上坡、加速并线与匝道汇入段'
        else:
            scene='快速路或主干路的中高负荷巡航、加速并线及上坡段'
        target.append({'靶点类型':'高负荷行驶靶点','高排放工况':r['最终工况'],'样本数':r['样本数'],
                       '平均排放强度（g/s）':r['均值'],'相对比较':'最终工况中位于高排放层级',
                       '对应候选靶点区域':scene,'证据等级':'PEMS工况证据；空间位置需视频/GPS匹配确认'})
    start_mean=fac.loc[fac['最终工况']=='VSP 0–3｜起步/加速','均值'].iloc[0]
    steady_mean=fac.loc[fac['最终工况']=='VSP 0–3｜匀速','均值'].iloc[0]
    start_n=fac.loc[fac['最终工况']=='VSP 0–3｜起步/加速','样本数'].iloc[0]
    target.append({'靶点类型':'频繁启停靶点','高排放工况':'VSP 0–3｜起步/加速','样本数':start_n,
                   '平均排放强度（g/s）':start_mean,'相对比较':f'较相同VSP匀速状态高 {(start_mean/steady_mean-1)*100:.1f}%',
                   '对应候选靶点区域':'信号交叉口放行段、排队消散段、匝道汇入段和频繁启停路段',
                   '证据等级':'PEMS工况证据；空间位置需视频/GPS匹配确认'})
    tar=pd.DataFrame(target)
    tar.to_csv(OUT.with_name(OUT.name+'_candidate_target_scenarios.csv'),index=False,encoding='utf-8-sig')

    # Chart: final factor library (adequately sampled) and selected rule. 
    q=fac['均值'].quantile(.67)
    font='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
    if Path(font).exists(): plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':11})
    show=fac.sort_values('均值').tail(15)
    fig,ax=plt.subplots(figsize=(12,8),dpi=220)
    colors=['#BF6F72' if x>=q else '#7AA6C2' for x in show['均值']]
    ax.barh(range(len(show)),show['中位数'],xerr=[show['中位数']-show['Q1'],show['Q3']-show['中位数']],color=colors,alpha=.86,ecolor='#526A7A',capsize=3)
    ax.set_yticks(range(len(show)),[x.replace('｜','\n') for x in show['最终工况']],fontsize=8.8)
    ax.set_xlabel('瞬时 CO$_2$ 排放强度（g·s$^{-1}$，柱为中位数）',fontsize=13)
    ax.set_title('VSP 骨架与条件修正下的高排放运行工况',fontsize=16,weight='bold',pad=12)
    ax.grid(axis='x',color='#DCE6F1');ax.set_axisbelow(True)
    ax.text(.99,.02,'红色：平均排放处于最终工况的最高三分之一\n误差线：第 1 至第 3 四分位数',transform=ax.transAxes,ha='right',va='bottom',fontsize=10,bbox={'facecolor':'white','edgecolor':'#9AA5AE','alpha':.92})
    fig.tight_layout();fig.savefig(OUT.with_suffix('.png'),bbox_inches='tight');fig.savefig(OUT.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)

    total_states=len(fac); selected=sel[sel['是否入选']]
    report=f'''# VSP骨架—运动阶段与坡度条件修正工况库\n\n## 筛选规则\n\n以 VSP 分箱为骨架；在每个 VSP 区间内分别检验“运动阶段修正”和“运动阶段＋坡度修正”。仅当候选修正同时满足：叶节点样本量不少于 {MIN_N} 秒、有效叶节点覆盖率不少于80%、组内加权标准差下降不少于5%、五折连续区段留出 RMSE 相对 VSP 基线降低不少于1%时，才纳入因子库。未通过的状态回退为 VSP 骨架。\n\n## 筛选结果\n\n{sel.to_csv(index=False)}\n\n最终保留 {total_states} 个工况因子。由 PEMS 支持的靶点是“高排放运行场景”，而非已定位到地图坐标的热点；其空间落点必须由同步视频、GPS轨迹、信号和流量数据进一步匹配。\n\n## 候选靶点场景\n\n{tar.to_csv(index=False)}\n\n## 使用限制\n\n当前共同清洗数据没有道路ID、交通量、排队长度和信号相位。因此，本研究可以识别“何种运行状态应成为靶点”，不能据此断言“哪一个具体路口或路段已经是靶点”。后续将工况标签与视频识别出的车辆轨迹、道路位置和车流状态连接后，才可计算分路段排放热点。\n'''
    OUT.with_name(OUT.name+'_report.md').write_text(report,encoding='utf-8')
    print('SELECTED RULES\n',selected[['VSP骨架','候选修正','样本数','有效叶节点覆盖率','离散度下降比例','RMSE改善比例']].to_string(index=False))
    print('\nFINAL FACTORS\n',fac.to_string(index=False))
    print('\nTARGETS\n',tar.to_string(index=False))
if __name__=='__main__':main()
