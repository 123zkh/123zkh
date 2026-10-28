from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parent
fp='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'; font_manager.fontManager.addfont(fp)
plt.rcParams['font.family']=font_manager.FontProperties(fname=fp).get_name(); plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(13,8),dpi=220); ax.set_xlim(0,13); ax.set_ylim(0,10); ax.axis('off')
def box(x,y,w,h,text,fs=16):
 ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor='black',lw=1.8)); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs)
def arrow(x1,y1,x2,y2): ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=1.6,color='black'))
box(4.6,9.0,3.8,.65,'视频帧输入'); box(4.6,7.55,3.8,.65,'目标检测结果'); box(4.6,6.1,3.8,.65,'卡尔曼滤波预测'); box(4.6,4.65,3.8,.65,'匈牙利算法匹配'); box(4.6,3.2,3.8,.65,'轨迹状态更新')
for y in [9,7.55,6.1,4.65]: arrow(6.5,y,6.5,y-.8)
box(.6,5.95,3.2,1.0,'预测目标位置\n与检测框')
arrow(4.6,6.42,3.8,6.42)
box(8.9,5.95,3.2,1.0,'计算交并比\n构造代价矩阵')
arrow(8.4,4.98,8.9,6.42)
box(1.2,1.55,3.7,.7,'匹配成功：更新轨迹')
box(8.1,1.55,3.7,.7,'未匹配目标：新建轨迹')
arrow(5.2,3.2,3.05,2.25); arrow(7.8,3.2,9.95,2.25)
box(4.0,.15,5.0,.7,'轨迹管理与结果输出：目标编号、位置和轨迹')
arrow(3.05,1.55,5.2,.85); arrow(9.95,1.55,7.8,.85)
# return unmatched tracks loop
ax.annotate('',xy=(10.0,7.55),xytext=(11.9,2.25),arrowprops=dict(arrowstyle='->',lw=1.2,color='black',connectionstyle='arc3,rad=.25'))
ax.text(11.85,4.8,'保留未失配轨迹',rotation=90,ha='center',va='center',fontsize=13)
fig.savefig(ROOT/'sort_algorithm_framework.png',bbox_inches='tight',facecolor='white'); fig.savefig(ROOT/'sort_algorithm_framework.pdf',bbox_inches='tight',facecolor='white'); plt.close(fig)
