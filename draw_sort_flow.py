from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parent
fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; font_manager.fontManager.addfont(fp)
plt.rcParams['font.family']=font_manager.FontProperties(fname=fp).get_name(); plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(14,9),dpi=220); ax.set_xlim(0,14); ax.set_ylim(0,10); ax.axis('off')
def box(x,y,w,h,text,fs=16):
 ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor='black',lw=1.8))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,wrap=True)
def line(points,arrow_end=True):
 xs,ys=zip(*points); ax.plot(xs,ys,color='black',lw=1.6)
 if arrow_end:
  x1,y1=points[-2]; x2,y2=points[-1]
  ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=1.6,color='black'))
# main flow
main=(4.5,3.8)
box(4.5,9.0,5.0,.65,'视频帧输入')
box(4.5,7.65,5.0,.65,'目标检测结果')
box(4.5,6.30,5.0,.65,'卡尔曼滤波预测')
box(4.5,4.95,5.0,.65,'匈牙利算法匹配')
box(4.5,3.60,5.0,.65,'轨迹状态更新')
for y in [9.0,7.65,6.30,4.95]: line([(7.0,y),(7.0,y-.7)])
# side processing boxes, aligned with relevant steps
box(.65,6.15,3.0,1.0,'预测目标位置\n与检测框',14)
line([(4.5,6.62),(3.65,6.62)])
box(10.35,6.15,3.0,1.0,'计算交并比\n构造代价矩阵',14)
line([(9.5,5.28),(10.35,6.62)])
# branch after update
box(.8,1.85,3.5,.8,'匹配成功\n更新轨迹',14)
box(5.25,1.85,3.5,.8,'未匹配轨迹\n暂存或删除',14)
box(9.7,1.85,3.5,.8,'未匹配目标\n新建轨迹',14)
line([(5.1,3.6),(2.55,2.65)]); line([(7.0,3.6),(7.0,2.65)]); line([(8.9,3.6),(11.45,2.65)])
box(3.35,.35,7.3,.72,'输出跟踪结果：目标编号、位置、速度与轨迹',15)
line([(2.55,1.85),(5.2,1.07)]); line([(7.0,1.85),(7.0,1.07)]); line([(11.45,1.85),(8.8,1.07)])
fig.savefig(ROOT/'sort_algorithm_framework.png',bbox_inches='tight',facecolor='white'); fig.savefig(ROOT/'sort_algorithm_framework.pdf',bbox_inches='tight',facecolor='white'); plt.close(fig)
