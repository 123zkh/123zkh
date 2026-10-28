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
box(4.6,9.0,3.8,.65,'连续视频帧输入'); box(4.6,7.55,3.8,.65,'读取相邻帧：I(t-1)、I(t)'); box(4.6,6.1,3.8,.65,'灰度化与去噪处理'); box(4.6,4.65,3.8,.65,'计算帧间绝对差值'); box(4.6,3.2,3.8,.65,'阈值分割得到前景');
for y in [9,7.55,6.1,4.65]: arrow(6.5,y,6.5,y-.8)
# equation side box
box(.6,5.8,3.2,1.3,'D(t,x,y)=|I(t,x,y)-I(t-1,x,y)|',13)
arrow(4.6,4.95,3.8,6.45)
box(.6,3.2,3.2,1.3,'B(t,x,y)=1，D(t,x,y)≥T\nB(t,x,y)=0，D(t,x,y)<T',13)
arrow(4.6,3.5,3.8,3.85)
# branch and merge
box(1.2,1.55,3.7,.7,'形态学处理：开运算、闭运算')
box(8.1,1.55,3.7,.7,'连通域分析与目标筛选')
arrow(5.2,3.2,3.05,2.25); arrow(7.8,3.2,9.95,2.25)
box(4.0,.15,5.0,.7,'输出车辆目标：位置、面积与轨迹')
arrow(3.05,1.55,5.2,.85); arrow(9.95,1.55,7.8,.85)
fig.savefig(ROOT/'adjacent_frame_difference_method_flow.png',bbox_inches='tight',facecolor='white'); fig.savefig(ROOT/'adjacent_frame_difference_method_flow.pdf',bbox_inches='tight',facecolor='white'); plt.close(fig)
