from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent
fp='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'; font_manager.fontManager.addfont(fp)
plt.rcParams['font.family']=font_manager.FontProperties(fname=fp).get_name(); plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(13,8),dpi=220); ax.set_xlim(0,13); ax.set_ylim(0,10); ax.axis('off')

def box(x,y,w,h,text,fs=17):
    ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor='black',lw=1.8))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs)
def arrow(x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=1.6,color='black'))
# top and main chain
box(4.6,9.0,3.8,.65,'交通视频输入')
box(4.6,7.55,3.8,.65,'建立背景模型')
box(4.6,6.10,3.8,.65,'差值运算')
box(4.6,4.65,3.8,.65,'前景二值图')
box(4.6,3.20,3.8,.65,'连通域判断')
arrow(6.5,9,6.5,8.2); arrow(6.5,7.55,6.5,6.75); arrow(6.5,6.1,6.5,5.3); arrow(6.5,4.65,6.5,3.85)
# branch after connected components
box(1.15,1.55,3.8,.7,'去除噪声与小区域')
box(8.05,1.55,3.8,.7,'筛选有效运动目标')
arrow(5.2,3.2,3.05,2.25); arrow(7.8,3.2,9.95,2.25)
# merge lower
box(4.0,.15,5.0,.7,'结果保存：目标位置、数量与轨迹')
arrow(3.05,1.55,5.2,.85); arrow(9.95,1.55,7.8,.85)
# branch details
ax.text(1.15,1.1,'连通域面积、形状及位置判断',ha='left',va='top',fontsize=13)
ax.text(8.05,1.1,'保留满足阈值的车辆运动目标',ha='left',va='top',fontsize=13)
fig.savefig(ROOT/'vehicle_motion_detection_text_flow.png',bbox_inches='tight',facecolor='white'); fig.savefig(ROOT/'vehicle_motion_detection_text_flow.pdf',bbox_inches='tight',facecolor='white'); plt.close(fig)
