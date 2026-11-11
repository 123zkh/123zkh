from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parent
font='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
if Path(font).exists():
    fp=font_manager.FontProperties(fname=font).get_name(); plt.rcParams['font.family']=fp
plt.rcParams.update({'axes.unicode_minus':False,'font.size':13})

# Illustrative data only: replace with detected video summaries.
data=pd.DataFrame([
 {'道路等级':'快速路','场景类型':'正常稳定行驶','车辆数':20,'平均速度(m/s)':15.7,'平均加速度(m/s²)':0.03},
 {'道路等级':'快速路','场景类型':'高VSP热点','车辆数':18,'平均速度(m/s)':20.0,'平均加速度(m/s²)':0.25},
 {'道路等级':'主干路','场景类型':'频繁启停热点','车辆数':16,'平均速度(m/s)':4.5,'平均加速度(m/s²)':0.12},
 {'道路等级':'次干路','场景类型':'一般扰动行驶','车辆数':14,'平均速度(m/s)':8.0,'平均加速度(m/s²)':0.05},
 {'道路等级':'支路','场景类型':'低速稳定行驶','车辆数':10,'平均速度(m/s)':4.0,'平均加速度(m/s²)':0.01},
])
v=data['平均速度(m/s)']; a=data['平均加速度(m/s²)']
data['单车平均排放(g/s)']=0.4699+0.0151*v+0.0022*v**2+0.2115*a+0.7217*a**2+0.1379*v*a
data['场景总排放(g/s)']=data['车辆数']*data['单车平均排放(g/s)']
data.to_csv(ROOT/'交通视频场景排放计算示例数据.csv',index=False,encoding='utf-8-sig')

# Plot
colors=['#5C8DAE','#D98282','#C58A52','#79A887','#8D7BB8']
fig,axs=plt.subplots(1,2,figsize=(15,7),dpi=220)
labels=[f"{r['道路等级']}\n{r['场景类型']}" for _,r in data.iterrows()]
axs[0].bar(labels,data['单车平均排放(g/s)'],color=colors,alpha=.9)
axs[0].set_ylabel('单车平均 CO$_2$ 排放强度（g·s$^{-1}$）',fontsize=15)
axs[0].set_title('不同交通场景的单车排放示例',fontsize=18,weight='bold')
axs[1].bar(labels,data['场景总排放(g/s)'],color=colors,alpha=.9)
axs[1].set_ylabel('场景总 CO$_2$ 排放强度（g·s$^{-1}$）',fontsize=15)
axs[1].set_title('不同交通场景的总排放示例',fontsize=18,weight='bold')
for ax in axs:
    ax.tick_params(axis='x',labelsize=11); ax.tick_params(axis='y',labelsize=12)
    ax.grid(axis='y',color='#DCE6F1'); ax.set_axisbelow(True); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('交通视频识别参数代入速度—加速度排放公式的示例结果',fontsize=21,weight='bold',y=.99)
fig.text(.5,.01,'注：本图为方法演示数据，实际结果应替换为视频识别得到的车辆参数。',ha='center',fontsize=12,color='#555555')
fig.tight_layout(rect=[0,.06,1,.94])
fig.savefig(ROOT/'交通视频场景排放计算示例图.png',bbox_inches='tight')
fig.savefig(ROOT/'交通视频场景排放计算示例图.pdf',bbox_inches='tight')
plt.close(fig)

# Word document helpers
def set_cell_shading(cell, fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),fill); tcPr.append(shd)
def set_cell_text(cell,text,bold=False):
    cell.text=''; p=cell.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=p.add_run(str(text)); r.bold=bold; r.font.name='宋体'; r.font.size=Pt(9)
    cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER

def add_table(doc, headers, rows, widths=None):
    t=doc.add_table(rows=1, cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style='Table Grid'
    for i,h in enumerate(headers): set_cell_text(t.rows[0].cells[i],h,True); set_cell_shading(t.rows[0].cells[i],'D9EAF7')
    for row in rows:
        cells=t.add_row().cells
        for i,val in enumerate(row): set_cell_text(cells[i],val)
    return t

doc=Document(); sec=doc.sections[0]; sec.top_margin=Inches(.7); sec.bottom_margin=Inches(.7); sec.left_margin=Inches(.8); sec.right_margin=Inches(.8)
styles=doc.styles
styles['Normal'].font.name='宋体'; styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); styles['Normal'].font.size=Pt(10.5)
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=p.add_run('交通视频场景划分与碳排放计算示例'); r.bold=True; r.font.size=Pt(18); r.font.name='黑体'
p=doc.add_paragraph('本文件用于说明交通视频识别结果如何整理，并将车辆运行参数代入既有速度—加速度碳排放拟合公式。文中示例数值为方法演示数据，不代表最终实测结果。')
p.paragraph_format.first_line_indent=Pt(21)

doc.add_heading('一、视频场景聚合划分',level=1)
doc.add_paragraph('考虑到交通视频数量有限，场景不进行过度拆分，统一聚合为正常稳定行驶、频繁启停热点、高 VSP 热点和一般扰动行驶四类。场景识别按照“频繁启停优先、高 VSP 次之、其余按运行稳定性归类”的顺序执行。')
add_table(doc,['道路等级','正常稳定行驶','频繁启停热点','高 VSP 热点','一般扰动行驶'],[
 ['快速路','主线非拥堵路段','拥堵、匝道汇入或排队消散段','高车速、上坡或匝道加速段','车道交织或轻度拥堵段'],
 ['主干路','交叉口间稳定通行路段','信号交叉口进口道及放行段','连续加速、上坡段','公交停靠、转向或跟驰干扰段'],
 ['次干路','非高峰期普通路段','中小型交叉口排队段','坡道或放行后加速段','路侧停车、行人和非机动车干扰段'],
 ['支路','低流量稳定通行路段','无信号交叉口或局部排队段','坡度较大或短距离加速段','低速跟驰和临时避让段']])

doc.add_heading('二、图像识别参数及数据表',level=1)
doc.add_paragraph('对每个视频帧或时间步，识别车辆编号、车型、能源类型、车辆位置和场景标签；由连续帧位置计算速度，由速度序列计算加速度。车辆数量、车型结构和新能源车比例用于描述交通组成，速度、加速度和坡度用于计算 VSP 及瞬时排放。')
add_table(doc,['时间/s','车辆编号','道路等级','场景','车型','能源类型','速度/(m·s⁻¹)','加速度/(m·s⁻²)','坡度/°','VSP/(kW·t⁻¹)'],[
 ['1','C001','主干路','正常稳定','小客车','燃油车','9.8','0.03','0.4','2.16'],
 ['2','C001','主干路','正常稳定','小客车','燃油车','10.1','0.05','0.4','2.38'],
 ['3','C001','主干路','频繁启停','小客车','燃油车','6.2','0.31','0.4','2.41'],
 ['1','C002','主干路','频繁启停','SUV','新能源车','2.1','0.00','0.2','0.31']])

doc.add_heading('三、速度—加速度多项式排放计算',level=1)
doc.add_paragraph('将视频识别得到的速度和加速度统一转换为与第三章 PEMS 拟合模型一致的单位，并代入速度—加速度二次多项式：')
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; rr=p.add_run('E = 0.4699 + 0.0151v + 0.0022v² + 0.2115a + 0.7217a² + 0.1379va'); rr.bold=True; rr.font.size=Pt(12)
doc.add_paragraph('式中，E 为单车瞬时 CO₂ 排放强度（g·s⁻¹），v 为速度（m·s⁻¹），a 为加速度（m·s⁻²）。对同一时间步内检测到的所有车辆分别计算 E，再求和得到场景总排放强度；对连续视频时段积分或按时间步累加，可得到场景排放总量。')
doc.add_paragraph('例如，某主干路频繁启停场景识别得到平均速度 v=4.5 m·s⁻¹、平均加速度 a=0.12 m·s⁻²，则单车平均排放为：E=0.4699+0.0151×4.5+0.0022×4.5²+0.2115×0.12+0.7217×0.12²+0.1379×4.5×0.12≈0.65 g·s⁻¹。若同一时间步有 16 辆车，则场景总排放强度约为 16×0.65=10.4 g·s⁻¹。正式计算时应使用每辆车每个时间步的参数，而不是仅使用场景平均值。')

doc.add_heading('四、示例场景排放结果',level=1)
doc.add_picture(str(ROOT/'交通视频场景排放计算示例图.png'),width=Inches(6.8))
doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph('图X.X 交通视频识别参数代入速度—加速度排放公式的示例结果。注：图中数据为方法演示数据，实际论文结果应替换为真实视频识别结果。')
rows=[]
for _,r in data.iterrows():
    rows.append([r['道路等级'],r['场景类型'],int(r['车辆数']),f"{r['平均速度(m/s)']:.2f}",f"{r['平均加速度(m/s²)']:.2f}",f"{r['单车平均排放(g/s)']:.3f}",f"{r['场景总排放(g/s)']:.3f}"])
add_table(doc,['道路等级','场景类型','车辆数','平均速度/(m·s⁻¹)','平均加速度/(m·s⁻²)','单车排放/(g·s⁻¹)','场景总排放/(g·s⁻¹)'],rows)

doc.add_heading('五、正式研究中的汇总方式',level=1)
doc.add_paragraph('正式分析时，首先按照道路等级和四类聚合场景统计车辆数量及交通参数；随后对每辆车的轨迹逐时间步计算瞬时排放，并在时间尺度上累加，在空间尺度上按照道路长度汇总。最终可形成“道路等级—场景类型—车辆组成—平均排放—总排放—单位长度排放”的结果表，并以柱状图、箱线图和速度—时间轨迹图展示不同场景之间的差异。')

out=ROOT/'交通视频场景识别与碳排放计算示例.docx'; doc.save(out)
print(out)
