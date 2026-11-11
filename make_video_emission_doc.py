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
fig.suptitle('交通视频识别参数对应场景排放示例',fontsize=21,weight='bold',y=.99)
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
p=doc.add_paragraph('本文件用于说明交通视频识别结果如何整理，并将识别得到的车辆运行参数输入待确定的碳排放模型。文中示例数值为方法演示数据，不代表最终实测结果。')
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

doc.add_heading('三、视频识别参数到排放结果的计算流程',level=1)
doc.add_paragraph('视频识别完成后，先将车辆编号、车型、能源类型、速度、加速度、坡度和场景标签整理为逐车辆、逐时间步的数据表。随后根据论文最终确定的碳排放模型，对每辆车每个时间步计算瞬时排放强度；同一时间步内将所有车辆排放相加，得到场景排放强度；再对连续视频时段进行时间累加，得到场景排放总量。若需要比较不同道路路段，则进一步按照路段长度计算单位长度排放。')
doc.add_paragraph('本阶段文档不预先限定具体碳排放公式。正式计算时应保证视频输入变量的单位、时间间隔和车辆筛选规则与模型训练数据保持一致，并在结果表中同时保留车辆数量、车型结构、平均速度、速度离散度、平均加速度、停车比例和排放结果。')

doc.add_heading('四、示例场景排放结果',level=1)
doc.add_picture(str(ROOT/'交通视频场景排放计算示例图.png'),width=Inches(6.8))
doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph('图X.X 交通视频识别参数对应场景排放示例。注：图中数据为方法演示数据，实际论文结果应替换为真实视频识别结果。')
rows=[]
for _,r in data.iterrows():
    rows.append([r['道路等级'],r['场景类型'],int(r['车辆数']),f"{r['平均速度(m/s)']:.2f}",f"{r['平均加速度(m/s²)']:.2f}",f"{r['单车平均排放(g/s)']:.3f}",f"{r['场景总排放(g/s)']:.3f}"])
add_table(doc,['道路等级','场景类型','车辆数','平均速度/(m·s⁻¹)','平均加速度/(m·s⁻²)','单车排放/(g·s⁻¹)','场景总排放/(g·s⁻¹)'],rows)

doc.add_heading('五、正式研究中的汇总方式',level=1)
doc.add_paragraph('正式分析时，首先按照道路等级和四类聚合场景统计车辆数量及交通参数；随后对每辆车的轨迹逐时间步计算瞬时排放，并在时间尺度上累加，在空间尺度上按照道路长度汇总。最终可形成“道路等级—场景类型—车辆组成—平均排放—总排放—单位长度排放”的结果表，并以柱状图、箱线图和速度—时间轨迹图展示不同场景之间的差异。')

doc.add_heading('六、推荐论文中最终保留的图表组合',level=1)
doc.add_paragraph('在视频样本数量有限的情况下，建议保留以下七类图表，既能展示识别结果，也能支撑场景排放比较：')
add_table(doc,['编号','建议图表','主要内容','展示目的'],[
 ['1','表X.X 车辆识别统计表','道路等级、场景、视频时长、车辆总数、燃油车/新能源车、车型数量','说明视频样本和车辆组成'],
 ['2','图X.X 车型及能源类型堆叠柱状图','小客车、SUV、货车及燃油车、新能源车比例','比较不同场景交通组成'],
 ['3','图X.X 车速和加速度箱线图','各场景速度、加速度及离散程度','识别正常、启停和高 VSP 场景的运行差异'],
 ['4','图X.X 典型车辆速度—时间轨迹图','连续帧跟踪得到的单车速度变化曲线','展示停车、起步、加速和稳定巡航过程'],
 ['5','图X.X 道路等级—场景类型热力图','各场景车辆数量或车辆占比','识别不同道路等级的主要交通场景'],
 ['6','表X.X 逐车辆逐时间步排放计算输入表','车辆编号、车型、能源类型、速度、加速度、坡度、场景标签','说明排放模型的输入来源'],
 ['7','表X.X 场景和路段排放汇总表','平均排放、场景总排放、单位长度排放和单位时间排放','比较不同场景及路段的排放效果']])
doc.add_paragraph('其中，第1、2、3和5项用于展示交通视频识别结果，第4项用于验证车辆轨迹和运行状态识别，第6项用于说明排放计算过程，第7项用于呈现最终排放结果。正式论文可根据视频数量删减第2或第5项，但建议保留逐车辆输入表和场景排放汇总表。')

doc.add_heading('七、示例结果的使用说明',level=1)
doc.add_paragraph('示例图和示例数据仅用于演示“视频参数—场景汇总—排放结果”的组织方式，不作为实测结论。待排放模型最终确定后，只需替换逐车辆参数表和模型输出，即可生成正式结果图和汇总表。')

out=ROOT/'交通视频场景识别与碳排放计算示例.docx'; doc.save(out)
print(out)
