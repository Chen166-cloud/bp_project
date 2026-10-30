"""Create a concise figure-led group-meeting report from saved measurements."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'vancouver_bp/results'
DOCX=ROOT/'output/docx/BP与FFBP成像实验报告.docx'

def p(text):return dict(kind='p',text=text)
def heading(text):return dict(kind='heading',text=text)
def table(headers,rows):return dict(kind='table',headers=headers,rows=rows)
def image(name,caption):return dict(kind='image',path=OUT/name,caption=caption)
def page():return dict(kind='page')
def link(label,path):
    target=Path(os.path.relpath(Path(path).resolve(),OUT)).as_posix()
    return f'[{label}](<{target}>)'

def content(record):
    targets=record['targets'];fine=record['scene']
    shape=fine['bp_timing']['shape'];err=fine['fidelity']['relative_complex_error']*100
    blocks=[
      dict(kind='title',text='BP与FFBP成像对比实验'),
      dict(kind='subtitle',text='CDdata1数据片段与两个强散射点的质量评价  2026年10月31日'),
      p(f'在相同回波、几何和加窗条件下完成无 PGA 的 BP 与 FFBP 成像。最终场景为 {shape[0]} × {shape[1]} 像素，FFBP 相对 BP 的复数误差为 {err:.3f}%；两个强散射点在精细局部网格下的 IRW、PSLR、ISLR 基本重合。'),
      heading('一 数据与共同处理条件'),
      p('输入为项目已有 CDdata1.mat，共 1728 × 3200 个复数 I/Q 样本；采用 RADARSAT.m 中的 RADARSAT-1 等效模型参数。该片段的原始来源、裁剪记录和温哥华归属仍待核实。'),
      table(['项目','设置'],[
        ['载频与距离采样率','5.3 GHz；32.317 MHz'],
        ['调频率与脉宽','−0.72135 × 10¹² Hz/s；41.74 μs，名义带宽 30.109 MHz'],
        ['慢时间与几何','PRF 1256.98 Hz；等效速度 7062 m/s；直线运动斜距平面'],
        ['距离压缩','线性匹配滤波，Hamming 窗，4 倍距离过采样'],
        ['相干累加','全部 1728 脉冲采用 Hamming 权重；无 PGA 和附加相位估计'],
        ['计算精度','FP64 几何与相位；complex64 回波和节点复图']]),
      image('scene_comparison.png','图1 BP与FFBP场景对照。两图使用同一输出网格、BP场景峰值归一化和−60～−10 dB显示窗，黄色圆圈标出T1、T2。纵轴为斜距，横轴为共同参考点的局部方位坐标。'),
      page(),
      heading('二 两个强散射点的局部响应'),
      p('从12个候选中选取主峰清楚的T1和局部背景较低的T2。含多个相邻半高峰的候选未用于单点旁瓣评价。两点均位于全孔径距离压缩有效范围内。'),
      image('target_comparison.png','图2 两个强散射点的局部成像。左列BP，右列FFBP；每行共用该点BP峰值归一化和−40～0 dB显示窗。坐标相对BP峰值，未扣除背景或进行图像锐化。'),
      table(['目标','参考斜距 / m','参考sin方位角','局部网格'],[
        [t['id'],f"{t['center']['r_m']:.2f}",f"{t['center']['u']:.8f}",
         f"距离{t['ffbp_parameters']['range_spacing_m']:g} m，角向过采样{t['ffbp_parameters']['angular_oversample']}倍"]
        for t in targets]),
      page(),
      heading('三 距离向和方位向质量指标'),
      image('target_profiles.png','图3 原复图峰值处的距离向与方位向剖面。蓝实线为BP，橙虚线为FFBP；水平点线为半功率−3.0103 dB，竖线为各自测得的首极小主瓣边界。每条曲线的积分窗口为峰值附近±50 m。'),
      table(['目标','算法','方向','IRW / m','PSLR / dB','ISLR / dB'],[
        [t['id'],alg.upper(),'斜距' if d=='range' else '方位',
         f"{t[alg]['metrics'][d]['irw_m']:.2f}",f"{t[alg]['metrics'][d]['pslr_db']:.2f}",
         f"{t[alg]['metrics'][d]['islr_db']:.2f}"]
         for t in targets for alg in ('bp','ffbp') for d in ('range','azimuth')]),
      p('空间分辨率在本轮用−3 dB IRW表征。两个实测点未标定为理想点目标，因此表值是表观响应宽度；缺少可靠入射角，未换算地距。按当前带宽计算的名义斜距分辨单元约4.98 m，实际IRW还受加窗、有效照射孔径和散射体尺寸影响。'),
      p('PSLR为旁瓣最大功率相对主峰功率的分贝值，ISLR为旁瓣能量相对主瓣能量的分贝值，均按一维首极小主瓣和有限窗口计算。T1方位主瓣有宽肩部，其约−29 dB的ISLR依赖该主瓣边界，不能据此认定系统旁瓣达到同一指标。'),
      page(),
      heading('四 整场与局部的数值对照'),
      p('最终整场采用1 m距离步长、4倍角向过采样，FFBP叶节点最多包含64个脉冲。BP与FFBP使用同一输出网格；复数误差以直接BP为参考。'),
      table(['输出','复数误差 / %','BP核心 / s','FFBP核心 / s'],[
        ['整场',f'{err:.3f}',f"{fine['bp_timing']['kernel_ms']/1000:.3f}",f"{fine['ffbp_timing']['kernel_ms']/1000:.3f}"],
        *[[t['id']+'局部',f"{100*t['fidelity']['relative_complex_error']:.6f}",
           f"{t['bp']['timing']['kernel_ms']/1000:.3f}",f"{t['ffbp']['timing']['kernel_ms']/1000:.3f}"] for t in targets]]),
      p('局部图使用更密的测量网格，不能把局部保真结果外推为整场误差。计时来自RTX 4070 Laptop GPU单次CUDA事件记录，不含距离压缩、传输、绘图和导出。'),
      heading('五 实验结论'),
      p('在本轮模型与选点下，FFBP保持了BP的两个局部强散射响应。两点两方向的IRW差均小于0.001 m，这是同网格算法差异，不是系统分辨率的标定精度。两种算法仍可能共享几何、定时及信号模型误差。'),
      heading('数据记录与代码'),
      p('实测数值、主瓣边界和积分范围见target_metrics.json与quality_table.csv。输入文件SHA-256及处理参数保存在结果JSON，运行顺序见vancouver_bp/README.md。'),
      dict(kind='links',items=[
        ('处理参数脚本',ROOT/'RADARSAT.m'),
        ('实测结果记录',OUT/'target_metrics.json'),('指标明细表',OUT/'quality_table.csv'),
        ('独立回波仿真',ROOT/'vancouver_bp/validate.py'),('成像与指标代码',ROOT/'vancouver_bp/imaging.py'),
        ('质量指标实现',ROOT/'vancouver_bp/quality.py'),('复现说明',ROOT/'vancouver_bp/README.md')]),
      heading('参考资料'),
      p('Cumming与Wong教材配套数据提取说明用于核对模型参数和格式，不作为本地文件的下载来源认证。FFBP理论背景参考Ulander等的经典论文。'),
      dict(kind='refs',items=[
        ('UBC Extracting Raw SAR Data from the RADARSAT CD 2005','https://sar.ece.ubc.ca/SAR_Book/Extracting_SAR_Data_from_the_CD.pdf'),
        ('Ulander等 Synthetic Aperture Radar Processing Using Fast Factorized Back Projection 2003','https://doi.org/10.1109/TAES.2003.1238734')])]
    return blocks

def write_markdown(blocks):
    parts=[]
    for b in blocks:
        k=b['kind']
        if k=='title':parts+=['# '+b['text'],'']
        elif k=='subtitle':parts+=[b['text'],'']
        elif k=='heading':parts+=['## '+b['text'],'']
        elif k=='p':parts+=[b['text'],'']
        elif k=='image':
            target=Path(os.path.relpath(b['path'],OUT)).as_posix()
            parts += [f"![{b['caption']}](<{target}>)",'',b['caption'],'']
        elif k=='table':
            parts+=['| '+' | '.join(b['headers'])+' |','| '+' | '.join(['---']*len(b['headers']))+' |']
            parts+=['| '+' | '.join(row)+' |' for row in b['rows']];parts+=['']
        elif k=='links':parts+=['；'.join(link(label,path) for label,path in b['items'])+'。','']
        elif k=='refs':parts += [f'- [{label}]({url})' for label,url in b['items']];parts+=['']
    (OUT/'REPORT.md').write_text('\n'.join(parts),encoding='utf-8')

def write_docx(blocks):
    from docx import Document
    from docx.shared import Inches,Pt,RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    doc=Document();section=doc.sections[0]
    section.page_width=Inches(8.5);section.page_height=Inches(11)
    section.top_margin=Inches(.6);section.bottom_margin=Inches(.6)
    section.left_margin=Inches(.7);section.right_margin=Inches(.7)
    for name,size,bold in [('Normal',10.5,False),('Title',19,True),('Subtitle',10,False),('Heading 1',13,True),('Caption',9,False)]:
        s=doc.styles[name];s.font.name='Microsoft YaHei';s.font.size=Pt(size)
        s.font.bold=bold;s.font.color.rgb=RGBColor(0,0,0)
        s.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
        s.paragraph_format.space_after=Pt(5);s.paragraph_format.line_spacing=1.16
    doc.styles['Heading 1'].paragraph_format.space_before=Pt(7)
    doc.styles['Heading 1'].paragraph_format.keep_with_next=True
    doc.styles['Normal'].paragraph_format.widow_control=True
    def hyperlink(par,label,url):
        h=OxmlElement('w:hyperlink');h.set(qn('r:id'),par.part.relate_to(str(url),RT.HYPERLINK,is_external=True))
        r=OxmlElement('w:r');pr=OxmlElement('w:rPr');color=OxmlElement('w:color');color.set(qn('w:val'),'1F4E79');pr.append(color);r.append(pr)
        t=OxmlElement('w:t');t.text=label;r.append(t);h.append(r);par._p.append(h)
    for b in blocks:
        k=b['kind']
        if k in ('title','subtitle','heading','p'):
            sty={'title':'Title','subtitle':'Subtitle','heading':'Heading 1','p':'Normal'}[k]
            q=doc.add_paragraph(b['text'],style=sty)
            if k in ('title','subtitle'):q.paragraph_format.keep_with_next=True
        elif k=='page':doc.add_page_break()
        elif k=='image':
            q=doc.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER
            q.paragraph_format.space_after=Pt(3);q.paragraph_format.keep_with_next=True
            q.add_run().add_picture(str(b['path']),width=Inches(7.05))
            caption=doc.add_paragraph(b['caption'],style='Caption')
            caption.paragraph_format.keep_together=True
        elif k=='table':
            t=doc.add_table(rows=1,cols=len(b['headers']));t.style='Table Grid';t.autofit=False
            widths=([1.45,5.6] if len(b['headers'])==2 else
                    [1.0,1.1,1.25,3.7] if len(b['headers'])==4 and b['headers'][0]=='目标' else
                    [1.2,1.25,1.25,1.35,2.0] if len(b['headers'])==5 else
                    [1.5,1.65,1.95,1.95] if len(b['headers'])==4 else [0.55,.7,.65,1.3,1.9,1.95])
            for c,text in zip(t.rows[0].cells,b['headers']):c.text=text
            for row in b['rows']:
                for c,text in zip(t.add_row().cells,row):c.text=text
            for ri,row in enumerate(t.rows):
                trpr=row._tr.get_or_add_trPr();nosplit=OxmlElement('w:cantSplit');trpr.append(nosplit)
                if ri==0:
                    repeat=OxmlElement('w:tblHeader');trpr.append(repeat)
                for ci,c in enumerate(row.cells):
                    c.width=Inches(widths[ci])
                    if ri==0:
                        shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'F2F2F2');c._tc.get_or_add_tcPr().append(shade)
                    for q in c.paragraphs:
                        q.paragraph_format.space_before=Pt(3);q.paragraph_format.space_after=Pt(3);q.paragraph_format.line_spacing=1.08
                        for run in q.runs:run.font.size=Pt(9.4);run.bold=ri==0
            doc.add_paragraph().paragraph_format.space_after=Pt(0)
        elif k in ('links','refs'):
            q=doc.add_paragraph();q.paragraph_format.space_after=Pt(4)
            for j,(label,url) in enumerate(b['items']):
                if j:q.add_run('； ')
                hyperlink(q,label,Path(url).as_posix() if k=='links' else url)
    doc.core_properties.title='BP与FFBP成像对比实验'
    doc.core_properties.subject='CDdata1片段的成像与强散射点质量评价'
    doc.core_properties.author=''
    DOCX.parent.mkdir(parents=True,exist_ok=True);doc.save(DOCX)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--docx',action='store_true');args=parser.parse_args()
    record=json.loads((OUT/'target_metrics.json').read_text(encoding='utf-8'))
    blocks=content(record);write_markdown(blocks)
    if args.docx:write_docx(blocks)
    print(json.dumps(dict(markdown=str(OUT/'REPORT.md'),docx=str(DOCX) if args.docx else None),ensure_ascii=False))

if __name__=='__main__':main()
