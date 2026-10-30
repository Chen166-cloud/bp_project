"""Render the figure-led report directly as a visually checked PDF."""
from pathlib import Path
from xml.sax.saxutils import escape
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT,TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak,KeepTogether
from PIL import Image as PILImage
from meeting_report import ROOT,OUT,content,write_markdown

DEST=ROOT/'output/pdf/BP与FFBP成像实验报告.pdf'
pdfmetrics.registerFont(TTFont('YaHei',r'C:\Windows\Fonts\msyh.ttc'))
pdfmetrics.registerFont(TTFont('YaHeiBold',r'C:\Windows\Fonts\msyhbd.ttc'))
pdfmetrics.registerFontFamily('YaHei',normal='YaHei',bold='YaHeiBold',italic='YaHei',boldItalic='YaHeiBold')
styles={
    'title':ParagraphStyle('Title',fontName='YaHeiBold',fontSize=18,leading=24,spaceAfter=5),
    'subtitle':ParagraphStyle('Subtitle',fontName='YaHei',fontSize=9.2,leading=14,spaceAfter=8),
    'heading':ParagraphStyle('Heading',fontName='YaHeiBold',fontSize=12,leading=17,spaceBefore=7,spaceAfter=6,keepWithNext=True),
    'body':ParagraphStyle('Body',fontName='YaHei',fontSize=10.3,leading=15.5,spaceAfter=6,wordWrap='CJK'),
    'caption':ParagraphStyle('Caption',fontName='YaHei',fontSize=8.7,leading=12.5,spaceAfter=7,wordWrap='CJK'),
    'cell':ParagraphStyle('Cell',fontName='YaHei',fontSize=9,leading=13,wordWrap='CJK'),
    'header':ParagraphStyle('Header',fontName='YaHeiBold',fontSize=9,leading=13,wordWrap='CJK')}

def para(text,style='body'):return Paragraph(escape(text),styles[style])

def main():
    record=json.loads((OUT/'target_metrics.json').read_text(encoding='utf-8'))
    blocks=content(record);write_markdown(blocks)
    story=[];page_number=1;width=511
    for b in blocks:
        k=b['kind']
        if k in ('title','subtitle','heading','p'):
            story.append(para(b['text'],{'p':'body'}.get(k,k)))
        elif k=='page':story.append(PageBreak());page_number+=1
        elif k=='image':
            # Give the scene room below the common conditions; other panels
            # use the full text width to retain readable chart type.
            display_width=470 if page_number==1 else width
            with PILImage.open(b['path']) as im:iw,ih=im.size
            img=Image(str(b['path']),width=display_width,height=display_width*ih/iw)
            img.hAlign='CENTER'
            story.append(KeepTogether([img,Spacer(1,4),para(b['caption'],'caption')]))
        elif k=='table':
            n=len(b['headers'])
            widths=([110,width-110] if n==2 else [56,78,103,width-237] if n==4 and b['headers'][0]=='目标' else
                [104,84,90,106,width-384] if n==5 else [90,130,145,width-365] if n==4 else [37,50,44,100,140,width-371])
            data=[[para(t,'header') for t in b['headers']]]+[[para(t,'cell') for t in row] for row in b['rows']]
            tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            tab.setStyle(TableStyle([
                ('GRID',(0,0),(-1,-1),.45,colors.HexColor('#777777')),
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#F1F1F1')),
                ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
                ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            story += [tab,Spacer(1,7)]
        elif k in ('links','refs'):
            links=[]
            for label,url in b['items']:
                target=Path(url).resolve().as_posix() if k=='links' else url
                links.append('<link href="'+escape(target,{'"':'&quot;'})+'" color="#1F4E79">'+escape(label)+'</link>')
            story.append(Paragraph('； '.join(links),styles['caption']))
    DEST.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(DEST),pagesize=letter,leftMargin=50.5,rightMargin=50.5,
        topMargin=29,bottomMargin=30,title='BP与FFBP成像对比实验',author='')
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('YaHei',8);canvas.setFillColor(colors.HexColor('#555555'))
        canvas.drawRightString(561,17,str(doc.page));canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(DEST)

if __name__=='__main__':main()
