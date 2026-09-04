from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

BG=RGBColor(0x0D,0x1B,0x2A); AC=RGBColor(0x2E,0xCC,0x71)
A2=RGBColor(0x3A,0x9B,0xD5); WH=RGBColor(0xFF,0xFF,0xFF)
GR=RGBColor(0xB0,0xB8,0xC8); DC=RGBColor(0x16,0x28,0x3A)
YL=RGBColor(0xF3,0xC6,0x23); IB=RGBColor(0x1E,0x35,0x4A)
W=Inches(13.33); H=Inches(7.5)

def mkprs():
    p=Presentation(); p.slide_width=W; p.slide_height=H; return p
def mks(p): return p.slides.add_slide(p.slide_layouts[6])
def bg(s):
    f=s.background.fill; f.solid(); f.fore_color.rgb=BG
def rc(s,l,t,w,h,c):
    sh=s.shapes.add_shape(1,l,t,w,h)
    sh.fill.solid(); sh.fill.fore_color.rgb=c; sh.line.fill.background()
def tx(s,text,l,t,w,h,sz=14,b=False,c=WH,al=PP_ALIGN.LEFT):
    bx=s.shapes.add_textbox(l,t,w,h); tf=bx.text_frame; tf.word_wrap=True
    p=tf.paragraphs[0]; p.alignment=al; r=p.add_run()
    r.text=text; r.font.size=Pt(sz); r.font.bold=b; r.font.color.rgb=c
def addimg(s,path,l,t,w,h):
    try: s.shapes.add_picture(path,l,t,w,h)
    except:
        sh=s.shapes.add_shape(1,l,t,w,h)
        sh.fill.solid(); sh.fill.fore_color.rgb=IB
        sh.line.color.rgb=A2; sh.line.width=Pt(1)
        tx(s,'[ '+path.split('/')[-1]+' ]',l,t+(h-Inches(0.4))/2,w,Inches(0.4),sz=10,c=A2,al=PP_ALIGN.CENTER)
def cd(s,l,t,w,h): rc(s,l,t,w,h,DC)
def bdg(s,lb):
    rc(s,Inches(0.4),Inches(0.27),Inches(3.8),Inches(0.44),AC)
    tx(s,lb,Inches(0.45),Inches(0.25),Inches(3.7),Inches(0.48),sz=13,b=True,c=BG,al=PP_ALIGN.CENTER)
def hd(s,t1,t2=None):
    tx(s,t1,Inches(0.5),Inches(0.85),Inches(12.3),Inches(0.85),sz=28,b=True)
    if t2: tx(s,t2,Inches(0.5),Inches(1.65),Inches(12.3),Inches(0.42),sz=13,c=GR)
    rc(s,Inches(0.5),Inches(2.18),Inches(12.3),Inches(0.04),AC)

pr=mkprs()
s=mks(pr); bg(s)
bdg(s,"05  실험 결과")
hd(s,"Test Set 최종 결과","Hold-out 199장 — 학습·검증 중 절대 미사용")

# 상단 결과표
cd(s,Inches(0.4),Inches(2.35),Inches(12.5),Inches(2.95))
tx(s,"최종 평가 결과 (Bootstrap 95% CI — best seed 기준)",
   Inches(0.6),Inches(2.45),Inches(8.0),Inches(0.42),sz=13,b=True,c=A2)

headers=[("모델",0.6,3.3),("Test Acc",4.05,1.2),("Macro F1",5.35,1.1),("95% CI",6.55,2.55),("Val-Test 갭",9.25,1.8)]
for h_txt,cx,cw in headers:
    tx(s,h_txt,Inches(cx),Inches(2.92),Inches(cw),Inches(0.38),sz=11,b=True,c=A2)

rows=[
    ("Swin-T SO s456  ★","72.9%","57.2%","[66.8, 78.9]","-6.9%p",True),
    ("EfficientNet SO","72.9%","55.4%","[66.8, 78.9]","-12.5%p",False),
    ("ConvNeXt SO","72.4%","55.8%","[66.3, 78.9]","-11.4%p",False),
    ("EfficientNet SX","70.4%","54.3%","[64.3, 76.9]","-15.5%p",False),
    ("DINOv2 SO","66.8%","54.6%","[60.3, 73.4]","-6.9%p",False),
    ("ViT-S/16 SO","60.3%","39.1%","[53.8, 67.8]","-7.4%p",False),
]
for i,(model,acc,f1,ci,gap,is_best) in enumerate(rows):
    ry = 3.38 + i*0.38
    if is_best: rc(s,Inches(0.5),Inches(ry-0.03),Inches(12.3),Inches(0.36),RGBColor(0x0A,0x22,0x12))
    clr=AC if is_best else WH
    for txt,cx,cw in [(model,0.6,3.3),(acc,4.05,1.2),(f1,5.35,1.1),(ci,6.55,2.55),(gap,9.25,1.8)]:
        tx(s,txt,Inches(cx),Inches(ry),Inches(cw),Inches(0.36),sz=11,b=is_best,c=clr)

# 하단 그림
addimg(s,'experiments/analysis/fig_fewshot_scatter.png',
       Inches(0.4),Inches(5.42),Inches(6.1),Inches(1.85))
addimg(s,'experiments/calibration/reliability_swin_full_SO_s456.png',
       Inches(6.65),Inches(5.42),Inches(6.25),Inches(1.85))

# 하단 해석
rc(s,Inches(0.4),Inches(5.28),Inches(12.5),Inches(0.1),RGBColor(0x2A,0x3A,0x4A))

for line,clr,yy in [
    ("★ Best: Swin-T SO s456 — test acc 공동 1위 / macro_f1 최고(57.2%) / val-test 갭 최소(-6.9%p)",AC,7.15),
    ("상위 3모델 95% CI [66.8, 78.9] 중첩 — 통계적 유의차 없음 (test 199장 한계) | ViT macro F1 39.1% — 희귀종에서 특히 불안정 (N4)",GR,7.42),
]:
    tx(s,line,Inches(0.5),Inches(yy),Inches(12.3),Inches(0.28),sz=10.5,c=clr)

pr.save("beetledex_midterm_2026_TestSet수정.pptx")
print("저장 완료")
