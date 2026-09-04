from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

BG=RGBColor(0x0D,0x1B,0x2A); AC=RGBColor(0x2E,0xCC,0x71)
A2=RGBColor(0x3A,0x9B,0xD5); WH=RGBColor(0xFF,0xFF,0xFF)
GR=RGBColor(0xB0,0xB8,0xC8); DC=RGBColor(0x16,0x28,0x3A)
YL=RGBColor(0xF3,0xC6,0x23)
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
def cd(s,l,t,w,h): rc(s,l,t,w,h,DC)
def bdg(s,lb):
    rc(s,Inches(0.4),Inches(0.27),Inches(3.8),Inches(0.44),AC)
    tx(s,lb,Inches(0.45),Inches(0.25),Inches(3.7),Inches(0.48),sz=13,b=True,c=BG,al=PP_ALIGN.CENTER)
def hd(s,t1,t2=None):
    tx(s,t1,Inches(0.5),Inches(0.85),Inches(12.3),Inches(0.85),sz=28,b=True)
    if t2: tx(s,t2,Inches(0.5),Inches(1.65),Inches(12.3),Inches(0.45),sz=13,c=GR)
    rc(s,Inches(0.5),Inches(2.18),Inches(12.3),Inches(0.04),AC)

pr=mkprs()
s=mks(pr); bg(s)
bdg(s,"05  실험 결과")
hd(s,"아키텍처 비교 결과",
   "full 전처리 / Val: SO 3-seed 평균 / Test·갭: best seed 기준")

# 차트 (업데이트된 그림)
s.shapes.add_picture(
    "experiments/analysis/fig_architecture_comparison_acc.png",
    Inches(0.4), Inches(2.35), Inches(6.5), Inches(4.75))

# 결과표
cd(s, Inches(7.2), Inches(2.35), Inches(5.7), Inches(4.75))
tx(s,"결과표 (full, SO, best seed test)",
   Inches(7.4), Inches(2.45), Inches(5.3), Inches(0.42), sz=12, b=True, c=A2)

# 헤더
for c_idx, (header, cx, cw) in enumerate([
    ("아키텍처", 7.4, 2.0), ("Val Acc", 9.55, 0.9),
    ("Test Acc", 10.5, 0.9), ("Macro F1", 11.45, 0.9), ("갭", 12.38, 0.8)
]):
    tx(s, header, Inches(cx), Inches(2.92), Inches(cw), Inches(0.38),
       sz=10.5, b=True, c=A2)

rows = [
    ("EfficientNet-B3", "84.0%", "72.9%", "55.4%", "-12.5%p", False),
    ("ConvNeXt-Tiny",   "81.3%", "72.4%", "55.8%", "-11.4%p", False),
    ("Swin-Tiny  ★",   "77.6%", "72.9%", "57.2%", " -6.9%p", True),
    ("DINOv2-ViT",      "72.1%", "66.8%", "54.6%", " -6.9%p", False),
    ("ViT-S/16",        "64.0%", "60.3%", "39.1%", " -7.4%p", False),
]
for r_idx, (arch, val, test, f1, gap, is_best) in enumerate(rows):
    row_y = 3.38 + r_idx * 0.62
    clr = AC if is_best else WH
    if is_best:
        rc(s, Inches(7.38), Inches(row_y-0.04), Inches(5.32), Inches(0.52),
           RGBColor(0x0A, 0x22, 0x12))
    for text, cx, cw in [
        (arch, 7.4, 1.98), (val, 9.55, 0.88),
        (test, 10.5, 0.88), (f1, 11.45, 0.88), (gap, 12.38, 0.78)
    ]:
        tx(s, text, Inches(cx), Inches(row_y), Inches(cw), Inches(0.48),
           sz=11, b=is_best, c=clr)

# 하단 설명
for line, clr, yy in [
    ("★ Swin 선정: val에서 낮지만 Test acc 공동1위 + 갭 최소(6.9%p) + Macro F1 최고(57.2%)", AC, 6.72),
    ("상위 3모델 95% CI 중첩 [66.8, 78.9] — 통계적 유의차 없음 / EfficientNet·ConvNeXt 갭 11~12%p → 과적합 경향", GR, 7.08),
]:
    tx(s, line, Inches(0.5), Inches(yy), Inches(12.3), Inches(0.35), sz=11, c=clr)

pr.save("beetledex_midterm_2026_아키텍처수정.pptx")
print(f"저장: beetledex_midterm_2026_아키텍처수정.pptx ({len(pr.slides)}장)")
