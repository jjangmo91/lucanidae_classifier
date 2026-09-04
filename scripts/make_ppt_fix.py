from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

BG=RGBColor(0x0D,0x1B,0x2A); AC=RGBColor(0x2E,0xCC,0x71)
A2=RGBColor(0x3A,0x9B,0xD5); WH=RGBColor(0xFF,0xFF,0xFF)
GR=RGBColor(0xB0,0xB8,0xC8); DC=RGBColor(0x16,0x28,0x3A)
IB=RGBColor(0x1E,0x35,0x4A); YL=RGBColor(0xF3,0xC6,0x23)
W=Inches(13.33); H=Inches(7.5)

def mkprs():
    p=Presentation(); p.slide_width=W; p.slide_height=H; return p
def mks(p): return p.slides.add_slide(p.slide_layouts[6])
def bg(s):
    f=s.background.fill; f.solid(); f.fore_color.rgb=BG
def rc(s,l,t,w,h,c,ln=False):
    sh=s.shapes.add_shape(1,l,t,w,h)
    sh.fill.solid(); sh.fill.fore_color.rgb=c
    if ln: sh.line.color.rgb=A2; sh.line.width=Pt(1)
    else: sh.line.fill.background()
def tx(s,text,l,t,w,h,sz=14,b=False,c=WH,al=PP_ALIGN.LEFT):
    bx=s.shapes.add_textbox(l,t,w,h); tf=bx.text_frame; tf.word_wrap=True
    p=tf.paragraphs[0]; p.alignment=al; r=p.add_run()
    r.text=text; r.font.size=Pt(sz); r.font.bold=b; r.font.color.rgb=c
def im(s,l,t,w,h,lb):
    rc(s,l,t,w,h,IB,ln=True)
    tx(s,f"[ {lb} ]",l,t+(h-Inches(0.4))/2,w,Inches(0.4),sz=11,c=A2,al=PP_ALIGN.CENTER)
def cd(s,l,t,w,h): rc(s,l,t,w,h,DC)
def bdg(s,lb):
    rc(s,Inches(0.4),Inches(0.27),Inches(3.8),Inches(0.44),AC)
    tx(s,lb,Inches(0.45),Inches(0.25),Inches(3.7),Inches(0.48),sz=13,b=True,c=BG,al=PP_ALIGN.CENTER)
def hd(s,t1,t2=None):
    tx(s,t1,Inches(0.5),Inches(0.85),Inches(12.3),Inches(0.85),sz=28,b=True)
    if t2: tx(s,t2,Inches(0.5),Inches(1.65),Inches(12.3),Inches(0.45),sz=13,c=GR)
    rc(s,Inches(0.5),Inches(2.18),Inches(12.3),Inches(0.04),AC)

pr=mkprs()

# ════════════════════════════════════════════════
# 슬라이드 1: 실험 설계 (137 → 149, SX Ablation 추가)
# ════════════════════════════════════════════════
s=mks(pr); bg(s)
bdg(s,"05  실험 결과")
hd(s,"실험 설계 — 총 149개","완전 요인 설계 + Ablation")

exps=[
    ("메인 Sweep","120개",
     "5 arch × 4 prep × 2 mode(SO, MT) × 3 seed",
     "python scripts/sweep.py", AC),
    ("MT Ablation","6개",
     "Best (arch, prep) × 2 mode(SX, FO) × 3 seed\n→ MLflow에서 베스트 조합 자동 선택",
     "python scripts/sweep_mt_ablation.py", A2),
    ("SX Ablation","12개",
     "나머지 4 arch (ConvNeXt/Swin/ViT/DINOv2) × full × SX × 3 seed\n→ SX 효과가 아키텍처 전반에서 재현되는가 검증",
     "python scripts/sweep_mt_ablation.py", YL),
    ("λ Ablation","11개",
     "λ_sex [0.05~1.0] × λ_form=0.2 = 6\nλ_form [0.05~1.0] × λ_sex=0.3 = 5  (seed=42 고정)",
     "python scripts/sweep_lambda.py", RGBColor(0xE7,0x4C,0x3C)),
]
for i,(nm,cnt,desc,cmd,col) in enumerate(exps):
    t=Inches(2.38+i*1.22)
    cd(s,Inches(0.4),t,Inches(12.5),Inches(1.06))
    rc(s,Inches(0.4),t,Inches(0.08),Inches(1.06),col)
    tx(s,nm,Inches(0.65),t+Inches(0.08),Inches(2.4),Inches(0.42),sz=14,b=True,c=col)
    tx(s,cnt,Inches(3.2),t+Inches(0.08),Inches(1.0),Inches(0.42),sz=20,b=True)
    tx(s,desc,Inches(0.65),t+Inches(0.54),Inches(8.2),Inches(0.62),sz=11.5,c=GR)
    tx(s,cmd,Inches(9.2),t+Inches(0.38),Inches(3.6),Inches(0.35),sz=10.5,c=A2)

tx(s,"run_name: {arch}_{prep}_{mode}_s{seed}   |   MLflow로 완료된 run 자동 스킵",
   Inches(0.5),Inches(7.18),Inches(12.3),Inches(0.28),sz=10.5,c=GR)

# ════════════════════════════════════════════════
# 슬라이드 2: 전처리 Ablation 결과 (해석 수정)
# ════════════════════════════════════════════════
s=mks(pr); bg(s)
bdg(s,"05  실험 결과")
hd(s,"전처리 Ablation 결과","3-seed 평균 val_acc (%) — 전 아키텍처에서 full이 우위")

im(s,Inches(0.4),Inches(2.35),Inches(6.3),Inches(4.75),
   "fig_preprocessing_ablation.png (bar chart)")
cd(s,Inches(6.95),Inches(2.35),Inches(5.95),Inches(4.75))
tx(s,"전처리 Ablation 결과표",
   Inches(7.15),Inches(2.45),Inches(5.55),Inches(0.42),sz=13,b=True,c=A2)
for r,row in enumerate([
    ["모드","ConvNeXt","EfficientNet","Swin","ViT"],
    ["full  ★","81.3%","84.0%","77.6%","64.0%"],
    ["bbox","65.7%","72.0%","62.5%","56.0%"],
    ["seg_soft","64.5%","68.3%","60.6%","55.0%"],
    ["seg_hard","65.5%","65.5%","60.6%","54.5%"],
]):
    for c,cell in enumerate(row):
        tx(s,cell,Inches(7.15+c*1.1),Inches(2.95+r*0.52),Inches(1.05),Inches(0.48),
           sz=11,b=(r<=1),c=A2 if r==0 else (AC if r==1 else WH))

tx(s,"full이 전 아키텍처에서 15~30%p 우위",
   Inches(7.15),Inches(5.22),Inches(5.55),Inches(0.38),sz=12,b=True,c=WH)
tx(s,"원인은 단정할 수 없음 — 다양한 해석이 가능:",
   Inches(7.15),Inches(5.62),Inches(5.55),Inches(0.35),sz=11.5,c=GR)
tx(s,"배경이 종 판별에 실제로 유효 / 세그멘테이션 과정에서 특징 손실\n배경 편향 가능성 (Prismognathus 1종에서 GradCAM 확인) / 데이터 부족 노이즈",
   Inches(7.15),Inches(6.0),Inches(5.55),Inches(0.72),sz=10.5,c=GR)
tx(s,"→ 데이터 확장 후 재검증 필요 (한계 고찰 참조)",
   Inches(7.15),Inches(6.78),Inches(5.55),Inches(0.35),sz=11.5,c=YL)

# ════════════════════════════════════════════════
# 슬라이드 3: 아키텍처 비교 결과 (ViT 64.0% 확인)
# ════════════════════════════════════════════════
s=mks(pr); bg(s)
bdg(s,"05  실험 결과")
hd(s,"아키텍처 비교 결과","full 전처리, 3-seed 평균 val_acc (%)")

im(s,Inches(0.4),Inches(2.35),Inches(6.3),Inches(4.75),"fig_architecture_comparison.png")
cd(s,Inches(6.95),Inches(2.35),Inches(5.95),Inches(4.75))
tx(s,"결과표 (full, SO, 3-seed 평균)",
   Inches(7.15),Inches(2.45),Inches(5.55),Inches(0.42),sz=13,b=True,c=A2)
for r,row in enumerate([
    ["아키텍처","Val Acc","Macro F1","Val-Test 갭"],
    ["EfficientNet-B3","84.0%","66.6%","-12.5%p"],
    ["ConvNeXt-Tiny","81.3%","63.7%","-11.4%p"],
    ["Swin-Tiny  ★","77.6%","61.9%","-6.9%p"],
    ["DINOv2-ViT","72.1%","57.2%","-6.9%p"],
    ["ViT-Small","64.0%","49.1%","-7.4%p"],
]):
    for c,cell in enumerate(row):
        tx(s,cell,Inches(7.15+c*1.45),Inches(2.95+r*0.6),Inches(1.4),Inches(0.55),
           sz=12,b=(r<=1 or r==3),c=A2 if r==0 else (AC if r==3 else WH))

for line,col,ty in [
    ("Val 기준 EfficientNet 1위",WH,6.08),
    ("Test 기준 Swin/EfficientNet 공동 1위",WH,6.48),
    ("Swin이 val-test 갭 가장 작음 (일반화 가능성)",WH,6.78),
    ("상위 3모델 95% CI 중첩 — 통계적 유의차 없음",GR,7.1),
]:
    tx(s,line,Inches(7.15),Inches(ty),Inches(5.55),Inches(0.35),sz=11.5,c=col)

pr.save("beetledex_midterm_2026_수정본.pptx")
print(f"저장 완료: beetledex_midterm_2026_수정본.pptx ({len(pr.slides)}장)")
