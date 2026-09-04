"""BeetleDex 중간발표 PPT (30장) — python scripts/make_ppt.py"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

BG=RGBColor(0x0D,0x1B,0x2A); AC=RGBColor(0x2E,0xCC,0x71)
A2=RGBColor(0x3A,0x9B,0xD5); WH=RGBColor(0xFF,0xFF,0xFF)
GR=RGBColor(0xB0,0xB8,0xC8); DC=RGBColor(0x16,0x28,0x3A)
IB=RGBColor(0x1E,0x35,0x4A); YL=RGBColor(0xF3,0xC6,0x23)
RD=RGBColor(0xE7,0x4C,0x3C)
W=Inches(13.33); H=Inches(7.5); TOT=31

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
    return sh
def tx(s,text,l,t,w,h,sz=14,b=False,c=WH,al=PP_ALIGN.LEFT):
    bx=s.shapes.add_textbox(l,t,w,h); tf=bx.text_frame; tf.word_wrap=True
    p=tf.paragraphs[0]; p.alignment=al; r=p.add_run()
    r.text=text; r.font.size=Pt(sz); r.font.bold=b; r.font.color.rgb=c
    return bx
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
def bl(s,items,l,t,w,h,hdr=None,hc=A2,sz=13):
    if hdr:
        tx(s,hdr,l,t,w,Inches(0.42),sz=14,b=True,c=hc)
        t+=Inches(0.46); h-=Inches(0.46)
    bx=s.shapes.add_textbox(l,t,w,h); tf=bx.text_frame; tf.word_wrap=True
    for i,item in enumerate(items):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.space_before=Pt(4); r=p.add_run()
        r.text=item; r.font.size=Pt(sz); r.font.color.rgb=WH
def sn(s,n): tx(s,f"{n}/{TOT}",Inches(12.5),Inches(7.15),Inches(0.7),Inches(0.3),sz=10,c=GR,al=PP_ALIGN.RIGHT)

pr=mkprs()

# 1. 표지
s=mks(pr); bg(s)
rc(s,Inches(0),Inches(0),Inches(0.28),H,AC)
rc(s,Inches(0.28),Inches(4.85),Inches(13.05),Inches(0.06),A2)
tx(s,"Taxonomy & Classification",Inches(1),Inches(1.5),Inches(11.3),Inches(0.7),sz=20,c=A2,al=PP_ALIGN.CENTER)
tx(s,"딥러닝으로 사슴벌레 분류기 만들기",Inches(1),Inches(2.2),Inches(11.3),Inches(1.0),sz=38,b=True,al=PP_ALIGN.CENTER)
rc(s,Inches(3),Inches(3.3),Inches(7.3),Inches(0.05),AC)
tx(s,"BeetleDex — 한국 사슴벌레 AI 도감",Inches(1),Inches(3.45),Inches(11.3),Inches(0.6),sz=18,c=AC,al=PP_ALIGN.CENTER)
tx(s,"beetledex.com",Inches(1),Inches(4.1),Inches(11.3),Inches(0.45),sz=14,c=GR,al=PP_ALIGN.CENTER)
tx(s,"2026. 06. 05",Inches(1),Inches(6.75),Inches(11.3),Inches(0.4),sz=13,c=GR,al=PP_ALIGN.CENTER)

# 2. 목차
s=mks(pr); bg(s); sn(s,2); bdg(s,"목차"); hd(s,"발표 순서")
toc=[("01","배경 & 동기","우주·생명의 기원 → 분류학 위기 → 프로젝트 비전","6분"),
     ("02","연구 목표","5가지 노벨티 · 논문 기여","3분"),
     ("03","데이터","수집 · 어노테이션 · 레이블 · Split","4분"),
     ("04","ML 파이프라인","전처리 · 아키텍처 · Multi-task · OOD","6분"),
     ("05","실험 결과","137개 실험 · Ablation · Best 모델","7분"),
     ("06","웹서비스 & 배포","beetledex.com · Data Flywheel","3분"),
     ("07","한계 & 전망","딥러닝의 근본 한계 · 향후 방향","1분")]
for i,(num,t1,sub,est) in enumerate(toc):
    row,col2=divmod(i,2)
    lx,ww=(Inches(3.1),Inches(7.1)) if i==6 else (Inches(0.45+col2*6.55),Inches(6.2))
    t=Inches(2.42+row*1.46); cd(s,lx,t,ww,Inches(1.28))
    tx(s,num,lx+Inches(0.18),t+Inches(0.08),Inches(0.65),Inches(0.5),sz=22,b=True,c=AC)
    tx(s,t1,lx+Inches(0.9),t+Inches(0.08),ww-Inches(1.3),Inches(0.48),sz=16,b=True)
    tx(s,sub,lx+Inches(0.9),t+Inches(0.62),ww-Inches(1.3),Inches(0.38),sz=11,c=GR)
    tx(s,est,lx+ww-Inches(1.05),t+Inches(0.08),Inches(0.95),Inches(0.38),sz=11,c=A2,al=PP_ALIGN.RIGHT)

# 3. 두 개의 위대한 질문
s=mks(pr); bg(s); sn(s,3); bdg(s,"01  배경 & 동기")
tx(s,"인류가 품어온 가장 위대한 두 가지 질문",Inches(0.5),Inches(0.85),Inches(12.3),Inches(0.65),sz=22,b=True,al=PP_ALIGN.CENTER)
rc(s,Inches(0.5),Inches(1.55),Inches(12.3),Inches(0.04),AC)
im(s,Inches(0.4),Inches(1.72),Inches(6.1),Inches(3.5),"우주/빅뱅 이미지 (NASA 허블·제임스웹 사진 권장)")
rc(s,Inches(0.4),Inches(5.28),Inches(6.1),Inches(1.55),RGBColor(0x08,0x10,0x1C))
tx(s,"우주는 어떻게\n시작되었는가?",Inches(0.55),Inches(5.35),Inches(5.8),Inches(0.9),sz=22,b=True)
tx(s,"Big Bang  ·  138억 년 전  ·  우주론",Inches(0.55),Inches(6.32),Inches(5.8),Inches(0.38),sz=12,c=A2)
rc(s,Inches(6.6),Inches(1.72),Inches(0.05),Inches(5.11),AC)
rc(s,Inches(6.28),Inches(3.88),Inches(0.7),Inches(0.58),BG)
tx(s,"&",Inches(6.28),Inches(3.86),Inches(0.7),Inches(0.62),sz=20,b=True,c=AC,al=PP_ALIGN.CENTER)
im(s,Inches(6.75),Inches(1.72),Inches(6.1),Inches(3.5),"생명 기원 이미지 (원시지구·세포·Tree of Life 개념도)")
rc(s,Inches(6.75),Inches(5.28),Inches(6.1),Inches(1.55),RGBColor(0x08,0x1C,0x10))
tx(s,"생명은 어떻게\n시작되었는가?",Inches(6.9),Inches(5.35),Inches(5.8),Inches(0.9),sz=22,b=True)
tx(s,"Origin of Life  ·  38억 년 전  ·  진화생물학",Inches(6.9),Inches(6.32),Inches(5.8),Inches(0.38),sz=12,c=AC)
rc(s,Inches(0.4),Inches(6.88),Inches(12.5),Inches(0.55),RGBColor(0x12,0x22,0x2E))
tx(s,"생명의 시작을 이해하기 위해, 인류는 먼저 지구의 생명을 기록하고 분류하기 시작했다  →  분류학(Taxonomy)",
   Inches(0.6),Inches(6.93),Inches(12.1),Inches(0.42),sz=13,b=True,c=A2,al=PP_ALIGN.CENTER)

# 4. 사슴벌레란?
s=mks(pr); bg(s); sn(s,4); bdg(s,"01  배경 & 동기")
hd(s,"사슴벌레란?","한국 사슴벌레과(Lucanidae) — 16종")
im(s,Inches(0.4),Inches(2.35),Inches(5.5),Inches(4.8),"한국 사슴벌레 16종 대표 이미지")
cd(s,Inches(6.2),Inches(2.35),Inches(6.7),Inches(4.8))
bl(s,["사슴벌레과(Lucanidae) — 전 세계 1,000종+ 분포","한국 서식 확인 종: 16종",
      "딱정벌레목 중 수컷 큰턱(mandible) 발달이 특징","",
      "대표종:","  왕사슴벌레 (Dorcus hopei binodulosus)",
      "  넓적사슴벌레 (Dorcus titanus castanicolor)",
      "  사슴벌레 (Lucanus maculifemoratus)",
      "  톱사슴벌레 (Prosopocoilus inclinatus)","",
      "생태적 가치: 목재 분해 · 산림 생태계 지표종",
      "관심도: 표본 수집 · 사육 문화 활성화"],
   Inches(6.45),Inches(2.45),Inches(6.2),Inches(4.55))

# 5. 곤충 분류의 위기
s=mks(pr); bg(s); sn(s,5); bdg(s,"01  배경 & 동기")
hd(s,"곤충 분류의 위기 — 왜 AI가 필요한가?","지구에서 가장 다양한 분류군, 그러나 전문가는 턱없이 부족")
cd(s,Inches(0.4),Inches(2.35),Inches(5.85),Inches(4.75))
cd(s,Inches(6.55),Inches(2.35),Inches(6.35),Inches(4.75))
bl(s,["곤충 다양성:","  지구 동물 종의 약 60% — 100만 종+ 기재",
      "  연간 신종 기재 ~20,000종","  미기재 추정: 수백만 종","",
      "Taxonomic Impediment (분류학적 장벽):",
      "  전 세계 분류학자 수 급감 추세",
      "  한국 국립공원 생태계조사",
      "  → 분류군별 담당 전문가 1~3명",
      "  → 한 명이 수천 종을 담당"],
   Inches(0.65),Inches(2.45),Inches(5.35),Inches(4.55),hdr="문제 현황")
im(s,Inches(6.75),Inches(2.45),Inches(5.9),Inches(2.55),"국립공원 생태계조사단 분류군별 전문가 인원 현황 (바 차트/표)")
tx(s,"프로젝트 비전",Inches(6.75),Inches(5.12),Inches(5.9),Inches(0.42),sz=14,b=True,c=AC)
bl(s,["사슴벌레는 시작점 — 재현 가능한 프레임워크",
      "소규모 데이터(수백~수천 장)로 각 분류군별 자동 동정 모델 구축",
      "나비 · 잠자리 · 딱정벌레 · 식물로 확장",
      "비전문가도 현장에서 즉시 동정 가능"],
   Inches(6.75),Inches(5.6),Inches(5.9),Inches(1.38),sz=13)

# 6. 분류학 vs 딥러닝
s=mks(pr); bg(s); sn(s,6); bdg(s,"01  배경 & 동기")
hd(s,"분류학 vs 딥러닝 — 본질적 차이","분류학자를 대체하는 것이 아니라, 그들의 작업을 지원하는 것")
cd(s,Inches(0.4),Inches(2.35),Inches(5.85),Inches(4.75))
cd(s,Inches(6.55),Inches(2.35),Inches(6.35),Inches(4.75))
im(s,Inches(0.55),Inches(2.45),Inches(2.6),Inches(2.3),"Tree of Life / 계통수 그림")
tx(s,"분류학자(Taxonomist)의 역할",Inches(3.3),Inches(2.45),Inches(2.75),Inches(0.42),sz=13,b=True,c=AC)
bl(s,["종 기재 — 신종 발견·라틴어 명명","계통 수립 — 진화적 관계 파악",
      "문헌 개정 — 200년+ 기재 문헌 재검토","동의어 처리 — 잘못 기재된 종 통합",
      "형질 정의 — 진단적 특징 판단"],
   Inches(3.3),Inches(2.92),Inches(2.75),Inches(1.85),sz=11)
tx(s,"단순한 분류가 아닙니다",Inches(0.65),Inches(4.9),Inches(5.15),Inches(0.38),sz=13,b=True,c=YL)
bl(s,["자연의 시스템 전체를 이해하고 기술하는 작업",
      "수십 년 경험으로 쌓인 형태학적 직관",
      "처음 보는 개체도 어떤 그룹인지 추론 가능"],
   Inches(0.65),Inches(5.35),Inches(5.15),Inches(1.55),sz=12)
tx(s,"딥러닝의 접근",Inches(6.75),Inches(2.45),Inches(5.9),Inches(0.42),sz=14,b=True,c=A2)
for i,(k,v) in enumerate([
    ("패턴 인식","'무엇이 비슷한가'를 통계적으로 학습\n훈련 데이터의 분포만 이해"),
    ("설명 불가","왜 그 종인지 설명 못함\n분류학: 진단 형질로 명확히 설명"),
    ("데이터 의존","성능 ∝ 데이터 양\n처음 보는 종 → 동정 불가 (closed-set)")]):
    t=Inches(2.95+i*1.05); rc(s,Inches(6.75),t,Inches(0.06),Inches(0.88),A2)
    tx(s,k,Inches(6.95),t+Inches(0.05),Inches(1.5),Inches(0.38),sz=13,b=True)
    tx(s,v,Inches(6.95),t+Inches(0.45),Inches(5.5),Inches(0.38),sz=12,c=GR)
rc(s,Inches(6.75),Inches(6.1),Inches(5.9),Inches(0.9),RGBColor(0x1A,0x3A,0x1A))
tx(s,"이 프로젝트의 역할",Inches(6.95),Inches(6.15),Inches(5.5),Inches(0.35),sz=13,b=True,c=AC)
tx(s,"동정(Identification) 자동화 — 분류학자가 반복 작업에서 벗어나 더 중요한 연구에 집중하도록 지원",
   Inches(6.95),Inches(6.52),Inches(5.5),Inches(0.42),sz=12)

# 7. 분류가 어려운 이유
s=mks(pr); bg(s); sn(s,7); bdg(s,"01  배경 & 동기")
hd(s,"왜 분류가 어려운가?","Fine-Grained Visual Classification의 핵심 난제")
cd(s,Inches(0.4),Inches(2.35),Inches(6.2),Inches(2.3))
cd(s,Inches(6.85),Inches(2.35),Inches(6.05),Inches(2.3))
tx(s,"성적 이형성 (Sexual Dimorphism)",Inches(0.6),Inches(2.45),Inches(5.8),Inches(0.42),sz=14,b=True,c=AC)
bl(s,["같은 종이어도 수컷·암컷 외형이 완전히 다름","큰턱 유무 · 체형 차이가 극명","기존 분류기: 암컷 오분류율 높음"],Inches(0.6),Inches(2.92),Inches(5.8),Inches(1.55))
tx(s,"수컷 크기 다형성 (Male Form)",Inches(7.05),Inches(2.45),Inches(5.6),Inches(0.42),sz=14,b=True,c=AC)
bl(s,["크기에 따라 큰턱 형태가 달라짐","major / minor / intermediate 3가지","같은 종의 major와 minor가 다른 종처럼 보임"],Inches(7.05),Inches(2.92),Inches(5.6),Inches(1.55))
im(s,Inches(0.4),Inches(4.8),Inches(6.2),Inches(2.35),"수컷 vs 암컷 비교 이미지 (같은 종)")
im(s,Inches(6.85),Inches(4.8),Inches(6.05),Inches(2.35),"major / minor / intermediate 비교")

# 8. 연구 목표 & 노벨티
s=mks(pr); bg(s); sn(s,8); bdg(s,"02  연구 목표"); hd(s,"연구 목표 & 5가지 노벨티")
for i,(tag,t1,desc,col) in enumerate([
    ("N1","성적 이형성 인식","종+성별 동시 학습 Multi-task 구조 제안\nSO vs SX ablation으로 기여도 정량 검증",AC),
    ("N2","수컷 크기 다형성","male_form 레이블을 보조 감독 신호로 활용\nFO / MT ablation으로 검증",A2),
    ("N3","전처리 전략 비교","4가지 전처리 모드 ablation study\nfull / bbox / seg_soft / seg_hard",AC),
    ("N4","희귀종 데이터 부족","커뮤니티 Data Flywheel로 점진적 해결\n시뮬레이션으로 효과 사전 검증",A2),
    ("N5","Open-Set 처리","외래종·비사슴벌레 강건 처리\nDINOv2 특징 공간 기반 OOD 탐지",AC)]):
    row,col2=divmod(i,3)
    lx=Inches(0.4+col2*4.3) if i<3 else Inches(2.0+(i-3)*4.67)
    t=Inches(2.38+row*2.35); ww=Inches(4.05) if i<3 else Inches(4.35)
    cd(s,lx,t,ww,Inches(2.15)); rc(s,lx,t,Inches(0.6),Inches(2.15),col)
    tx(s,tag,lx+Inches(0.05),t+Inches(0.75),Inches(0.5),Inches(0.55),sz=14,b=True,c=BG,al=PP_ALIGN.CENTER)
    tx(s,t1,lx+Inches(0.75),t+Inches(0.1),ww-Inches(0.9),Inches(0.48),sz=14,b=True)
    tx(s,desc,lx+Inches(0.75),t+Inches(0.62),ww-Inches(0.9),Inches(1.35),sz=12,c=GR)
tx(s,"타겟 저널: Methods in Ecology and Evolution",Inches(0.5),Inches(7.1),Inches(12.3),Inches(0.33),sz=12,c=A2,al=PP_ALIGN.RIGHT)

# 9. 전체 파이프라인 개요 (NEW)
s=mks(pr); bg(s); sn(s,9)
rc(s,Inches(0.4),Inches(0.27),Inches(4.2),Inches(0.44),A2)
tx(s,"전체 파이프라인 개요",Inches(0.45),Inches(0.25),Inches(4.1),Inches(0.48),sz=13,b=True,c=BG,al=PP_ALIGN.CENTER)
hd(s,"이 연구의 전체 흐름","데이터 수집 → 어노테이션 & 전처리 → 학습 → 추론 & 서비스")
stages=[
    ("① 데이터 수집",["iNaturalist API","현장 사진 (전문가 제공)","총 1,985장","종 레이블 포함"],AC),
    ("② 어노테이션 & 전처리",["Grounded SAM 2 → bbox/mask 자동","YOLOv8n-seg 학습","4가지 전처리 버전 생성","Label Studio (성별/형태)"],A2),
    ("③ 학습 (137개 실험)",["5 backbone architectures","4 preprocessing modes","Multi-task SO/SX/FO/MT","3 seeds × MLflow 추적"],YL),
    ("④ 추론 & 서비스",["YOLOv8 → OOD → Classifier","3단계 Open-Set 처리","beetledex.com 배포","Data Flywheel"],AC),
]
BW=Inches(2.9); BH=Inches(4.5); GAP=Inches(0.4); START=Inches(0.4)
for i,(ttl,items,col) in enumerate(stages):
    lx=START+i*(BW+GAP); t=Inches(2.35)
    cd(s,lx,t,BW,BH); rc(s,lx,t,BW,Inches(0.08),col)
    tx(s,ttl,lx+Inches(0.15),t+Inches(0.15),BW-Inches(0.3),Inches(0.5),sz=13,b=True,c=col)
    bl(s,[f"• {it}" for it in items],lx+Inches(0.15),t+Inches(0.72),BW-Inches(0.3),BH-Inches(0.8),sz=12)
    if i<3:
        ax=lx+BW+Inches(0.05); ay=Inches(4.55)
        tx(s,"→",ax,ay,GAP-Inches(0.1),Inches(0.55),sz=22,b=True,c=WH,al=PP_ALIGN.CENTER)

# 10. 데이터 수집
s=mks(pr); bg(s); sn(s,11); bdg(s,"03  데이터"); hd(s,"데이터 수집 & 자동 어노테이션")
cd(s,Inches(0.4),Inches(2.35),Inches(5.85),Inches(4.75))
cd(s,Inches(6.55),Inches(2.35),Inches(6.35),Inches(4.75))
bl(s,["iNaturalist API","  16종, Research Grade 필터링","  observation_id 포함 CSV 수집",
      "  학명 정규화 (cleaner.py)","","현장(field) 사진","  전문가 제공 표본 사진",
      "  image당 고유 ID 부여","","총 1,985장 (train 80 / val 10 / test 10%)"],
   Inches(0.65),Inches(2.45),Inches(5.35),Inches(4.55),hdr="데이터 출처")
tx(s,"자동 어노테이션 파이프라인",Inches(6.75),Inches(2.45),Inches(5.9),Inches(0.42),sz=14,b=True,c=A2)
for i,(step,desc) in enumerate([("텍스트 프롬프트",'"stag beetle"'),
    ("Grounded SAM 2","자동 bbox + 마스크 생성"),("YOLOv8n-seg 학습","class-agnostic detector"),
    ("서비스 적용","새 이미지 자동 전처리")]):
    t=Inches(2.95+i*1.05); rc(s,Inches(6.75),t,Inches(5.9),Inches(0.82),RGBColor(0x12,0x22,0x33))
    tx(s,step,Inches(6.95),t+Inches(0.05),Inches(3.0),Inches(0.38),sz=13,b=True,c=AC)
    tx(s,desc,Inches(6.95),t+Inches(0.42),Inches(5.5),Inches(0.34),sz=12)
    if i<3: tx(s,"↓",Inches(9.4),t+Inches(0.82),Inches(0.5),Inches(0.28),sz=14,c=A2,al=PP_ALIGN.CENTER)

# 10. 레이블 체계
s=mks(pr); bg(s); sn(s,11); bdg(s,"03  데이터"); hd(s,"레이블 체계 & Label Studio")
cd(s,Inches(0.4),Inches(2.35),Inches(12.5),Inches(3.1))
tx(s,"3단계 레이블 체계",Inches(0.65),Inches(2.45),Inches(5.0),Inches(0.42),sz=14,b=True,c=AC)
cxs=[0.65,1.35,4.05,7.1,9.5]; cws=[0.6,2.55,2.85,2.2,3.3]
for ci,ht in enumerate(["","레이블","출처","현황","활용 모드"]):
    if ht: tx(s,ht,Inches(cxs[ci]),Inches(2.92),Inches(cws[ci]),Inches(0.38),sz=12,b=True,c=A2)
for i,(lv,info,src,st,use) in enumerate([
    ("L1","종 (Species)","iNaturalist 자동","전체 1,985장","기본 학습 — species_only"),
    ("L2","종 + 성별 (Sex)","Label Studio 수동 라벨링","1,985장 완료","sex_only / multi_task"),
    ("L3","종 + 성별 + male_form","관리자 검수 (웹서비스)","진행중","form_only / multi_task")]):
    t=Inches(3.38+i*0.72); rc(s,Inches(cxs[0]),t,Inches(0.55),Inches(0.55),AC)
    tx(s,lv,Inches(cxs[0]),t,Inches(0.55),Inches(0.55),sz=13,b=True,c=BG,al=PP_ALIGN.CENTER)
    for ci,val in enumerate([info,src,st,use]):
        tx(s,val,Inches(cxs[ci+1]),t+Inches(0.07),Inches(cws[ci+1]),Inches(0.45),sz=12,c=AC if ci==3 else WH)
im(s,Inches(0.4),Inches(5.6),Inches(6.1),Inches(1.7),"Label Studio 라벨링 화면 스크린샷")
im(s,Inches(6.65),Inches(5.6),Inches(6.25),Inches(1.7),"성별/형태 라벨 분포 차트")

# 11. Specimen-level Split
s=mks(pr); bg(s); sn(s,12); bdg(s,"03  데이터")
hd(s,"Specimen-level Split","같은 개체가 train/test에 동시 포함되면 성능이 부풀려진다")
cd(s,Inches(0.4),Inches(2.35),Inches(5.85),Inches(4.75))
cd(s,Inches(6.55),Inches(2.35),Inches(6.35),Inches(4.75))
bl(s,["문제: 같은 개체를 여러 각도로 찍은 사진이",
      "  train과 test에 동시 포함 → 성능 과대 추정","",
      "해결: observation_id 기준 그룹핑",
      "  GroupShuffleSplit으로 그룹 단위 분할",
      "  같은 개체의 사진은 반드시 같은 split","",
      "분할: Train 80 / Val 10 / Test 10%",
      "Test: 199장 (hold-out, 최종 1회만 사용)"],
   Inches(0.65),Inches(2.45),Inches(5.35),Inches(4.55),hdr="왜 중요한가?")
im(s,Inches(6.75),Inches(2.45),Inches(5.9),Inches(2.4),"Specimen-level split 다이어그램")
bl(s,["종별 샘플 수 편차 큼:",
      "  최다: 왕사슴벌레 ~400장","  최소: 희귀종 < 10장",
      "macro F1으로 희귀종 별도 평가",
      "Data Flywheel로 점진적 보완 예정"],
   Inches(6.75),Inches(5.0),Inches(5.9),Inches(2.05),hdr="종별 분포 현황")

# 12. 전체 시스템 아키텍처
s=mks(pr); bg(s); sn(s,13); bdg(s,"04  ML 파이프라인"); hd(s,"전체 시스템 아키텍처")
for i,(nm,desc,col) in enumerate([("사용자","사진 업로드",A2),
    ("Next.js","Frontend — /api/* rewrites",DC),
    ("FastAPI","Backend — /predict  /feedback  /admin",DC),
    ("ML Layer","Segmenter(YOLOv8) → Classifier → OOD",AC),
    ("PostgreSQL","specimens · predictions · feedback · users",DC)]):
    t=Inches(2.38+i*0.96); rc(s,Inches(1.5),t,Inches(10.3),Inches(0.78),col)
    tx(s,nm,Inches(1.65),t+Inches(0.18),Inches(1.8),Inches(0.45),sz=13,b=True,c=BG if col==AC else WH)
    tx(s,desc,Inches(3.6),t+Inches(0.18),Inches(8.0),Inches(0.45),sz=13,c=BG if col==AC else GR)
    if i<4: tx(s,"↕",Inches(6.4),t+Inches(0.78),Inches(0.5),Inches(0.2),sz=12,c=A2,al=PP_ALIGN.CENTER)
bl(s,["Cloudflare Tunnel","→ beetledex.com","Redis 캐시","MLflow 추적","Alembic DB"],
   Inches(11.95),Inches(2.45),Inches(1.15),Inches(4.5),sz=10)

# 13. 전처리 4종
s=mks(pr); bg(s); sn(s,14); bdg(s,"04  ML 파이프라인")
hd(s,"전처리 전략 비교 — 4가지 모드","\"어떤 형태로 Classifier에 넘기느냐가 성능을 결정한다\" 는 가설 검증")
for i,(key,nm,desc,best) in enumerate([("full","원본 그대로","배경 포함 전체\n전처리 없음",True),
    ("bbox","BBox Crop","YOLOv8 bbox crop\n배경 일부 포함",False),
    ("seg_soft","Seg Soft","마스크 외부\nAlpha blending",False),
    ("seg_hard","Seg Hard","마스크 외부\n완전 제거 → 회색",False)]):
    lx=Inches(0.38+i*3.26)
    im(s,lx,Inches(2.35),Inches(3.08),Inches(2.5),f"{key} 예시 이미지")
    tx(s,("★  " if best else "")+nm,lx,Inches(4.95),Inches(3.08),Inches(0.42),sz=14,b=True,c=AC if best else WH)
    tx(s,desc,lx,Inches(5.4),Inches(3.08),Inches(0.75),sz=12,c=GR)
tx(s,"★ full이 최종 선택 — 배경·전체 맥락이 종 판별에 유효 (15~30%p 우위)",
   Inches(0.5),Inches(7.12),Inches(12.3),Inches(0.33),sz=12,c=AC)

# 14. 전처리 Ablation 결과
s=mks(pr); bg(s); sn(s,15); bdg(s,"04  ML 파이프라인")
hd(s,"전처리 Ablation 결과","3-seed 평균 val_acc (%) — 전 아키텍처에서 full이 우위")
im(s,Inches(0.4),Inches(2.35),Inches(6.3),Inches(4.75),"fig_preprocessing_ablation.png (bar chart)")
cd(s,Inches(6.95),Inches(2.35),Inches(5.95),Inches(4.75))
tx(s,"결과표",Inches(7.15),Inches(2.45),Inches(5.55),Inches(0.42),sz=13,b=True,c=A2)
for r,row in enumerate([["모드","ConvNeXt","EfficientNet","Swin","ViT"],
    ["full  ★","81.3%","84.0%","77.6%","64.0%"],["bbox","65.7%","72.0%","62.5%","56.0%"],
    ["seg_soft","64.5%","68.3%","60.6%","55.0%"],["seg_hard","65.5%","65.5%","60.6%","54.5%"]]):
    for c,cell in enumerate(row):
        tx(s,cell,Inches(7.15+c*1.1),Inches(2.95+r*0.52),Inches(1.05),Inches(0.48),
           sz=11,b=(r<=1),c=A2 if r==0 else (AC if r==1 else WH))
bl(s,["full이 전 아키텍처 15~30%p 우위","세그멘테이션 과정에서 형태 특징 손실",
      "GradCAM: 배경 편향 확인 (Prismognathus)","Flywheel로 데이터 다양성 확보 예정"],
   Inches(7.15),Inches(5.2),Inches(5.55),Inches(1.75))

# 15. 5개 아키텍처
s=mks(pr); bg(s); sn(s,16); bdg(s,"04  ML 파이프라인"); hd(s,"5개 아키텍처 비교 설계")
for i,(nm,kind,desc) in enumerate([("ConvNeXt-Tiny","CNN","ImageNet supervised pretrain\n448×448 입력"),
    ("EfficientNet-B3","CNN","경량·빠른 추론 / 448×448 입력"),
    ("Swin-Tiny","Transformer","Window Attention / 448×448 입력"),
    ("ViT-Small","Transformer","Pure ViT (ViT-B/16) / 448→224 resize"),
    ("DINOv2-ViT-S/14","Self-supervised","OOD backbone 공유 / 소수 데이터 강점")]):
    row,col2=divmod(i,3)
    if i<3: lx,t,ww=Inches(0.4+col2*4.35),Inches(2.35),Inches(4.1)
    else: lx,t,ww=Inches(2.0+(i-3)*4.67),Inches(4.55),Inches(4.35)
    cd(s,lx,t,ww,Inches(1.95))
    clr=AC if kind=="CNN" else (A2 if kind=="Transformer" else YL)
    rc(s,lx,t,ww,Inches(0.08),clr)
    tx(s,nm,lx+Inches(0.18),t+Inches(0.15),ww-Inches(0.3),Inches(0.45),sz=14,b=True)
    rc(s,lx+Inches(0.18),t+Inches(0.65),Inches(1.3),Inches(0.32),clr)
    tx(s,kind,lx+Inches(0.18),t+Inches(0.65),Inches(1.3),Inches(0.32),sz=10,b=True,c=BG,al=PP_ALIGN.CENTER)
    tx(s,desc,lx+Inches(0.18),t+Inches(1.05),ww-Inches(0.3),Inches(0.75),sz=12,c=GR)
cd(s,Inches(0.4),Inches(6.55),Inches(12.5),Inches(0.78))
bl(s,["공통: pretrained backbone fine-tuning  |  LR 1e-4  |  AdamW  |  CosineAnnealingLR  |  3 seeds [42,123,456]"],
   Inches(0.65),Inches(6.65),Inches(12.0),Inches(0.55),sz=12)

# 16. Multi-task 분류기
s=mks(pr); bg(s); sn(s,17); bdg(s,"04  ML 파이프라인"); hd(s,"Multi-task 분류기 구조")
cd(s,Inches(0.4),Inches(2.35),Inches(5.85),Inches(4.75))
cd(s,Inches(6.55),Inches(2.35),Inches(6.35),Inches(4.75))
tx(s,"모델 구조",Inches(0.65),Inches(2.45),Inches(5.35),Inches(0.42),sz=14,b=True,c=AC)
tx(s,"    Backbone (5종 중 선택)\n          ↓\n    Feature Vector\n    ┌──────┼──────┐\n  종 Head  성별   형태\n  (16cls) (2cls) (3cls)",
   Inches(0.65),Inches(2.92),Inches(5.35),Inches(2.1),sz=14,c=A2)
tx(s,"Loss 함수:",Inches(0.65),Inches(5.08),Inches(5.35),Inches(0.38),sz=13,b=True,c=A2)
tx(s,"L = L_species\n  + λ_sex  × L_sex\n  + λ_form × L_male_form\n\nunknown(-1) → masked loss로 제외",
   Inches(0.65),Inches(5.5),Inches(5.35),Inches(1.4),sz=13)
tx(s,"4가지 학습 모드",Inches(6.75),Inches(2.45),Inches(5.9),Inches(0.42),sz=14,b=True,c=AC)
for i,(tag,code,inputs,note) in enumerate([("SO","species_only","종 레이블만","기준선"),
    ("SX","sex_only","종 + 성별","N1 검증 — 성적 이형성"),
    ("FO","form_only","종 + male_form","N2 검증 — 크기 다형성"),
    ("MT","multi_task","종 + 성별 + 형태","최종 모델 후보")]):
    t=Inches(2.95+i*1.05); clr=AC if tag in("SO","MT") else A2
    rc(s,Inches(6.75),t,Inches(0.55),Inches(0.88),clr)
    tx(s,tag,Inches(6.75),t+Inches(0.2),Inches(0.55),Inches(0.45),sz=13,b=True,c=BG,al=PP_ALIGN.CENTER)
    tx(s,code,Inches(7.38),t+Inches(0.05),Inches(2.0),Inches(0.38),sz=12,b=True)
    tx(s,inputs,Inches(7.38),t+Inches(0.48),Inches(2.5),Inches(0.34),sz=11,c=GR)
    tx(s,note,Inches(10.0),t+Inches(0.2),Inches(2.7),Inches(0.45),sz=12,c=clr)
tx(s,"λ_sex: 0.3  |  λ_form: 0.2  |  λ ablation으로 cherry-pick 아님 증명",
   Inches(6.75),Inches(7.1),Inches(5.9),Inches(0.33),sz=11,c=GR)

# 17. Open-Set Recognition
s=mks(pr); bg(s); sn(s,18); bdg(s,"04  ML 파이프라인")
hd(s,"Open-Set Recognition — 3단계 처리","실서비스에서 외래종·타 곤충·비곤충 이미지가 입력될 수 있다")
for i,(lyr,t1,mth,res,col) in enumerate([
    ("Layer 1","사슴벌레 감지","YOLOv8n-seg Detector","미검출 →\nno_beetle",A2),
    ("Layer 2","OOD 탐지","DINOv2 feature distance\n(centroid 기반)","OOD score 높음 →\nuncertain",YL),
    ("Layer 3","종 분류","Multi-task Classifier\nconfidence < 0.4","→ low_confidence\n정상 → identified",AC)]):
    lx=Inches(0.4+i*4.3); cd(s,lx,Inches(2.35),Inches(4.05),Inches(3.8))
    rc(s,lx,Inches(2.35),Inches(4.05),Inches(0.08),col)
    tx(s,lyr,lx+Inches(0.2),Inches(2.48),Inches(1.2),Inches(0.38),sz=11,b=True,c=col)
    tx(s,t1,lx+Inches(0.2),Inches(2.9),Inches(3.65),Inches(0.45),sz=15,b=True)
    tx(s,mth,lx+Inches(0.2),Inches(3.42),Inches(3.65),Inches(0.65),sz=12,c=GR)
    rc(s,lx+Inches(0.2),Inches(4.15),Inches(3.65),Inches(0.04),col)
    tx(s,res,lx+Inches(0.2),Inches(4.25),Inches(3.65),Inches(0.78),sz=13,b=True,c=col)
    if i<2: tx(s,"↓",Inches(4.38+i*4.3),Inches(3.7),Inches(0.5),Inches(0.4),sz=20,c=A2,al=PP_ALIGN.CENTER)
cd(s,Inches(0.4),Inches(6.25),Inches(12.5),Inches(1.05))
for i,(rt,msg,col) in enumerate([("no_beetle","사슴벌레를 찾을 수 없어요",A2),
    ("uncertain","인식하기 어려운 사진이에요",YL),
    ("low_confidence","확신하기 어렵습니다. 후보: ...",GR),
    ("identified","종명 + 성별 + 신뢰도 표시",AC)]):
    lx=Inches(0.65+i*3.1)
    tx(s,rt,lx,Inches(6.35),Inches(3.0),Inches(0.38),sz=11,b=True,c=col)
    tx(s,msg,lx,Inches(6.75),Inches(3.0),Inches(0.38),sz=11)

# 18. 실험 설계
s=mks(pr); bg(s); sn(s,19); bdg(s,"05  실험 결과"); hd(s,"실험 설계 — 총 137개","완전 요인 설계 + Ablation")
for i,(nm,cnt,desc,cmd,col) in enumerate([
    ("메인 Sweep","120개","5 arch × 4 prep × 2 mode(SO, MT) × 3 seed","python scripts/sweep.py",AC),
    ("MT Ablation","6개","Best (arch, prep) × 2 mode(SX, FO) × 3 seed\nMLflow에서 베스트 조합 자동 선택","python scripts/sweep_mt_ablation.py",A2),
    ("λ Ablation","11개","λ_sex [0.05~1.0] × λ_form=0.2 = 6  /  λ_form [0.05~1.0] × λ_sex=0.3 = 5","python scripts/sweep_lambda.py",YL)]):
    t=Inches(2.38+i*1.65); cd(s,Inches(0.4),t,Inches(12.5),Inches(1.48))
    rc(s,Inches(0.4),t,Inches(0.08),Inches(1.48),col)
    tx(s,nm,Inches(0.65),t+Inches(0.1),Inches(2.5),Inches(0.45),sz=15,b=True,c=col)
    tx(s,cnt,Inches(3.3),t+Inches(0.1),Inches(1.0),Inches(0.45),sz=22,b=True)
    tx(s,desc,Inches(0.65),t+Inches(0.62),Inches(8.0),Inches(0.72),sz=12,c=GR)
    tx(s,cmd,Inches(9.0),t+Inches(0.48),Inches(3.7),Inches(0.38),sz=11,c=A2)
tx(s,"run_name: {arch}_{prep}_{mode}_s{seed}   |   MLflow로 완료된 run 자동 스킵",
   Inches(0.5),Inches(7.12),Inches(12.3),Inches(0.33),sz=11,c=GR)

# 19. 아키텍처 비교 결과
s=mks(pr); bg(s); sn(s,20); bdg(s,"05  실험 결과")
hd(s,"아키텍처 비교 결과","full 전처리, 3-seed 평균 val_acc (%)")
im(s,Inches(0.4),Inches(2.35),Inches(6.3),Inches(4.75),"fig_architecture_comparison.png")
cd(s,Inches(6.95),Inches(2.35),Inches(5.95),Inches(4.75))
tx(s,"결과표 (full, SO, 3-seed 평균)",Inches(7.15),Inches(2.45),Inches(5.55),Inches(0.42),sz=13,b=True,c=A2)
for r,row in enumerate([["아키텍처","Val Acc","Macro F1","Val-Test 갭"],
    ["EfficientNet-B3","84.0%","66.6%","-12.5%p"],["ConvNeXt-Tiny","81.3%","63.7%","-11.4%p"],
    ["Swin-Tiny  ★","77.6%","61.9%","-6.9%p"],["DINOv2-ViT","72.1%","57.2%","-6.9%p"],
    ["ViT-Small","64.0%","49.1%","-7.4%p"]]):
    for c,cell in enumerate(row):
        tx(s,cell,Inches(7.15+c*1.45),Inches(2.95+r*0.6),Inches(1.4),Inches(0.55),
           sz=12,b=(r<=1 or r==3),c=A2 if r==0 else (AC if r==3 else WH))
bl(s,["Val 기준 EfficientNet 1위","Test 기준 Swin/EfficientNet 공동 1위",
      "Swin이 val-test 갭 가장 작음 (일반화)","상위 3모델 95% CI 중첩 — 통계적 유의차 없음"],
   Inches(7.15),Inches(6.0),Inches(5.55),Inches(1.0))

# 20. MT Ablation
s=mks(pr); bg(s); sn(s,21); bdg(s,"05  실험 결과")
hd(s,"Multi-task Ablation 결과","EfficientNet-B3, full 전처리, 3-seed 평균")
im(s,Inches(0.4),Inches(2.35),Inches(6.3),Inches(2.25),"fig_multitask_effect.png (SO vs SX vs FO vs MT)")
im(s,Inches(0.4),Inches(4.7),Inches(6.3),Inches(2.4),"fig_female_vs_male_accuracy.png (N1 핵심 증거)")
cd(s,Inches(6.95),Inches(2.35),Inches(5.95),Inches(4.75))
tx(s,"MT Ablation 결과표",Inches(7.15),Inches(2.45),Inches(5.55),Inches(0.42),sz=13,b=True,c=A2)
for r,row in enumerate([["Mode","Val Acc","Sex Acc","비고"],["SO","84.0%","-","기준선"],
    ["SX ★","85.0%","96.6%","+1.0%p"],["FO","82.2%","-","-1.8%p"],["MT","83.2%","94.3%","-0.8%p"]]):
    for c,cell in enumerate(row):
        tx(s,cell,Inches(7.15+c*1.4),Inches(2.95+r*0.55),Inches(1.35),Inches(0.5),
           sz=12,b=(r==0 or r==2),c=A2 if r==0 else (AC if r==2 else WH))
bl(s,["SX 보조 학습: val_acc +1.0%p 향상 → N1 성적 이형성 유효",
      "","주의: SX val 우위가 test에서 역전 (SX 70.4% < SO 72.9%)",
      "→ 과적합, 데이터 확장 후 재검증","","DINOv2 SX: 5.7% → 완전 붕괴",
      "Self-supervised backbone은 multi-task gradient에 민감"],
   Inches(7.15),Inches(5.1),Inches(5.55),Inches(1.85))

# 21. GradCAM & t-SNE
s=mks(pr); bg(s); sn(s,22); bdg(s,"05  실험 결과")
hd(s,"정성 분석 — GradCAM & t-SNE","Best 모델: Swin-T SO s456")
im(s,Inches(0.4),Inches(2.35),Inches(6.3),Inches(4.75),"gradcam_*.png (정분류 3장 + 오분류 2장)")
cd(s,Inches(6.95),Inches(2.35),Inches(5.95),Inches(2.1))
tx(s,"GradCAM 분석 결과",Inches(7.15),Inches(2.45),Inches(5.55),Inches(0.42),sz=13,b=True,c=AC)
bl(s,["정분류: 머리·큰턱 히트맵 집중 → 형태학적으로 의미 있는 학습",
      "오분류 1: Dorcus 속간 혼동 (형태 유사성)",
      "오분류 2: 흰 배경에 히트맵 집중 (데이터 편향)"],
   Inches(7.15),Inches(2.92),Inches(5.55),Inches(1.35))
im(s,Inches(6.95),Inches(4.55),Inches(5.95),Inches(2.55),"tsne_features.png (val+test 397장, 종별·성별 컬러)")

# 22. Test set 최종
s=mks(pr); bg(s); sn(s,23); bdg(s,"05  실험 결과")
hd(s,"Test Set 최종 결과","Hold-out 199장 — 학습·검증 중 절대 미사용")
cd(s,Inches(0.4),Inches(2.35),Inches(12.5),Inches(2.5))
tx(s,"최종 평가 결과 (Bootstrap 95% CI)",Inches(0.65),Inches(2.45),Inches(6.0),Inches(0.42),sz=13,b=True,c=A2)
for r,row in enumerate([["모델","Test Acc","Macro F1","95% CI","Val-Test 갭"],
    ["Swin-T SO s456  ★","72.9%","57.2%","[66.8, 78.9]","-6.9%p"],
    ["EfficientNet SO","72.9%","55.4%","[66.8, 78.9]","-12.5%p"],
    ["ConvNeXt SO","72.4%","55.8%","[66.3, 78.9]","-11.4%p"],
    ["EfficientNet SX","70.4%","54.3%","[64.3, 76.9]","-15.5%p"],
    ["DINOv2 SO","66.8%","54.6%","[60.3, 73.4]","-6.9%p"]]):
    for c,cell in enumerate(row):
        tx(s,cell,Inches([0.65,3.9,5.3,6.6,9.3][c]),Inches(2.92+r*0.38),
           Inches([3.15,1.3,1.2,2.6,1.7][c]),Inches(0.38),
           sz=12,b=(r==0 or r==1),c=A2 if r==0 else (AC if r==1 else WH))
im(s,Inches(0.4),Inches(5.0),Inches(6.1),Inches(2.3),"fig_fewshot_scatter.png (N4)")
im(s,Inches(6.65),Inches(5.0),Inches(6.25),Inches(2.3),"reliability_diagram.png (Temperature Scaling 전·후)")
tx(s,"Best: Swin-T SO s456 — test acc 공동 1위, macro_f1 최고, val-test 갭 최소",
   Inches(0.5),Inches(7.12),Inches(12.3),Inches(0.33),sz=12,c=AC)

# 23. 오분류 패턴
s=mks(pr); bg(s); sn(s,24); bdg(s,"05  실험 결과")
hd(s,"오분류 패턴 분석","Swin-T SO s456, val set — 전 오분류가 데이터 부족에 기인")
im(s,Inches(0.4),Inches(2.35),Inches(5.5),Inches(4.75),"오분류 예시 이미지 (experiments/error_analysis/)")
cd(s,Inches(6.15),Inches(2.35),Inches(6.75),Inches(4.75))
tx(s,"주요 오분류 패턴",Inches(6.35),Inches(2.45),Inches(6.35),Inches(0.42),sz=14,b=True,c=AC)
for i,(t1,desc) in enumerate([
    ("속 내 혼동 (Dorcus spp.)","D. consentaneus → D. titanus (4건)\n→ 형태학적으로 합리적인 실수"),
    ("다속 혼동","Prosopocoilus inclinatus → 6가지 종\n(총 12건, 학습 샘플 절대 부족)"),
    ("배경 편향","Prismognathus → 흰 트레이에 GradCAM 집중\n동일 배경 데이터 편중")]):
    t=Inches(2.95+i*1.42); rc(s,Inches(6.35),t,Inches(0.06),Inches(1.2),A2)
    tx(s,t1,Inches(6.55),t+Inches(0.05),Inches(2.1),Inches(0.5),sz=13,b=True)
    tx(s,desc,Inches(8.75),t+Inches(0.05),Inches(3.9),Inches(1.1),sz=12,c=GR)
tx(s,"Prosopocoilus inclinatus 우선 수집  |  Flywheel 확장으로 해결 예정",
   Inches(6.35),Inches(7.1),Inches(6.55),Inches(0.33),sz=11,c=A2)

# 24. 웹서비스 기능
s=mks(pr); bg(s); sn(s,25); bdg(s,"06  웹서비스 & 배포")
hd(s,"beetledex.com — 서비스 기능","일반 사용자가 사진 한 장으로 종을 동정")
im(s,Inches(0.4),Inches(2.35),Inches(6.3),Inches(4.75),"beetledex.com 메인 → 결과 페이지 스크린샷")
cd(s,Inches(6.95),Inches(2.35),Inches(5.95),Inches(4.75))
tx(s,"주요 기능",Inches(7.15),Inches(2.45),Inches(5.55),Inches(0.42),sz=14,b=True,c=AC)
for i,(k,v) in enumerate([("AI 분류","사진 업로드 → 종 동정 (0.x초)"),
    ("결과 표시","종명·성별·신뢰도·Top-3 후보"),("피드백","성별 확인 → 커뮤니티 레이블(L2)"),
    ("갤러리","커뮤니티 업로드 모아보기"),("랭킹","종별 관찰 수 · 유저 활동 순위"),
    ("종 도감","16종 설명·분포·이미지"),("분포 지도","GPS 기반 관찰 위치 시각화"),
    ("관리자","L3 검수 대시보드")]):
    t=Inches(2.95+i*0.52)
    tx(s,k,Inches(7.15),t,Inches(1.35),Inches(0.45),sz=12,b=True,c=A2)
    tx(s,v,Inches(8.6),t,Inches(4.1),Inches(0.45),sz=12)

# 25. 인프라 & Flywheel
s=mks(pr); bg(s); sn(s,26); bdg(s,"06  웹서비스 & 배포"); hd(s,"인프라 & Data Flywheel")
cd(s,Inches(0.4),Inches(2.35),Inches(5.85),Inches(4.75))
cd(s,Inches(6.55),Inches(2.35),Inches(6.35),Inches(4.75))
tx(s,"기술 스택",Inches(0.65),Inches(2.45),Inches(5.35),Inches(0.42),sz=14,b=True,c=AC)
for i,(k,v) in enumerate([("Frontend","Next.js 14 + Tailwind CSS"),("Backend","FastAPI + SQLAlchemy async"),
    ("DB","PostgreSQL 16  /  Redis 7"),("ML","PyTorch 2.6+cu124 — Swin-T SO"),
    ("OOD","DINOv2 centroid 기반"),("Tunnel","Cloudflare Zero Trust Tunnel"),
    ("서버","Contabo VPS — Singapore"),("추적","MLflow (Docker, localhost:5001)")]):
    t=Inches(2.95+i*0.52)
    tx(s,k,Inches(0.65),t,Inches(1.4),Inches(0.45),sz=12,b=True,c=A2)
    tx(s,v,Inches(2.15),t,Inches(3.8),Inches(0.45),sz=12)
tx(s,"Data Flywheel",Inches(6.75),Inches(2.45),Inches(5.9),Inches(0.42),sz=14,b=True,c=AC)
for i,(step,desc) in enumerate([("사용자 업로드","사진 한 장"),("모델 예측 (L1)","종 자동 동정"),
    ("피드백 (L2)","사용자 성별 확인"),("검수 (L3)","관리자 male_form 라벨"),
    ("재학습","성능 향상"),("더 많은 사용자","더 많은 데이터")]):
    t=Inches(2.95+i*0.68); clr=AC if i%2==0 else A2
    rc(s,Inches(6.75),t,Inches(0.08),Inches(0.55),clr)
    tx(s,step,Inches(6.95),t+Inches(0.05),Inches(2.3),Inches(0.42),sz=13,b=True,c=clr)
    tx(s,desc,Inches(9.35),t+Inches(0.05),Inches(3.3),Inches(0.42),sz=13)
tx(s,"목표: 1,985장 → 4,000장 → 재학습 (learning curve 분석)",
   Inches(6.75),Inches(7.05),Inches(5.9),Inches(0.38),sz=11,c=GR)

# 26. Flywheel 시뮬레이션
s=mks(pr); bg(s); sn(s,27); bdg(s,"06  웹서비스 & 배포")
hd(s,"Data Flywheel 시뮬레이션 (N4)","희귀종 데이터 추가 수집 시 성능 회복 예측")
im(s,Inches(0.4),Inches(2.35),Inches(7.5),Inches(4.75),"simulate_flywheel.py 결과 (accuracy recovery curve)")
cd(s,Inches(8.15),Inches(2.35),Inches(4.75),Inches(4.75))
bl(s,["시뮬레이션 설정:","  희귀종 비율 점진적 증가","  → 성능 변화 추적","",
      "결과 요약:","  데이터 2배 → F1 예상 회복",
      "  우선 수집:","  Prosopocoilus inclinatus","  Prismognathus spp.","",
      "N4 해결 경로 사전 검증 완료"],
   Inches(8.35),Inches(2.45),Inches(4.35),Inches(4.55))

# 27. 한계 고찰
s=mks(pr); bg(s); sn(s,28); bdg(s,"07  한계 & 전망")
hd(s,"한계 고찰 — 딥러닝의 근본적 한계","더 많은 데이터가 항상 답인가?  인간과 딥러닝의 인지 방식은 본질적으로 다르다")
cd(s,Inches(0.4),Inches(2.35),Inches(6.2),Inches(4.75))
cd(s,Inches(6.85),Inches(2.35),Inches(6.05),Inches(2.15))
cd(s,Inches(6.85),Inches(4.65),Inches(6.05),Inches(2.45))
tx(s,"딥러닝의 근본적 한계",Inches(0.65),Inches(2.45),Inches(5.7),Inches(0.42),sz=14,b=True,c=RD)
for i,(t1,desc) in enumerate([
    ("데이터 의존성","성능 ∝ 데이터 양 / 희귀종 10장 미만 → 동정 불가\n인간: 처음 보는 개체도 형태 추론으로 구별 가능"),
    ("Closed-set 한계","훈련 분포 밖은 알 수 없음\n새로운 종 발견 불가 — 알려진 것만 분류\nOOD 탐지로 완화하지만 근본 해결 아님"),
    ("설명 불가능성","왜 그 종인지 설명 못함\n분류학: 진단 형질로 명확히 기술 가능"),
    ("새 알고리즘이 필요한가?","더 많은 데이터 → 점진적 개선\nFew-shot / Meta-learning → 근본 해결 가능성\nFoundation models → 현재 가장 현실적 접근")]):
    t=Inches(2.95+i*1.05); clr=RD if i<3 else YL
    rc(s,Inches(0.65),t,Inches(0.06),Inches(0.88),clr)
    tx(s,t1,Inches(0.85),t+Inches(0.02),Inches(5.1),Inches(0.38),sz=13,b=True)
    tx(s,desc,Inches(0.85),t+Inches(0.42),Inches(5.1),Inches(0.42),sz=11,c=GR)
tx(s,"우리 연구의 대응",Inches(7.05),Inches(2.45),Inches(5.65),Inches(0.42),sz=14,b=True,c=AC)
bl(s,["N4 Flywheel: 데이터 의존성 점진적 해결","N5 OOD: closed-set 한계 완화",
      "Temperature Scaling: 신뢰도 보정","Specimen-level split: 과대 추정 방지"],
   Inches(7.05),Inches(2.92),Inches(5.65),Inches(1.42))
tx(s,"앞으로의 방향",Inches(7.05),Inches(4.75),Inches(5.65),Inches(0.42),sz=14,b=True,c=A2)
bl(s,["Few-shot learning / Meta-learning","  → 소수 데이터로 새 종 학습",
      "Foundation models (DINOv2, SAM)","  → 가장 현실적인 접근",
      "형태학적 형질 직접 학습 → Explainable AI"],
   Inches(7.05),Inches(5.22),Inches(5.65),Inches(1.75))

# 28. 향후 계획
s=mks(pr); bg(s); sn(s,29); bdg(s,"07  한계 & 전망"); hd(s,"향후 계획 & 논문 구성")
for i,(t1,items,col) in enumerate([
    ("단기\n(2026 여름)",["커뮤니티 데이터 수집 ~4,000장","데이터 2배 도달 시 재학습",
      "OOD 강화 — 외래종 이미지 추가","Cloudflare R2 이미지 저장"],AC),
    ("논문\n(2026 하반기)",["전처리 ablation + 아키텍처 비교","Multi-task SO/SX/FO/MT ablation",
      "Data Flywheel 시뮬레이션 포함","타겟: Methods in Ecology and Evolution"],A2),
    ("장기\n서비스",["GPS 분포 지도 전체 공개","관리자 대시보드 고도화",
      "재학습 자동화 파이프라인","나비·잠자리 등 타 분류군 확장"],YL)]):
    lx=Inches(0.4+i*4.33); cd(s,lx,Inches(2.35),Inches(4.1),Inches(4.75))
    rc(s,lx,Inches(2.35),Inches(4.1),Inches(0.07),col)
    tx(s,t1,lx+Inches(0.2),Inches(2.45),Inches(3.7),Inches(0.65),sz=14,b=True,c=col)
    bl(s,[f"• {it}" for it in items],lx+Inches(0.2),Inches(3.18),Inches(3.7),Inches(3.75),sz=12)
rc(s,Inches(0.4),Inches(7.15),Inches(12.5),Inches(0.3),DC)
tx(s,"논문: \"Sexual Dimorphism-Aware Fine-Grained Classification of Korean Lucanidae\"",
   Inches(0.65),Inches(7.18),Inches(12.0),Inches(0.28),sz=11,c=GR)

# 29. 결론
s=mks(pr); bg(s); sn(s,30); bdg(s,"07  한계 & 전망"); hd(s,"결론 — 무엇을 했나")
for i,(cat,desc) in enumerate([
    ("데이터","iNaturalist 1,985장 수집 · Grounded SAM 자동 어노테이션\nSpecimen-level split으로 신뢰할 수 있는 평가 환경 구축"),
    ("모델","5 arch × 4 prep × 4 mode = 137개 실험\nBest: Swin-T SO s456 (test acc 72.9%, macro F1 57.2%)"),
    ("노벨티","N1 SX 보조 학습 유효  |  N3 full 전처리 압도적 우위\nN5 OOD 3단계 처리  |  N4 Flywheel 경로 확보"),
    ("서비스","beetledex.com 운영 중 — 커뮤니티 데이터 수집 시작\n사슴벌레 → 곤충 전반으로 확장 가능한 프레임워크 제시")]):
    row,col2=divmod(i,2); lx=Inches(0.4+col2*6.55); t=Inches(2.38+row*2.35)
    cd(s,lx,t,Inches(6.2),Inches(2.15)); rc(s,lx,t,Inches(0.07),Inches(2.15),AC)
    tx(s,cat,lx+Inches(0.22),t+Inches(0.12),Inches(5.7),Inches(0.42),sz=15,b=True,c=AC)
    tx(s,desc,lx+Inches(0.22),t+Inches(0.6),Inches(5.7),Inches(1.38),sz=13)

# 30. 마무리
s=mks(pr); bg(s)
rc(s,Inches(0),Inches(0),Inches(0.28),H,AC)
rc(s,Inches(0.28),Inches(4.95),Inches(13.05),Inches(0.06),A2)
tx(s,"Taxonomy & Classification",Inches(1),Inches(1.6),Inches(11.3),Inches(0.7),sz=20,c=A2,al=PP_ALIGN.CENTER)
tx(s,"딥러닝으로 사슴벌레 분류기 만들기",Inches(1),Inches(2.3),Inches(11.3),Inches(0.95),sz=36,b=True,al=PP_ALIGN.CENTER)
tx(s,"BeetleDex — beetledex.com",Inches(1),Inches(3.35),Inches(11.3),Inches(0.55),sz=18,c=AC,al=PP_ALIGN.CENTER)
tx(s,"감사합니다",Inches(1),Inches(5.2),Inches(11.3),Inches(0.8),sz=34,b=True,al=PP_ALIGN.CENTER)
tx(s,"Q & A",Inches(1),Inches(6.1),Inches(11.3),Inches(0.5),sz=20,c=GR,al=PP_ALIGN.CENTER)

pr.save("beetledex_midterm_2026.pptx")
print(f"저장 완료: beetledex_midterm_2026.pptx  ({len(pr.slides)}장)")
