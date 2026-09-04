"""통계 요건에서 역산한 종별 수집 목표와 부족분.

DESIGN.md §13 의 수치를 산출한 스크립트.
실행: python scripts/analysis/collection_targets.py
출력: experiments/analysis_v7/
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "experiments" / "analysis_v7"
OUTDIR.mkdir(parents=True, exist_ok=True)

import glob, math, collections
from pathlib import Path

OUT = []
R = str(ROOT) + '/'
SCR = str(OUTDIR) + "/"

KO = {
    "Dorcus_titanus_castanicolor": "넓적사슴벌레",
    "Prosopocoilus_inclinatus_inclinatus": "톱사슴벌레",
    "Lucanus_maculifemoratus_dybowskyi": "사슴벌레",
    "Dorcus_rectus_rectus": "애사슴벌레",
    "Prismognathus_dauricus": "다우리아사슴벌레",
    "Prosopocoilus_astacoides_blanchardi": "두점박이사슴벌레",
    "Dorcus_consentaneus_consentaneus": "참넓적사슴벌레",
    "Dorcus_hopei_binodulosus": "왕사슴벌레",
    "Dorcus_rubrofemoratus_rubrofemoratus": "홍다리사슴벌레",
    "Platycerus_hongwonpyoi_hongwonpyoi": "원표애보라사슴벌레",
    "Figulus_punctatus": "길쭉꼬마사슴벌레",
    "Aegus_laevicollis_subnitidus": "꼬마넓적사슴벌레",
    "Nigidius_miwai": "뿔꼬마사슴벌레",
    "Dorcus_carinulatus_koreanus": "털보왕사슴벌레",
    "Dorcus_tenuihirsutus": "엷은털왕사슴벌레",
    "Figulus_binodulus": "큰꼬마사슴벌레",
}

# 현재 보유 개체 수 = processed 이미지 수 (관찰당 1장이므로 개체 수와 같음)
cur = collections.Counter()
for f in glob.glob(R + "data/processed/*/*"):
    cur[Path(f).parts[-2]] += 1

OUT.append("=== 통계 요건에서 역산한 목표 (개체 수 기준) ===")
OUT.append("")
OUT.append("[test] 종별 recall CI 반폭")
for n in [30, 50, 80, 100]:
    hw = 1.96 * math.sqrt(0.7 * 0.3 / n)
    OUT.append(f"  종당 test {n:3d}개체 -> +/- {hw*100:4.1f}%p")

OUT.append("")
OUT.append("[test] 전체 정확도 CI 반폭 및 5%p 차이 검출력 (10종 합계)")
for per in [30, 50, 80, 100]:
    n = per * 10
    hw = 1.96 * math.sqrt(0.75 * 0.25 / n)
    disc = 0.20 * n
    pi = 0.5 + (0.05 * n) / (2 * disc)
    se = math.sqrt(0.25 / disc)
    z = (pi - 0.5) / se
    from scipy import stats
    power = 1 - stats.norm.cdf(1.96 - z)
    OUT.append(f"  종당 {per:3d} (총 {n:4d}) -> CI +/-{hw*100:4.1f}%p, 5%p 검출력 {power*100:5.1f}%")

OUT.append("")
OUT.append("=== 3단계 목표안 (종당 개체 수) ===")
plans = {
    "최소": dict(train=100, val=25, test=50),
    "권장": dict(train=250, val=40, test=80),
    "충분": dict(train=400, val=50, test=100),
}
for k, v in plans.items():
    tot = sum(v.values())
    OUT.append(f"  {k}: train {v['train']} + val {v['val']} + test {v['test']} = 종당 {tot}개체  (10종 = {tot*10:,})")

OUT.append("")
OUT.append("=== 종별 현재 보유 vs 목표 부족분 (개체 수) ===")
OUT.append(f"{'국명':22s}{'현재':>6s}{'최소175':>9s}{'권장370':>9s}")
order = sorted(KO, key=lambda s: -cur.get(s, 0))
gap_min = gap_rec = 0
for s in order:
    c = cur.get(s, 0)
    gm, gr = max(0, 175 - c), max(0, 370 - c)
    gap_min += gm
    gap_rec += gr
    OUT.append(f"{KO[s]:22s}{c:6d}{gm:9d}{gr:9d}")
OUT.append(f"{'합계':22s}{sum(cur.values()):6d}{gap_min:9d}{gap_rec:9d}")

OUT.append("")
OUT.append("=== 성별·형태 요건 ===")
OUT.append("  male_form 종별 분위수 추정: 종당 라벨된 수컷 30개체 이상 필요")
OUT.append("  성별 층화 평가: 종당 test 암컷 15개체 이상 -> 암컷 비율 최소 35-40% 유지")
OUT.append("  현재 전체 성비: male 60.5% / female 36.4% / unknown 3.1%")

open(SCR + "collect_out.txt", "w", encoding="utf-8").write("\n".join(OUT))
