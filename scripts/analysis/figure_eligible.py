"""논문 그림에 실을 수 있는 사진을 집계하고 후보 목록을 만든다.

DESIGN.md 11.7 의 수치를 산출한 스크립트.
실행: python scripts/analysis/figure_eligible.py
출력: experiments/analysis_v7/figure_eligible.csv  (종별 집계)
      experiments/analysis_v7/figure_candidates.csv (사진별 후보 + 출처표시)

배경: Elsevier 는 논문에 타인의 저작물을 실을 때 서면 허락을 요구한다.
CC BY-NC 는 상업 출판사의 지면과 충돌하므로 그림에 쓸 수 없다.
학습에 쓰는 것과는 별개 문제다.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "experiments" / "analysis_v7"
OUTDIR.mkdir(parents=True, exist_ok=True)

# 그림에 실을 수 있는 iNaturalist 라이선스. NC/SA/ND/ARR 은 제외한다.
FIGURE_OK_LICENSES = {"cc0", "cc-by"}

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
}


def main() -> None:
    meta = pd.read_csv(ROOT / "data/raw/merged_metadata.csv")
    meta["stem"] = meta.image_path.apply(lambda p: Path(str(p)).stem)

    rows = []
    for f in glob.glob(str(ROOT / "data/final/*/*/*")):
        p = Path(f).parts
        rows.append({"split": p[-3], "species": p[-2], "stem": Path(f).stem,
                     "path": str(Path(f).relative_to(ROOT)).replace("\\", "/")})
    if not rows:
        raise SystemExit("data/final 에 이미지가 없습니다.")

    d = pd.DataFrame(rows).merge(meta, on="stem", how="left")
    d["license"] = d.photo_license.fillna("UNKNOWN").str.strip().str.lower()
    d["fig_ok"] = d.license.isin(FIGURE_OK_LICENSES) | (d.source == "field")

    total, ok = len(d), int(d.fig_ok.sum())
    print(f"data/final 총 {total}장 | 그림 사용 가능 {ok}장 ({ok/total*100:.1f}%)")
    print()
    print("구성:")
    comp = d[d.fig_ok].groupby(["source", "license"]).size()
    print(comp.to_string())
    print()

    t = d.groupby("species").agg(total=("stem", "size"), fig_ok=("fig_ok", "sum"))
    t["ratio_pct"] = (t.fig_ok / t.total * 100).round(1)
    t["species_ko"] = [KO.get(i, "") for i in t.index]
    t = t.sort_values("total", ascending=False)
    print("=== 종별 ===")
    print(t.to_string())

    blocked = t[t.fig_ok == 0]
    if len(blocked):
        print("\n[경고] 그림에 쓸 사진이 0장인 종:", list(blocked.index))
    thin = t[(t.fig_ok > 0) & (t.fig_ok <= 6)]
    if len(thin):
        print("\n[주의] 선택지가 6장 이하인 종 (대안 없음):")
        for i, r in thin.iterrows():
            print(f"  {r.species_ko:14s} {int(r.fig_ok)}장")

    t.to_csv(OUTDIR / "figure_eligible.csv", encoding="utf-8-sig")

    cand = d[d.fig_ok][["species", "split", "path", "license", "source",
                        "photo_attribution", "observation_id"]].copy()
    cand["species_ko"] = cand.species.map(KO)
    cand = cand.sort_values(["species", "source", "license"])
    cand.to_csv(OUTDIR / "figure_candidates.csv", index=False, encoding="utf-8-sig")

    print(f"\n저장: {OUTDIR / 'figure_eligible.csv'}")
    print(f"      {OUTDIR / 'figure_candidates.csv'}  ({len(cand)}장, 출처표시 포함)")
    print("\n그림용으로 확정한 사진은 docs/figure_image_whitelist.csv 로 옮겨 관리한다.")


if __name__ == "__main__":
    main()
