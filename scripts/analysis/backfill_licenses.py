"""iNaturalist 사진 라이선스·출처표기 역조회 백필.

DESIGN.md §11.2 의 수치를 산출한 스크립트.
실행: python scripts/analysis/backfill_licenses.py
출력: experiments/analysis_v7/
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "experiments" / "analysis_v7"
OUTDIR.mkdir(parents=True, exist_ok=True)

"""보유 중인 iNaturalist 관찰 ID로 사진 라이선스를 역조회한다.
목적: 재배포 가능한 이미지가 몇 %인지 실측 (저널 데이터 공개 대응).
"""
import collections, json, time
import pandas as pd
import requests

R = str(ROOT) + '/'
df = pd.read_csv(R + "data/raw/inaturalist/metadata.csv")
ids = [int(i) for i in df.observation_id.tolist()]
print("총 관찰:", len(ids))

API = "https://api.inaturalist.org/v1/observations"
rows = []
BATCH = 100
for k in range(0, len(ids), BATCH):
    chunk = ids[k:k + BATCH]
    try:
        r = requests.get(API, params={"id": ",".join(map(str, chunk)), "per_page": BATCH},
                         timeout=30, headers={"User-Agent": "lucanidae-classifier-research"})
        r.raise_for_status()
        for obs in r.json().get("results", []):
            ph = (obs.get("photos") or [{}])[0]
            rows.append({
                "observation_id": obs.get("id"),
                "obs_license": obs.get("license_code"),
                "photo_license": ph.get("license_code"),
                "photo_id": ph.get("id"),
                "attribution": ph.get("attribution"),
                "user": (obs.get("user") or {}).get("login"),
                "obscured": obs.get("obscured"),
                "taxon_threatened": (obs.get("taxon") or {}).get("threatened"),
            })
    except Exception as e:
        print("batch", k, "fail:", e)
    time.sleep(1.0)
    if k % 500 == 0:
        print("  fetched", len(rows), flush=True)

out = pd.DataFrame(rows)
out.to_csv(R + "../lic_probe.csv", index=False, encoding="utf-8-sig")
print("\n조회 성공:", len(out), "/", len(ids))
print("\n=== photo_license 분포 ===")
print(out.photo_license.fillna("ALL_RIGHTS_RESERVED").value_counts().to_string())
print("\n=== observation license 분포 ===")
print(out.obs_license.fillna("ALL_RIGHTS_RESERVED").value_counts().to_string())
print("\n=== obscured(좌표 가림) ===")
print(out.obscured.value_counts(dropna=False).to_string())
print("\n=== taxon threatened ===")
print(out.taxon_threatened.value_counts(dropna=False).to_string())
json.dump({"n": len(out)}, open("C:/Users/doruc/AppData/Local/Temp/claude/c--Users-doruc-project-lucanidae-classifier/fd26116e-2056-4dad-87ea-6f43b2ca314d/scratchpad/lic_done.json", "w"))
