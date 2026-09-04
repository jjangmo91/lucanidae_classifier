"""좌표·날짜 단독 종 예측 (위치 배제의 근거가 되는 음성 대조군).

DESIGN.md §12.6-A / E9 의 수치를 산출한 스크립트.
실행: python scripts/analysis/metadata_leakage.py
출력: experiments/analysis_v7/
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "experiments" / "analysis_v7"
OUTDIR.mkdir(parents=True, exist_ok=True)

import glob
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

OUT = []
R = str(ROOT) + '/'
SCR = str(OUTDIR) + "/"

meta = pd.read_csv(R + "data/raw/merged_metadata.csv")
meta["stem"] = meta.image_path.apply(lambda p: Path(str(p)).stem)

rows = []
for f in glob.glob(R + "data/final/*/*/*"):
    p = Path(f).parts
    rows.append({"split": p[-3], "species": p[-2], "stem": Path(f).stem})
d = pd.DataFrame(rows).merge(meta, on="stem", how="left")
OUT.append(f"joined={len(d)} missing_meta={d.latitude.isna().sum()}")

d["dt"] = pd.to_datetime(d.observed_on, errors="coerce")
d["doy"] = d.dt.dt.dayofyear
d["year"] = d.dt.dt.year
d["doy_sin"] = np.sin(2 * np.pi * d.doy / 365)
d["doy_cos"] = np.cos(2 * np.pi * d.doy / 365)

FEATS = ["latitude", "longitude", "doy_sin", "doy_cos", "year"]
d = d.dropna(subset=["latitude", "longitude"])

tr, te = d[d.split == "train"], d[d.split == "test"]
OUT.append(f"train={len(tr)} test={len(te)}")
ytr, yte = tr.species.values, te.species.values

dummy = DummyClassifier(strategy="most_frequent").fit(tr[FEATS].values, ytr)
pr = dummy.predict(te[FEATS].values)
OUT.append(f"MAJORITY        acc={accuracy_score(yte,pr)*100:5.1f}  macroF1={f1_score(yte,pr,average='macro')*100:5.1f}")

for name, fs in [("COORD_ONLY", ["latitude", "longitude"]),
                 ("DATE_ONLY", ["doy_sin", "doy_cos", "year"]),
                 ("COORD+DATE", FEATS)]:
    c = HistGradientBoostingClassifier(max_iter=300, random_state=42).fit(tr[fs].values, ytr)
    pr = c.predict(te[fs].values)
    OUT.append(f"{name:15s} acc={accuracy_score(yte,pr)*100:5.1f}  macroF1={f1_score(yte,pr,average='macro')*100:5.1f}")

OUT.append("REFERENCE image model Swin SO s456: acc=72.9 macroF1=57.2")

# per-species recall for the metadata model
c = HistGradientBoostingClassifier(max_iter=300, random_state=42).fit(tr[FEATS].values, ytr)
pr = c.predict(te[FEATS].values)
OUT.append("")
OUT.append("per-species (metadata COORD+DATE): n_test / correct")
for sp in sorted(set(yte)):
    m = yte == sp
    OUT.append(f"  {sp:40s} {m.sum():3d} / {(pr[m]==sp).sum():3d}")

open(SCR + "meta_out.txt", "w", encoding="utf-8").write("\n".join(OUT))
