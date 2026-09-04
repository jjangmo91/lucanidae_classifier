"""Label Studio keypoint export -> R 비율 -> 종별 분위수 -> male_form 3분류.

규칙: docs/male_form_protocol.md
근거: DESIGN.md 4.4

라벨러는 점 4개만 찍는다. 등급 판정은 전부 여기서 한다.
임계값은 train split 만으로 계산해 동결하고 val/test 에 그대로 적용한다.

사용법:
    python scripts/compute_male_form.py \
        --annotations data/labels/male_form_export.json \
        --out data/labels/male_form.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
NEEDED = ("mandible_apex", "mandible_base", "body_front", "body_rear")


# ---------------------------------------------------------------- 파싱

def _points(task: dict) -> dict[str, tuple[float, float]]:
    """Label Studio 결과에서 keypoint 를 픽셀 좌표로 뽑는다.

    Label Studio 는 x, y 를 원본 크기 대비 백분율로 준다.
    가로세로 비율이 다르므로 반드시 original_width/height 를 곱해 픽셀로 바꾼 뒤
    거리를 계산해야 한다. 백분율 그대로 재면 길이가 왜곡된다.
    """
    out: dict[str, tuple[float, float]] = {}
    for ann in task.get("annotations", []):
        for r in ann.get("result", []):
            if r.get("type") != "keypointlabels":
                continue
            v = r.get("value", {})
            labels = v.get("keypointlabels") or []
            if not labels:
                continue
            w = r.get("original_width")
            h = r.get("original_height")
            if not w or not h:
                continue
            out[labels[0]] = (v["x"] / 100.0 * w, v["y"] / 100.0 * h)
    return out


def _choice(task: dict, name: str) -> str | None:
    for ann in task.get("annotations", []):
        for r in ann.get("result", []):
            if r.get("from_name") == name:
                ch = r.get("value", {}).get("choices") or []
                if ch:
                    return ch[0]
    return None


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def parse_export(path: Path) -> pd.DataFrame:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for t in tasks:
        data = t.get("data", {})
        img = data.get("image") or data.get("local_path") or ""
        rec = {
            "image": Path(str(img)).name,
            "species": data.get("species"),
            "split": data.get("split"),
            "sex": _choice(t, "sex"),
            "measurable": _choice(t, "measurable"),
            "R": None,
            "M": None,
            "B": None,
            "exclude_reason": None,
        }

        if rec["sex"] != "male":
            rec["exclude_reason"] = "not_male"
            rows.append(rec)
            continue
        if rec["measurable"] and rec["measurable"] != "yes":
            rec["exclude_reason"] = rec["measurable"]
            rows.append(rec)
            continue

        pts = _points(t)
        missing = [k for k in NEEDED if k not in pts]
        if missing:
            rec["exclude_reason"] = "missing_points:" + ",".join(missing)
            rows.append(rec)
            continue

        M = _dist(pts["mandible_apex"], pts["mandible_base"])
        B = _dist(pts["body_front"], pts["body_rear"])
        if B <= 0:
            rec["exclude_reason"] = "zero_body_length"
        else:
            rec.update(M=M, B=B, R=M / B)
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 등급

def fit_thresholds(df: pd.DataFrame, scope: dict, lo: float, hi: float) -> pd.DataFrame:
    """train split 만으로 종별 분위수를 계산한다."""
    tr = df[(df.split == "train") & df.R.notna() & df.species.isin(scope["in_scope"])]
    rows = []
    for sp, g in tr.groupby("species"):
        n = len(g)
        if n < scope["min_labeled_males"]:
            print(f"  [경고] {sp}: train 계측 {n}개체 (기준 {scope['min_labeled_males']}) - 제외")
            continue
        rows.append({
            "species": sp, "n_train": n,
            "q_low": round(float(g.R.quantile(lo)), 6),
            "q_high": round(float(g.R.quantile(hi)), 6),
            "quantile_low": lo, "quantile_high": hi,
        })
    return pd.DataFrame(rows)


def assign(df: pd.DataFrame, th: pd.DataFrame, scope: dict) -> pd.DataFrame:
    na = set(scope.get("not_applicable") or {})
    lut = {r.species: (r.q_low, r.q_high) for r in th.itertuples()}

    def grade(r):
        if r.species in na:
            return "not_applicable"
        if r.species not in lut:
            return "not_applicable"
        if pd.isna(r.R):
            return "unknown"
        lo, hi = lut[r.species]
        if r.R >= hi:
            return "major"
        if r.R <= lo:
            return "minor"
        return "intermediate"

    df = df.copy()
    df["male_form"] = df.apply(grade, axis=1)
    return df


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--scope", default=str(ROOT / "configs" / "male_form_scope.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data" / "labels" / "male_form.csv"))
    a = ap.parse_args()

    scope = yaml.safe_load(Path(a.scope).read_text(encoding="utf-8"))
    lo = scope["quantiles"]["lower"]
    hi = scope["quantiles"]["upper"]

    df = parse_export(Path(a.annotations))
    print(f"[1] 파싱 {len(df)} tasks | 계측 성공 {df.R.notna().sum()}")

    th = fit_thresholds(df, scope, lo, hi)
    print(f"[2] 임계값 산출 {len(th)} 종 (train split 기준, 동결)")

    out = assign(df, th, scope)
    print("[3] 등급 분포")
    print(out.male_form.value_counts().to_string())

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    th_path = out_path.with_name("male_form_thresholds.csv")
    th.to_csv(th_path, index=False, encoding="utf-8-sig")

    # 민감도 분석용 대안 경계
    alts = []
    for alo, ahi in scope.get("alternatives", []):
        t2 = fit_thresholds(df, scope, alo, ahi)
        o2 = assign(df, t2, scope)
        vc = o2.male_form.value_counts()
        alts.append({"quantile_low": alo, "quantile_high": ahi,
                     **{k: int(vc.get(k, 0)) for k in ("major", "intermediate", "minor")}})
    if alts:
        pd.DataFrame(alts).to_csv(
            out_path.with_name("male_form_sensitivity.csv"), index=False, encoding="utf-8-sig")

    print(f"\n저장: {out_path}")
    print(f"      {th_path}  (논문 보충자료 공개 대상)")


if __name__ == "__main__":
    main()
