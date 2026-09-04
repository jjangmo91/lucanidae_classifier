"""설치된 의존성의 라이선스를 훑어 카피레프트 전염 위험을 점검한다.

배경: 자체 코드는 MIT 지만 ultralytics 가 AGPL-3.0 이라 결합 저작물 배포와
네트워크 서비스 운영에 조건이 붙는다. 자세한 내용은 THIRD_PARTY_LICENSES.md.

의존성을 추가·변경한 뒤 실행해 새로운 카피레프트가 들어왔는지 확인한다.

사용법:
    python scripts/check_licenses.py
    python scripts/check_licenses.py --all      # 전체 패키지 출력
"""

from __future__ import annotations

import argparse
import importlib.metadata as md

# 전염성이 강해 결합 저작물 전체에 조건을 거는 것들
STRONG = ("AGPL", "GPLV3", "GPLV2", "GPL-3", "GPL-2", "GENERAL PUBLIC LICENSE")
# 라이브러리 단위로만 조건이 붙는 것들 (그냥 import 해서 쓰면 무해)
WEAK = ("LGPL", "LESSER GENERAL PUBLIC", "MPL", "MOZILLA PUBLIC", "EUPL", "CDDL")


def _license_of(dist) -> str:
    m = dist.metadata
    classifiers = [c for c in (m.get_all("Classifier") or []) if c.startswith("License ::")]
    if classifiers:
        return classifiers[0].split("::")[-1].strip()
    return (m.get("License") or "UNKNOWN").strip().replace("\n", " ")[:70]


def classify(lic: str) -> str:
    u = lic.upper()
    # LGPL 은 GPL 을 부분 문자열로 포함하므로 약한 쪽을 먼저 본다
    if any(k in u for k in WEAK):
        return "weak"
    if any(k in u for k in STRONG):
        return "strong"
    return "permissive"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="전체 패키지 출력")
    a = ap.parse_args()

    rows = []
    for d in md.distributions():
        try:
            name = d.metadata["Name"]
            if not name:
                continue
            lic = _license_of(d)
            rows.append((name.lower(), lic, classify(lic)))
        except Exception:
            continue
    rows = sorted(set(rows))

    strong = [r for r in rows if r[2] == "strong"]
    weak = [r for r in rows if r[2] == "weak"]

    if a.all:
        for n, l, k in rows:
            print(f"  {k:10s} {n:32s} {l}")
        print()

    print("=== 강한 카피레프트 (결합 저작물 전체에 조건) ===")
    if strong:
        for n, l, _ in strong:
            print(f"  [!] {n:30s} {l}")
    else:
        print("  없음")

    print("\n=== 약한 카피레프트 (라이브러리 단위, import 사용은 무해) ===")
    if weak:
        for n, l, _ in weak:
            print(f"      {n:30s} {l}")
    else:
        print("  없음")

    print(f"\n총 {len(rows)}개 · 강함 {len(strong)} · 약함 {len(weak)}")
    if strong:
        print("\n조치가 필요하다. THIRD_PARTY_LICENSES.md 의 선택지를 참고할 것.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
