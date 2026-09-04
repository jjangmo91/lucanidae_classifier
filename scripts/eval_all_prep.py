"""
전처리별 test 성능 전수 평가 스크립트
5 arch × 4 prep × 2 mode × 3 seed = 120개 조합

사용법:
    conda activate lucanidae_classifier_venv
    python scripts/eval_all_prep.py

완료된 것은 자동 스킵 (experiments/test_results/prep_ablation/ 기준)
"""

import subprocess
import sys
import os
from pathlib import Path

# 프로젝트 루트를 작업 디렉토리로 고정
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)

WEIGHTS_DIR = Path("models/weights/sweep")
OUT_DIR     = Path("experiments/test_results/prep_ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# prep → data_dir 매핑
DATA_DIRS = {
    "full":     "data/final",
    "bbox":     "data/final_bbox",
    "seg_soft": "data/final_seg_soft",
    "seg_hard": "data/final_seg_hard",
}

ARCHS = ["convnext", "efficientnet", "swin", "vit", "dinov2"]
PREPS = ["full", "bbox", "seg_soft", "seg_hard"]
MODES = ["SO", "MT"]
SEEDS = ["s42", "s123", "s456"]

total   = len(ARCHS) * len(PREPS) * len(MODES) * len(SEEDS)
done    = 0
skipped = 0
failed  = 0

print(f"총 {total}개 조합 평가 시작\n{'='*60}")

for arch in ARCHS:
    for prep in PREPS:
        for mode in MODES:
            for seed in SEEDS:
                stem    = f"{arch}_{prep}_{mode}_{seed}"
                weights = WEIGHTS_DIR / f"{stem}.pth"
                out_json = OUT_DIR / f"test_metrics_{stem}.json"

                # 이미 완료된 것 스킵
                if out_json.exists():
                    skipped += 1
                    continue

                # 웨이트 파일 없으면 스킵
                if not weights.exists():
                    print(f"  [SKIP] 웨이트 없음: {stem}")
                    skipped += 1
                    continue

                data_dir = DATA_DIRS[prep]
                print(f"  [{done+skipped+1}/{total}] {stem} ...", end=" ", flush=True)

                env = os.environ.copy()
                env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

                result = subprocess.run(
                    [sys.executable, "scripts/evaluate_test.py",
                     "--weights",    str(weights),
                     "--data_dir",   data_dir,
                     "--output_dir", str(OUT_DIR)],
                    capture_output=True, text=True,
                    cwd=str(PROJECT_ROOT),
                    env=env
                )

                if result.returncode == 0:
                    done += 1
                    # 결과에서 test_acc 추출해서 출력
                    for line in result.stdout.split("\n"):
                        if "Test Acc" in line or "test_acc" in line:
                            print(line.strip())
                            break
                    else:
                        print("완료")
                else:
                    failed += 1
                    print(f"실패: {result.stderr[-200:]}")

print(f"\n{'='*60}")
print(f"완료: {done}  |  스킵: {skipped}  |  실패: {failed}")
print(f"결과 위치: {OUT_DIR}")
