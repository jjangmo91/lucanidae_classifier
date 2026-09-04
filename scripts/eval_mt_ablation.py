import subprocess, sys, os
from pathlib import Path

PROJECT_ROOT = Path('.').resolve()
OUT_DIR = 'experiments/test_results/mt_ablation'
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

import json
for mode in ['SX','FO']:
    for seed in ['s42','s123','s456']:
        w = f'models/weights/mt_ablation/efficientnet_full_{mode}_{seed}.pth'
        out_f = f'{OUT_DIR}/test_metrics_efficientnet_full_{mode}_{seed}.json'
        if Path(out_f).exists():
            with open(out_f) as fp: d = json.load(fp)
            m = d.get('male_species_acc',0); fe = d.get('female_species_acc',0)
            print(f'  {mode} {seed}: male={round(m*100,1)}% female={round(fe*100,1)}% acc={round(d.get(\"accuracy\",0)*100,1)}%')
            continue
        print(f'{mode} {seed}: not yet evaluated - run eval_all_prep.py with conda env')
