import json, numpy as np, pandas as pd
from pathlib import Path

out_dir = Path('experiments/test_results/prep_ablation')
rows = []
for f in out_dir.glob('test_metrics_efficientnet_full_*.json'):
    with open(f) as fp: d = json.load(fp)
    stem = f.stem.replace('test_metrics_','')
    parts = stem.split('_')
    seed=parts[-1]; mode=parts[-2]
    rows.append({'mode':mode,'seed':seed,
                 'male':d.get('male_species_acc'),
                 'female':d.get('female_species_acc'),
                 'acc':d.get('accuracy')})

df = pd.DataFrame(rows)
grouped = df.groupby('mode')[['male','female','acc']].mean()*100
print('EfficientNet full 3-seed mean:')
print(grouped.round(1).to_string())

# SX test
sx_files = list(Path('experiments/test_results').glob('test_metrics_efficientnet*SX*.json'))
print()
print('SX test files:', len(sx_files))
for f in sx_files:
    with open(f) as fp: d = json.load(fp)
    m = d.get('male_species_acc',0)
    fe = d.get('female_species_acc',0)
    a = d.get('accuracy',d.get('test_acc',0))
    print(f.name, 'male='+str(round(m*100,1))+'% female='+str(round(fe*100,1))+'% acc='+str(round(a*100,1))+'%')
