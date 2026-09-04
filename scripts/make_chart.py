import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

font_path = 'C:/Windows/Fonts/malgun.ttf'
if Path(font_path).exists():
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('experiments/analysis/all_runs.csv')
stats = pd.read_csv('experiments/analysis/table_dataset_stats.csv', index_col=0)

arch_map  = {'convnext_tiny':'ConvNeXt-Tiny','efficientnet_b3':'EfficientNet-B3',
             'swin_tiny':'Swin-Tiny','dinov2_vits14':'DINOv2-ViT','vit_small':'ViT-S/16'}
colors    = {'ConvNeXt-Tiny':'#2E86AB','EfficientNet-B3':'#A23B72',
             'Swin-Tiny':'#F18F01','DINOv2-ViT':'#C73E1D','ViT-S/16':'#7B2D8B'}

so_full = df[(df['mode']=='species_only')&(df['prep']=='full')].copy()
so_full['arch2'] = so_full['arch'].map(arch_map)

f1_cols = [c for c in df.columns if c.startswith('f1_')]
species_list = [c.replace('f1_','') for c in f1_cols]

train_counts = {}
if 'train' in stats.columns:
    for sp in species_list:
        sp_clean = sp.replace('_',' ').strip()
        if sp in stats.index:
            train_counts[sp] = stats.loc[sp,'train']
        else:
            for idx in stats.index:
                if sp.lower() in idx.lower() or idx.lower() in sp.lower():
                    train_counts[sp] = stats.loc[idx,'train']
                    break

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')

plotted_archs = set()
for arch_code, arch_label in arch_map.items():
    sub = so_full[so_full['arch']==arch_code]
    if len(sub)==0: continue
    color = colors[arch_label]
    for sp, f1_col in zip(species_list, f1_cols):
        f1_val = sub[f1_col].mean() * 100
        n_train = train_counts.get(sp, None)
        if n_train is not None and not np.isnan(f1_val):
            label = arch_label if arch_label not in plotted_archs else '_nolegend_'
            ax.scatter(n_train, f1_val, color=color, alpha=0.75, s=60, label=label, zorder=3)
            plotted_archs.add(arch_label)

ax.set_xlabel('Training Samples per Class', fontsize=11)
ax.set_ylabel('Per-class F1 (%)', fontsize=11)
ax.set_title('Few-shot Performance: Training Samples vs Per-class F1\n(full preprocessing, SO, 3-seed mean)', fontsize=11.5, fontweight='bold')
ax.yaxis.grid(True, color='#E8E8E8', linewidth=0.8, zorder=0); ax.set_axisbelow(True)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
for sp in ['left','bottom']: ax.spines[sp].set_color('#BBBBBB')
ax.tick_params(colors='#333333', labelsize=10)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=10, frameon=True, framealpha=0.9, edgecolor='#CCCCCC', loc='lower right')
ax.text(0.02, 0.97, '* 학습 데이터 적을수록 F1 분산 크고 불안정 (N4 희귀종 문제)',
    transform=ax.transAxes, ha='left', va='top', fontsize=9.5, color='#555555', style='italic')

plt.tight_layout()
plt.savefig('experiments/analysis/fig_fewshot_scatter.png', dpi=150, bbox_inches='tight', facecolor='white')
print('fewshot saved')
