"""
Generate Learning Curve Visualizations for GA Mutation Study
Creates plots similar to the reference image showing accuracy vs iterations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import seaborn as sns

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

STUDY_ROOT = Path(__file__).parent
SNAPSHOTS_DIR = STUDY_ROOT / "snapshots"
RESULTS_DIR = STUDY_ROOT / "analysis_results"
RESULTS_DIR.mkdir(exist_ok=True)

print("="*60)
print("GENERATING LEARNING CURVE VISUALIZATIONS")
print("="*60)

# ============================================================
# PART 1: LOAD SNAPSHOT DATA
# ============================================================

def load_snapshot_accuracy(part, condition, dataset, seed, checkpoint):
    """Load accuracy from a specific checkpoint snapshot."""
    snap_path = SNAPSHOTS_DIR / part / condition / dataset / f"seed_{seed}" / f"iter_{checkpoint:06d}.csv"
    
    if not snap_path.exists():
        return None
    
    df = pd.read_csv(snap_path)
    if len(df) == 0:
        return None
    
    # Calculate population-weighted accuracy
    weighted_acc = np.average(df['accuracy'], weights=df['numerosity'])
    return weighted_acc

# Checkpoints from study design
CHECKPOINTS = [0, 10000, 50000, 100000, 200000, 300000, 400000, 500000]

# ============================================================
# PART 2: EXTRACT LEARNING CURVES FOR PART B.1
# ============================================================

print("\nExtracting learning curves for Part B.1 (GA/Mutation Comparison)...")

datasets = ['ham', 'isic']
conditions = ['ga_on_mut_on', 'ga_on_mut_off', 'ga_off']
seeds = [42, 43, 44, 45, 46]

learning_curves = {}

for dataset in datasets:
    for condition in conditions:
        key = f"{dataset}_{condition}"
        curves_per_seed = []
        
        for seed in seeds:
            curve = []
            for checkpoint in CHECKPOINTS:
                acc = load_snapshot_accuracy('PartB1', condition, dataset, seed, checkpoint)
                if acc is not None:
                    curve.append(acc)
                else:
                    curve.append(np.nan)
            
            if not all(np.isnan(curve)):  # Only add if we have data
                curves_per_seed.append(curve)
        
        if curves_per_seed:
            learning_curves[key] = curves_per_seed
            print(f"  {key}: {len(curves_per_seed)} seeds loaded")

# ============================================================
# PART 3: GENERATE PLOTS (SIMILAR TO REFERENCE)
# ============================================================

print("\nGenerating publication-quality plots...")

# Color scheme
colors = {
    'ga_on_mut_on': '#D62728',    # Red (with mutation)
    'ga_on_mut_off': '#1F77B4',   # Blue (GA only)
    'ga_off': '#2CA02C'            # Green (covering only)
}

labels = {
    'ga_on_mut_on': 'GA + Mutation',
    'ga_on_mut_off': 'GA Only',
    'ga_off': 'Covering Only'
}

# ============================================================
# FIGURE 1: HAM10000 Learning Curves
# ============================================================

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

for condition in conditions:
    key = f"ham_{condition}"
    if key not in learning_curves:
        continue
    
    curves = np.array(learning_curves[key])
    mean_curve = np.nanmean(curves, axis=0)
    std_curve = np.nanstd(curves, axis=0)
    
    iterations = [c/1000 for c in CHECKPOINTS]  # Convert to thousands
    
    ax.plot(iterations, mean_curve, 
            color=colors[condition], 
            label=labels[condition], 
            linewidth=2)
    
    # Add std as shaded region
    ax.fill_between(iterations, 
                     mean_curve - std_curve, 
                     mean_curve + std_curve,
                     color=colors[condition], 
                     alpha=0.2)

ax.set_xlabel('Iterations (× 1000)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('HAM10000 - Learning Curves', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.3, 1.0])

plt.tight_layout()
plt.savefig(RESULTS_DIR / "learning_curve_ham.png", dpi=300, bbox_inches='tight')
print(f"  Saved: learning_curve_ham.png")

# ============================================================
# FIGURE 2: ISIC2019 Learning Curves
# ============================================================

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

for condition in conditions:
    key = f"isic_{condition}"
    if key not in learning_curves:
        continue
    
    curves = np.array(learning_curves[key])
    mean_curve = np.nanmean(curves, axis=0)
    std_curve = np.nanstd(curves, axis=0)
    
    iterations = [c/1000 for c in CHECKPOINTS]
    
    ax.plot(iterations, mean_curve, 
            color=colors[condition], 
            label=labels[condition], 
            linewidth=2)
    
    ax.fill_between(iterations, 
                     mean_curve - std_curve, 
                     mean_curve + std_curve,
                     color=colors[condition], 
                     alpha=0.2)

ax.set_xlabel('Iterations (× 1000)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('ISIC2019 - Learning Curves', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.3, 1.0])

plt.tight_layout()
plt.savefig(RESULTS_DIR / "learning_curve_isic.png", dpi=300, bbox_inches='tight')
print(f"  Saved: learning_curve_isic.png")

# ============================================================
# FIGURE 3: COMBINED COMPARISON (2x1 GRID like reference)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# HAM10000
ax = axes[0]
for condition in conditions:
    key = f"ham_{condition}"
    if key not in learning_curves:
        continue
    
    curves = np.array(learning_curves[key])
    mean_curve = np.nanmean(curves, axis=0)
    std_curve = np.nanstd(curves, axis=0)
    iterations = [c/1000 for c in CHECKPOINTS]
    
    ax.plot(iterations, mean_curve, color=colors[condition], 
            label=labels[condition], linewidth=2)
    ax.fill_between(iterations, mean_curve - std_curve, mean_curve + std_curve,
                     color=colors[condition], alpha=0.2)

ax.set_xlabel('Iterations (× 1000)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('HAM10000', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.3, 1.0])

# ISIC2019
ax = axes[1]
for condition in conditions:
    key = f"isic_{condition}"
    if key not in learning_curves:
        continue
    
    curves = np.array(learning_curves[key])
    mean_curve = np.nanmean(curves, axis=0)
    std_curve = np.nanstd(curves, axis=0)
    iterations = [c/1000 for c in CHECKPOINTS]
    
    ax.plot(iterations, mean_curve, color=colors[condition], 
            label=labels[condition], linewidth=2)
    ax.fill_between(iterations, mean_curve - std_curve, mean_curve + std_curve,
                     color=colors[condition], alpha=0.2)

ax.set_xlabel('Iterations (× 1000)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('ISIC2019', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.3, 1.0])

plt.tight_layout()
plt.savefig(RESULTS_DIR / "learning_curves_combined.png", dpi=300, bbox_inches='tight')
print(f"  Saved: learning_curves_combined.png")

# ============================================================
# FIGURE 4: ERROR BAR PLOT (like "Full Dataset" in reference)
# ============================================================

print("\nGenerating error bar comparison plot...")

final_accuracies = {}

for dataset in datasets:
    for condition in conditions:
        key = f"{dataset}_{condition}"
        if key not in learning_curves:
            continue
        
        # Get final checkpoint (500k iterations)
        curves = np.array(learning_curves[key])
        final_accs = curves[:, -1]  # Last checkpoint
        final_accs = final_accs[~np.isnan(final_accs)]
        
        if len(final_accs) > 0:
            final_accuracies[key] = {
                'mean': np.mean(final_accs),
                'std': np.std(final_accs),
                'n': len(final_accs)
            }

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

x_positions = []
x_labels = []
x_pos = 0

for dataset in datasets:
    for condition in conditions:
        key = f"{dataset}_{condition}"
        if key not in final_accuracies:
            continue
        
        stats = final_accuracies[key]
        
        ax.errorbar(x_pos, stats['mean'], yerr=stats['std'],
                    fmt='o', markersize=10, capsize=5,
                    color=colors[condition],
                    label=labels[condition] if x_pos < 3 else "")
        
        x_positions.append(x_pos)
        x_labels.append(f"{dataset.upper()}\n{labels[condition]}")
        x_pos += 1
    
    x_pos += 0.5  # Gap between datasets

ax.set_ylabel('Final Accuracy (500k iterations)', fontsize=12, fontweight='bold')
ax.set_title('GA Mutation Study: Final Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0.3, 1.0])

plt.tight_layout()
plt.savefig(RESULTS_DIR / "final_accuracy_comparison.png", dpi=300, bbox_inches='tight')
print(f"  Saved: final_accuracy_comparison.png")

print("\n✅ All learning curve visualizations generated!")
print(f"   Location: {RESULTS_DIR}/")
print("\nFiles created:")
print("  - learning_curve_ham.png")
print("  - learning_curve_isic.png")
print("  - learning_curves_combined.png (like reference)")
print("  - final_accuracy_comparison.png (error bars)")
