"""
Phase 1.3: Selective Degree-3 Feature Engineering
Adds degree-3 polynomial interactions on top 20 features only.
Controlled expansion: 20 choose 3 = 1,140 new 3-way interactions.
"""
import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

# Paths
CHAMPION_PATH = Path(__file__).parent.parent / "Derived_Features_Champion"
RESULTS_DIR = Path(__file__).parent / "phase1_results"

RESULTS_DIR.mkdir(exist_ok=True)

print("="*60)
print("PHASE 1.3: SELECTIVE DEGREE-3 FEATURE ENGINEERING")
print("="*60)

# Load Top 100 champion features
print("\nLoading Top 100 champion features...")
champion_data_path = Path(__file__).parent / "champion_top100_features.csv"

if not champion_data_path.exists():
    print("ERROR: champion_top100_features.csv not found!")
    print("Please run generate_top100_dataset.py first.")
    sys.exit(1)

df_full = pd.read_csv(champion_data_path)

# Get feature names (all columns except 'label')
selected_names = [col for col in df_full.columns if col != 'label']

X_top100 = df_full.drop('label', axis=1).values
y = df_full['label'].values

print(f"Current features: {X_top100.shape[1]}")
print(f"Samples: {len(y)}")

# ============================================================
# IDENTIFY TOP 20 FEATURES BY MI
# ============================================================

print("\nRanking Top 100 features by Mutual Information...")
mi_scores_top100 = mutual_info_classif(X_top100, y, random_state=42)

# Get top 20
top20_indices_in_top100 = np.argsort(mi_scores_top100)[-20:][::-1]
top20_mi_scores = mi_scores_top100[top20_indices_in_top100]
top20_names = [selected_names[i] for i in top20_indices_in_top100]

print("\nTop 20 Features for Degree-3 Expansion:")
for rank, (idx, name, score) in enumerate(zip(top20_indices_in_top100, top20_names, top20_mi_scores), 1):
    print(f"  {rank:2d}. {name[:50]:50s} | MI={score:.4f}")

# Extract Top 20 feature matrix
X_top20 = X_top100[:, top20_indices_in_top100]

# ============================================================
# GENERATE DEGREE-3 INTERACTIONS (SELECTIVE)
# ============================================================

print("\nGenerating degree-3 polynomial features on Top 20...")
print("Expected new features: 20 choose 3 = 1,140")

poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)
X_degree3 = poly.fit_transform(X_top20)

print(f"Total features after expansion: {X_degree3.shape[1]}")

# Identify degree-3 features using powers array
# powers_ shows the exponent of each original feature in each polynomial term
powers = poly.powers_

# A degree-3 interaction has sum of powers = 3
# interaction_only=True means each power is 0 or 1, so sum=3 means 3 different features
degree3_indices = []
for idx, power_vector in enumerate(powers):
    power_sum = np.sum(power_vector)
    num_nonzero = np.count_nonzero(power_vector)
    
    # Degree-3 = exactly 3 features involved (interaction_only means each has power 1)
    if power_sum == 3 and num_nonzero == 3:
        degree3_indices.append(idx)

X_degree3_only = X_degree3[:, degree3_indices]
feature_names_degree3 = poly.get_feature_names_out(top20_names)
degree3_only_names = [feature_names_degree3[i] for i in degree3_indices]

print(f"\nPure degree-3 features extracted: {X_degree3_only.shape[1]}")

if X_degree3_only.shape[1] == 0:
    print("\nERROR: No degree-3 features generated!")
    print("This shouldn't happen with 20 features. Check PolynomialFeatures setup.")
    sys.exit(1)

# ============================================================
# RANK DEGREE-3 FEATURES BY MI
# ============================================================

print("\nRanking degree-3 features by Mutual Information...")
mi_scores_degree3 = mutual_info_classif(X_degree3_only, y, random_state=42)

# Combine with original Top 100 scores for comparison
combined_features = np.hstack([X_top100, X_degree3_only])
combined_names = selected_names + degree3_only_names

print(f"Combined feature set: {combined_features.shape[1]} features")

# Rank all combined
print("\nRanking combined features (Top 100 + Degree-3)...")
mi_scores_combined = mutual_info_classif(combined_features, y, random_state=42)

# Select Top 120 from combined set (expand from 100 to 120)
top120_indices = np.argsort(mi_scores_combined)[-120:][::-1]
top120_names = [combined_names[i] for i in top120_indices]
top120_mi_scores = mi_scores_combined[top120_indices]

print("\nTop 20 Features in New Combined Set:")
for rank, (name, score) in enumerate(zip(top120_names[:20], top120_mi_scores[:20]), 1):
    is_degree3 = " [DEGREE-3]" if name in degree3_only_names else ""
    print(f"  {rank:2d}. {name[:60]:60s} | MI={score:.4f}{is_degree3}")

# Count how many degree-3 features made it to Top 120
degree3_in_top120 = sum(1 for name in top120_names if name in degree3_only_names)
print(f"\nDegree-3 features in Top 120: {degree3_in_top120} / 120 ({degree3_in_top120/120*100:.1f}%)")

# ============================================================
# SAVE RESULTS
# ============================================================

# Save Top 120 feature set
X_top120 = combined_features[:, top120_indices]
df_top120 = pd.DataFrame(X_top120, columns=top120_names)
df_top120['label'] = y

output_path = RESULTS_DIR / "degree3_enhanced_features.csv"
df_top120.to_csv(output_path, index=False)
print(f"\nSaved enhanced feature set to: {output_path}")

# Save feature metadata
enhanced_metadata = {
    'total_features': 120,
    'degree3_count': degree3_in_top120,
    'top120_names': top120_names,
    'top120_mi_scores': top120_mi_scores.tolist(),
    'top20_base_features': top20_names,
}

metadata_path = RESULTS_DIR / "degree3_feature_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(enhanced_metadata, f, indent=2)
print(f"Saved metadata to: {metadata_path}")

# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. MI Score Distribution
axes[0].hist(mi_scores_combined, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].axvline(top120_mi_scores[-1], color='red', linestyle='--', lw=2, label=f'Top 120 cutoff ({top120_mi_scores[-1]:.4f})')
axes[0].set_xlabel('Mutual Information Score', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('MI Score Distribution (Top 100 + Degree-3)', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. Top 30 Features Comparison
top30_indices_plot = list(range(30))
colors = ['orange' if name in degree3_only_names else 'steelblue' for name in top120_names[:30]]

axes[1].barh(top30_indices_plot, top120_mi_scores[:30], color=colors, edgecolor='black')
axes[1].set_yticks(top30_indices_plot)
axes[1].set_yticklabels([name[:40] + '...' if len(name) > 40 else name for name in top120_names[:30]], fontsize=8)
axes[1].set_xlabel('Mutual Information Score', fontsize=12)
axes[1].set_title('Top 30 Features (Blue=Degree-2, Orange=Degree-3)', fontsize=14, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
fig_path = RESULTS_DIR / "degree3_feature_importance.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Saved visualization to: {fig_path}")

print("\n✅ Phase 1.3 Complete!")
print("\nNext Steps:")
print("  1. Train ExSTraCS on Top 120 features (degree3_enhanced_features.csv)")
print("  2. Compare performance: Top 100 (baseline) vs Top 120 (with degree-3)")
print("  3. Run phase1_validate_improvements.py for final validation")
