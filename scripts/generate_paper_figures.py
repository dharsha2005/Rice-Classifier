"""
Generate and curate publication-quality figures for the research paper.
Creates high-resolution figures in paper/figures/ based on verified experimental data.
"""

from pathlib import Path
import shutil
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path("c:/Rice classifier final project")
RESULTS_DIR = PROJECT_ROOT / "results"
HYBRID_DIR = RESULTS_DIR / "models" / "hybrid_xgboost"
PREPROC_DIR = RESULTS_DIR / "preprocessing"
PAPER_FIG_DIR = PROJECT_ROOT / "paper" / "figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set global publication styling
plt.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

print("1. Copying existing high-resolution figures...")
# Direct copies of verified generated figures
src_dst_pairs = [
    (HYBRID_DIR / "confusion_matrix.png", PAPER_FIG_DIR / "fig4_confusion_matrix_raw.png"),
    (HYBRID_DIR / "normalized_confusion_matrix.png", PAPER_FIG_DIR / "fig4_confusion_matrix_normalized.png"),
    (HYBRID_DIR / "shap_summary.png", PAPER_FIG_DIR / "fig7b_shap_beeswarm_summary.png"),
    (HYBRID_DIR / "shap_top20_features.png", PAPER_FIG_DIR / "fig7a_shap_top20_bar.png"),
    (HYBRID_DIR / "shap_feature_group_importance.png", PAPER_FIG_DIR / "fig8_shap_group_contribution.png"),
    (HYBRID_DIR / "hybrid_feature_group_importance.png", PAPER_FIG_DIR / "fig9a_gain_group_importance.png"),
    (HYBRID_DIR / "per_class_metrics.png", PAPER_FIG_DIR / "fig10a_per_class_metrics_hybrid.png"),
    (PREPROC_DIR / "preprocessing_summary_grid.png", PAPER_FIG_DIR / "fig3_preprocessing_grid.png"),
]

for src, dst in src_dst_pairs:
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  Copied: {src.name} -> {dst.name}")
    else:
        print(f"  WARNING: Source not found: {src}")

print("2. Generating Fig 1: Overall Architecture Framework Diagram...")
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.axis("off")

# Draw flowchart blocks
boxes = [
    {"text": "Input Rice Image\n(RGB, Variable Res)", "xy": (0.08, 0.78), "w": 0.15, "h": 0.14, "color": "#e1f5fe", "border": "#0288d1"},
    {"text": "Preprocessing & Denoising\n• Gaussian Blur (5x5)\n• Otsu Thresholding\n• Morphological Close/Open", "xy": (0.30, 0.78), "w": 0.18, "h": 0.14, "color": "#e8f5e9", "border": "#388e3c"},
    {"text": "Grain Isolation & Standard.\n• Contour Extraction\n• Mask & Bounding Box\n• Isotropic 224x224x3 Crop", "xy": (0.55, 0.78), "w": 0.18, "h": 0.14, "color": "#e8f5e9", "border": "#388e3c"},
    
    # Handcrafted branch
    {"text": "Handcrafted Feature Pipeline\n• Shape / Morphology (14-D)\n• GLCM Texture (12-D)\n• Multichannel Colour (36-D)\nTotal: 62 Dimensions", "xy": (0.30, 0.40), "w": 0.22, "h": 0.18, "color": "#fff3e0", "border": "#f57c00"},
    # Deep branch
    {"text": "Deep Feature Extractor\n• Pretrained EfficientNet-B0\n• ImageNet Normalization\n• Pooled Global Representation\nTotal: 1280 Dimensions", "xy": (0.58, 0.40), "w": 0.22, "h": 0.18, "color": "#ede7f6", "border": "#512da8"},
    
    # Fusion & scaling
    {"text": "Feature Concatenation & Scaling\n• Fused Vector: 1342 Dimensions\n• StandardScaler (Train-Fit Only)", "xy": (0.44, 0.15), "w": 0.22, "h": 0.14, "color": "#fce4ec", "border": "#c2185b"},
    
    # Classifier & Output
    {"text": "XGBoost Classifier\n(n_est=100, depth=4, lr=0.1)", "xy": (0.74, 0.15), "w": 0.15, "h": 0.14, "color": "#e0f2f1", "border": "#00796b"},
    {"text": "Output & Explainability\n• 8-Class Prediction\n• Class Probabilities\n• SHAP Explanations\n• Quality Alert / Review Queue", "xy": (0.91, 0.50), "w": 0.16, "h": 0.35, "color": "#fffde7", "border": "#fbc02d"},
]

from matplotlib.patches import FancyBboxPatch

for b in boxes:
    x, y = b["xy"]
    w, h = b["w"], b["h"]
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.02",
                          facecolor=b["color"], edgecolor=b["border"], linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, b["text"], ha="center", va="center", fontsize=8.5, fontweight="bold",
            color="#212121", zorder=3, multialignment="center")

# Draw connection arrows
arrows = [
    ((0.155, 0.78), (0.21, 0.78)),
    ((0.39, 0.78), (0.46, 0.78)),
    ((0.64, 0.78), (0.41, 0.49)), # To handcrafted
    ((0.64, 0.78), (0.69, 0.49)), # To deep
    ((0.41, 0.31), (0.50, 0.22)), # Handcrafted to fusion
    ((0.69, 0.31), (0.60, 0.22)), # Deep to fusion
    ((0.55, 0.15), (0.665, 0.15)), # Fusion to XGBoost
    ((0.815, 0.15), (0.91, 0.325)), # XGBoost to output
]

for start, end in arrows:
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color="#37474f", lw=1.5, shrinkA=2, shrinkB=2), zorder=4)

ax.set_xlim(0, 1.0)
ax.set_ylim(0.05, 0.95)
plt.title("Fig. 1. End-to-End Architectural Pipeline of the Proposed Explainable Hybrid Feature-Fusion Framework", fontsize=11, fontweight="bold", pad=12)
fig.savefig(PAPER_FIG_DIR / "fig1_overall_framework.png")
plt.close(fig)
print("  Generated: fig1_overall_framework.png")

print("3. Generating Fig 2: Preprocessing 4-Stage Workflow Diagram...")
# Create a figure showing the 4 steps of preprocessing using actual samples
sample_comp_path = PREPROC_DIR / "visual_comparisons" / "sample_01_class_0_0_NOR_Grainset_rice_2020-11-10-18-37-05_3_p600s.png"
if sample_comp_path.exists():
    shutil.copy2(sample_comp_path, PAPER_FIG_DIR / "fig2_preprocessing_stages.png")
    print(f"  Copied: {sample_comp_path.name} -> fig2_preprocessing_stages.png")

print("4. Generating Fig 5: Model Comparison Chart...")
models_df = pd.read_csv(PROJECT_ROOT / "results" / "models" / "model_comparison_final.csv")
fig, ax = plt.subplots(figsize=(8, 4.5))

x = np.arange(len(models_df))
width = 0.25

rects1 = ax.bar(x - width, models_df["Accuracy"] * 100, width, label="Test Accuracy (%)", color="#1976d2")
rects2 = ax.bar(x, models_df["Macro F1"] * 100, width, label="Macro F1 (%)", color="#388e3c")
rects3 = ax.bar(x + width, models_df["Weighted F1"] * 100, width, label="Weighted F1 (%)", color="#f57c00")

ax.set_ylabel("Metric Score (%)", fontweight="bold")
ax.set_title("Fig. 5. Benchmark Comparison: Handcrafted SVM vs. EfficientNet-B0 XGBoost vs. Proposed Hybrid XGBoost", fontweight="bold", pad=10)
ax.set_xticks(x)
ax.set_xticklabels(["SVM\n(62 Handcrafted)", "XGBoost\n(1280 EfficientNet-B0)", "Proposed Hybrid XGBoost\n(1342 Fused)"], fontweight="bold")
ax.set_ylim(75, 100)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(loc="lower right", framealpha=0.95)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.savefig(PAPER_FIG_DIR / "fig5_model_comparison_bars.png")
plt.close(fig)
print("  Generated: fig5_model_comparison_bars.png")

print("5. Generating Fig 6: Unified Ablation Study Chart...")
ablation_df = pd.read_csv(HYBRID_DIR / "ablation_results.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

configs = [c.split(". ")[-1] for c in ablation_df["configuration"]]
val_acc = ablation_df["validation_accuracy"] * 100
test_acc = ablation_df["test_accuracy"] * 100
val_f1 = ablation_df["validation_macro_f1"] * 100
test_f1 = ablation_df["test_macro_f1"] * 100

y = np.arange(len(configs))

ax1.barh(y - 0.18, val_acc, height=0.35, label="Validation Accuracy", color="#64b5f6")
ax1.barh(y + 0.18, test_acc, height=0.35, label="Test Accuracy", color="#1565c0")
ax1.set_yticks(y)
ax1.set_yticklabels(configs, fontweight="bold")
ax1.set_xlabel("Accuracy (%)", fontweight="bold")
ax1.set_xlim(70, 100)
ax1.set_title("Classification Accuracy Across Feature Subsets", fontweight="bold")
ax1.grid(axis="x", linestyle="--", alpha=0.5)
ax1.legend(loc="lower right")
ax1.invert_yaxis()

ax2.barh(y - 0.18, val_f1, height=0.35, label="Validation Macro F1", color="#81c784")
ax2.barh(y + 0.18, test_f1, height=0.35, label="Test Macro F1", color="#2e7d32")
ax2.set_yticks(y)
ax2.set_yticklabels([])
ax2.set_xlabel("Macro F1 (%)", fontweight="bold")
ax2.set_xlim(45, 95)
ax2.set_title("Macro F1 Across Feature Subsets", fontweight="bold")
ax2.grid(axis="x", linestyle="--", alpha=0.5)
ax2.legend(loc="lower right")
ax2.invert_yaxis()

fig.suptitle("Fig. 6. Feature Ablation Study: Validation and Test Performance Across 7 Feature Configurations", fontsize=12, fontweight="bold")
fig.savefig(PAPER_FIG_DIR / "fig6_ablation_comparison.png")
plt.close(fig)
print("  Generated: fig6_ablation_comparison.png")

print("6. Generating Fig 9: Dual Comparison of XGBoost Gain vs. SHAP Group Contribution...")
gain_df = pd.read_csv(HYBRID_DIR / "hybrid_feature_group_importance.csv")
shap_df = pd.read_csv(HYBRID_DIR / "shap_feature_group_importance.csv")

# Standardize group names
gain_groups = {"EfficientNet Deep": "EfficientNet-B0 (1280)", "Shape": "Shape / Morph (14)", "Colour": "Colour (36)", "GLCM Texture": "GLCM Texture (12)"}
shap_groups = {"EfficientNet-B0": "EfficientNet-B0 (1280)", "Shape/Morphological": "Shape / Morph (14)", "Colour": "Colour (36)", "GLCM Texture": "GLCM Texture (12)"}

gain_dict = {gain_groups[r["feature_group"]]: r["relative_importance_percent"] for _, r in gain_df.iterrows()}
shap_dict = {shap_groups[r["feature_group"]]: r["shap_contribution_percent"] for _, r in shap_df.iterrows()}

ordered_labels = ["EfficientNet-B0 (1280)", "Shape / Morph (14)", "Colour (36)", "GLCM Texture (12)"]
gain_vals = [gain_dict[l] for l in ordered_labels]
shap_vals = [shap_dict[l] for l in ordered_labels]

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(ordered_labels))
width = 0.35

r1 = ax.bar(x - width/2, gain_vals, width, label="XGBoost Tree-Gain Importance (%)", color="#7b1fa2")
r2 = ax.bar(x + width/2, shap_vals, width, label="SHAP Attribution Contribution (%)", color="#0288d1")

ax.set_ylabel("Group Share (%)", fontweight="bold")
ax.set_title("Fig. 9. Disentangling Model Explanations: XGBoost Gain-Based Importance vs. SHAP Attributions", fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(ordered_labels, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(loc="upper right", framealpha=0.95)

for rect in r1:
    h = rect.get_height()
    ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
for rect in r2:
    h = rect.get_height()
    ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_ylim(0, 100)
fig.savefig(PAPER_FIG_DIR / "fig9_gain_vs_shap_comparison.png")
plt.close(fig)
print("  Generated: fig9_gain_vs_shap_comparison.png")

print("7. Generating Fig 10: Per-Class F1 Score Radar / Multi-Bar Comparison...")
svm_report = PROJECT_ROOT / "results" / "models" / "svm" / "svm_baseline_report.txt"
enet_report = PROJECT_ROOT / "results" / "models" / "efficientnet_xgboost" / "efficientnet_xgboost_report.txt"
hybrid_df = pd.read_csv(HYBRID_DIR / "hybrid_per_class_metrics.csv")

# Extract SVM and EfficientNet per-class F1 scores
# From verified reports:
svm_f1 = [0.9579, 0.8291, 0.6704, 0.7492, 0.8276, 0.9195, 0.8662, 0.9750]
enet_f1 = [0.9533, 0.7987, 0.7440, 0.7172, 0.7566, 0.9215, 0.8481, 0.9800]
hybrid_f1 = list(hybrid_df["f1_score"])
class_labels = list(hybrid_df["class_name"])

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(class_labels))
width = 0.26

r1 = ax.bar(x - width, [v * 100 for v in svm_f1], width, label="SVM (62 Handcrafted)", color="#78909c")
r2 = ax.bar(x, [v * 100 for v in enet_f1], width, label="EfficientNet-B0 + XGBoost (1280)", color="#ab47bc")
r3 = ax.bar(x + width, [v * 100 for v in hybrid_f1], width, label="Proposed Hybrid XGBoost (1342)", color="#00897b")

ax.set_ylabel("F1-Score (%)", fontweight="bold")
ax.set_title("Fig. 10. Per-Class F1-Score Across All Eight Grain Categories for the Evaluated Frameworks", fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(class_labels, fontweight="bold")
ax.set_ylim(55, 105)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(loc="lower right", framealpha=0.95)

for rect in r3:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#004d40")

fig.savefig(PAPER_FIG_DIR / "fig10_per_class_f1_comparison.png")
plt.close(fig)
print("  Generated: fig10_per_class_f1_comparison.png")

print("All figures successfully created in paper/figures/!")
