# List of Figures for Research Paper

This document catalogues all publication figures prepared for the research paper:  
*"An Explainable Hybrid Feature Fusion Framework for Rice Quality and Defect Classification Using EfficientNet-B0 and XGBoost"*

All image files reside in `paper/figures/` at 300 DPI publication resolution.

---

### Fig. 1. End-to-End Architectural Pipeline of the Proposed Explainable Hybrid Feature-Fusion Framework
- **File Path:** `paper/figures/fig1_overall_framework.png`
- **Section Referenced:** Section IV (Proposed Methodology)
- **Description:** Complete architectural flowchart tracing the journey of an input rice image through RGB/Grayscale/HSV/LAB conversion, Gaussian blur denoising, Otsu thresholding, morphological closing/opening, contour isolation, isotropic 224×224 standardization, parallel handcrafted extraction (62-D) and EfficientNet-B0 deep feature extraction (1,280-D), concatenation into a 1,342-D hybrid vector, training-fitted StandardScaler normalization, XGBoost multiclass inference, and SHAP explainability.
- **Source Data:** System architecture implementation across `src/preprocessing/preprocess.py`, `src/features/`, and `src/models/train_hybrid_xgboost.py`.

---

### Fig. 2. Preprocessing and Grain Segmentation Workflow Across Representative Stages
- **File Path:** `paper/figures/fig2_preprocessing_stages.png`
- **Section Referenced:** Section IV-A (Image Preprocessing and Segmentation)
- **Description:** 4-panel visual demonstration of the preprocessing lifecycle for a representative rice sample:
  1. *Original Image:* Unstandardized raw RGB grain image with camera sensor background.
  2. *Grayscale & Denoised:* Intensity representation smoothed using calibrated Gaussian filter ((5, 5), $\sigma=0$).
  3. *Otsu Binary Mask:* Segmented grain silhouette post morphological closing and opening.
  4. *Segmented & Standardized Crop:* Aspect-ratio preserved, background-subtracted grain centered on a 224×224 black canvas.
- **Source Data:** `results/preprocessing/visual_comparisons/sample_01_class_0_0_NOR_Grainset_rice_2020-11-10-18-37-05_3_p600s.png`.

---

### Fig. 3. Multi-Class Preprocessing and Segmentation Validation Grid
- **File Path:** `paper/figures/fig3_preprocessing_grid.png`
- **Section Referenced:** Section IV-A & V (Preprocessing & Experimental Setup)
- **Description:** Multi-sample composite grid showcasing successful segmentation, boundary preservation, and standardized 224×224 centering across diverse grain categories, validating robustness against morphological irregularity, chalkiness, and aspect-ratio variations.
- **Source Data:** `results/preprocessing/preprocessing_summary_grid.png`.

---

### Fig. 4. Confusion Matrix Heatmaps on Untouched Test Set (3,100 Samples)
- **File Paths:**
  - Raw Counts: `paper/figures/fig4_confusion_matrix_raw.png`
  - Normalized: `paper/figures/fig4_confusion_matrix_normalized.png`
- **Section Referenced:** Section VI & IX (Results & Error Analysis)
- **Description:** 8×8 confusion matrix heatmaps showing exact true vs. predicted counts and normalized recall proportions for the proposed Hybrid XGBoost model on the 3,100 test images. Highlights the strong diagonal dominance (2,855 / 3,100 correct = 92.10%), with minor off-diagonal confusion between `2_SD` and `1_F&S`.
- **Source Data:** `results/models/hybrid_xgboost/hybrid_confusion_matrix.csv`.

---

### Fig. 5. Benchmark Performance Comparison Across Architectural Paradigms
- **File Path:** `paper/figures/fig5_model_comparison_bars.png`
- **Section Referenced:** Section VI (Results and Discussion)
- **Description:** Grouped bar chart comparing Test Accuracy (%), Macro F1 (%), and Weighted F1 (%) across:
  1. Conventional Handcrafted Baseline: Support Vector Machine (62 features)
  2. Deep Feature Baseline: EfficientNet-B0 + XGBoost (1,280 features)
  3. Proposed Hybrid Framework: Handcrafted + EfficientNet-B0 + XGBoost (1,342 features)
  Demonstrates that the hybrid model surpasses both single-modality baselines across all three summary metrics.
- **Source Data:** `results/models/model_comparison_final.csv`.

---

### Fig. 6. Feature Ablation Study Across 7 Methodological Configurations
- **File Path:** `paper/figures/fig6_ablation_comparison.png`
- **Section Referenced:** Section VII (Ablation Study)
- **Description:** Dual-panel horizontal bar chart illustrating Validation and Test Accuracy (left panel) alongside Validation and Test Macro F1 (right panel) across all 7 ablation configurations: Shape only (14-D), GLCM Texture only (12-D), Colour only (36-D), All Handcrafted (62-D), EfficientNet-B0 only (1280-D), Retrained Hybrid (1342-D), and Saved Final Hybrid (1342-D).
- **Source Data:** `results/models/hybrid_xgboost/ablation_results.csv`.

---

### Fig. 7. Top 20 Global Features Ranked by Mean Absolute SHAP Attribution
- **File Paths:**
  - Bar Ranking: `paper/figures/fig7a_shap_top20_bar.png`
  - Beeswarm Summary: `paper/figures/fig7b_shap_beeswarm_summary.png`
- **Section Referenced:** Section VIII (SHAP Explainability)
- **Description:**
  - *Fig. 7(a):* Horizontal bar chart ranking the top 20 individual features by mean absolute SHAP value across 300 test explanation samples and 8 output classes.
  - *Fig. 7(b):* Beeswarm scatter distribution displaying how high vs. low feature values push individual class probabilities, showing that physical descriptors (`shape_height`, `color_lab_b_mean`, `shape_major_axis_length`, `shape_eccentricity`) dominate the top tier alongside deep embeddings.
- **Source Data:** `results/models/hybrid_xgboost/shap_top20_features.csv` and `shap_summary.png`.

---

### Fig. 8. SHAP Feature Group Contribution Breakdown
- **File Path:** `paper/figures/fig8_shap_group_contribution.png`
- **Section Referenced:** Section VIII (SHAP Explainability)
- **Description:** Vertical bar chart detailing the proportional contribution of the four feature groups to the total mean-absolute SHAP attribution: EfficientNet-B0 (68.59%), Shape/Morphology (13.52%), Multichannel Colour (12.86%), and GLCM Texture (5.02%).
- **Source Data:** `results/models/hybrid_xgboost/shap_feature_group_importance.csv`.

---

### Fig. 9. Disentangling Model Explanations: XGBoost Gain-Based Importance vs. SHAP Attribution
- **File Path:** `paper/figures/fig9_gain_vs_shap_comparison.png`
- **Section Referenced:** Section VIII (SHAP Explainability)
- **Description:** Side-by-side grouped bar chart illustrating the critical methodological distinction between internal tree-splitting gain importance (where deep features dominate at 85.50%) and external SHAP marginal attribution (where handcrafted features capture 31.41% of the explanatory weight).
- **Source Data:** `results/models/hybrid_xgboost/hybrid_feature_group_importance.csv` and `shap_feature_group_importance.csv`.

---

### Fig. 10. Per-Class F1-Score Comparison Across Evaluated Architectures
- **File Path:** `paper/figures/fig10_per_class_f1_comparison.png`
- **Section Referenced:** Section VI & IX (Results & Error Analysis)
- **Description:** Multi-bar comparison of F1-scores across all eight grain classes (`0_NOR` through `7_IM`) comparing the Handcrafted SVM, Deep XGBoost, and Proposed Hybrid XGBoost models, highlighting how hybrid fusion lifts challenging minority defect classes.
- **Source Data:** `results/models/hybrid_xgboost/hybrid_per_class_metrics.csv`, `results/models/svm/svm_baseline_report.txt`, and `results/models/efficientnet_xgboost/efficientnet_xgboost_report.txt`.
