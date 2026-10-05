# Verified Experimental Facts & Evidence Audit

This document catalogues every critical empirical value, parameter, metric, and artifact from the **Rice Quality and Defect Assessment** project. Every entry references the concrete source file and location within the repository.

---

## 1. Project Identity & Attribution

- **FACT:** Research Paper Title  
  **SOURCE FILE:** User Specification & Repository Root Documentation  
  **LOCATION:** `PROJECT_FINAL_DOCUMENTATION.md`  
  **VALUE:** "An Explainable Hybrid Feature Fusion Framework for Rice Quality and Defect Classification Using EfficientNet-B0 and XGBoost"

- **FACT:** Authors and Academic Affiliation  
  **SOURCE FILE:** User Specification & Project Metadata  
  **LOCATION:** `PROJECT_FINAL_DOCUMENTATION.md`  
  **VALUE:** Dharshan B¹, Dinesh G L¹ (Mentored by Ms. Sripriya S), Department of Information Technology, Kongu Engineering College, Perundurai, Tamil Nadu, India

---

## 2. Dataset Distribution & Integrity

- **FACT:** Total Dataset Image Count  
  **SOURCE FILE:** `reports/dataset_report.txt`, `reports/dataset_summary.csv`  
  **LOCATION:** `dataset_report.txt:22`, `dataset_summary.csv`  
  **VALUE:** 30,962 images

- **FACT:** Training Split Image Count  
  **SOURCE FILE:** `reports/dataset_report.txt`, `results/models/hybrid_xgboost/hybrid_reproducibility.json`  
  **LOCATION:** `dataset_report.txt:19`, `hybrid_reproducibility.json:4`  
  **VALUE:** 24,767 images (79.99%)

- **FACT:** Validation Split Image Count  
  **SOURCE FILE:** `reports/dataset_report.txt`, `results/models/hybrid_xgboost/hybrid_reproducibility.json`  
  **LOCATION:** `dataset_report.txt:20`, `hybrid_reproducibility.json:5`  
  **VALUE:** 3,095 images (10.00%)

- **FACT:** Test Split Image Count  
  **SOURCE FILE:** `reports/dataset_report.txt`, `results/models/hybrid_xgboost/hybrid_reproducibility.json`  
  **LOCATION:** `dataset_report.txt:21`, `hybrid_reproducibility.json:6`  
  **VALUE:** 3,100 images (10.01%)

- **FACT:** Balanced Training Subset Count  
  **SOURCE FILE:** `reports/dataset_report.txt`, `datasets/rice_train_bal.txt`  
  **LOCATION:** `dataset_report.txt:23, 75-86`  
  **VALUE:** 9,528 images (Sub-sample of training split; not used for final model training)

- **FACT:** Number of Grain Classes  
  **SOURCE FILE:** `reports/dataset_report.txt`, `src/models/train_hybrid_xgboost.py`  
  **LOCATION:** `dataset_report.txt:18`, `train_hybrid_xgboost.py:50-59`  
  **VALUE:** 8 classes

- **FACT:** Raw Class Identifiers  
  **SOURCE FILE:** `reports/dataset_report.txt`, `datasets/*.txt`  
  **LOCATION:** `dataset_report.txt:25-35`  
  **VALUE:** `0_NOR`, `1_F&S`, `2_SD`, `3_MY`, `4_AP`, `5_BN`, `6_UN`, `7_IM`  
  *(Note: Semantic expansions are not documented in dataset files and are explicitly preserved as raw labels)*

- **FACT:** Class Distribution in Training Split  
  **SOURCE FILE:** `reports/dataset_summary.csv`, `reports/dataset_report.txt`  
  **LOCATION:** `dataset_summary.csv:2-9`  
  **VALUE:** `0_NOR`: 15,980 (64.52%), `1_F&S`: 1,191 (4.81%), `2_SD`: 1,194 (4.82%), `3_MY`: 1,223 (4.94%), `4_AP`: 1,208 (4.88%), `5_BN`: 1,198 (4.84%), `6_UN`: 1,185 (4.78%), `7_IM`: 1,588 (6.41%)

- **FACT:** Class Distribution in Validation Split  
  **SOURCE FILE:** `reports/dataset_summary.csv`, `reports/dataset_report.txt`  
  **LOCATION:** `dataset_summary.csv:10-17`  
  **VALUE:** `0_NOR`: 2,020 (65.27%), `1_F&S`: 145 (4.68%), `2_SD`: 156 (5.04%), `3_MY`: 127 (4.10%), `4_AP`: 142 (4.59%), `5_BN`: 143 (4.62%), `6_UN`: 150 (4.85%), `7_IM`: 212 (6.85%)

- **FACT:** Class Distribution in Test Split  
  **SOURCE FILE:** `reports/dataset_summary.csv`, `reports/dataset_report.txt`  
  **LOCATION:** `dataset_summary.csv:18-25`  
  **VALUE:** `0_NOR`: 2,000 (64.52%), `1_F&S`: 150 (4.84%), `2_SD`: 150 (4.84%), `3_MY`: 150 (4.84%), `4_AP`: 150 (4.84%), `5_BN`: 150 (4.84%), `6_UN`: 150 (4.84%), `7_IM`: 200 (6.45%)

- **FACT:** Split Filename Overlap / Data Leakage Check  
  **SOURCE FILE:** `reports/dataset_report.txt`  
  **LOCATION:** `dataset_report.txt:132-138`  
  **VALUE:** 0 overlapping filenames across train, val, and test splits (Strict leakage-free split)

- **FACT:** Corrupted or Missing Images  
  **SOURCE FILE:** `reports/dataset_report.txt`  
  **LOCATION:** `dataset_report.txt:87-94`  
  **VALUE:** 0 missing files, 0 corrupt images across 40,490 checked manifest entries

---

## 3. Preprocessing & Grain Segmentation Pipeline

- **FACT:** Noise Reduction Filter  
  **SOURCE FILE:** `src/preprocessing/preprocess.py`  
  **LOCATION:** `preprocess.py:175-180`  
  **VALUE:** Gaussian Blur, Kernel Size = (5, 5), Sigma = 0

- **FACT:** Foreground Thresholding Algorithm  
  **SOURCE FILE:** `src/preprocessing/preprocess.py`  
  **LOCATION:** `preprocess.py:195-198`  
  **VALUE:** Automatic Otsu Thresholding (`cv2.THRESH_BINARY + cv2.THRESH_OTSU`)

- **FACT:** Morphological Structuring Element  
  **SOURCE FILE:** `src/preprocessing/preprocess.py`  
  **LOCATION:** `preprocess.py:133, 200-202`  
  **VALUE:** Elliptical Structuring Element (`cv2.MORPH_ELLIPSE`), Kernel Size = (5, 5); Morphological Close (bridge internal voids) followed by Morphological Open (eliminate stray noise)

- **FACT:** Primary Grain Isolation Strategy  
  **SOURCE FILE:** `src/preprocessing/preprocess.py`  
  **LOCATION:** `preprocess.py:204-219`  
  **VALUE:** External contour extraction (`cv2.RETR_EXTERNAL`), largest contour filtered by area (minimum grain area threshold = 500 px)

- **FACT:** Bounding Box Margin  
  **SOURCE FILE:** `src/preprocessing/preprocess.py`  
  **LOCATION:** `preprocess.py:136, 274-282`  
  **VALUE:** Margin ratio = 0.05 (5% margin expansion around bounding box)

- **FACT:** Standardized Output Image Dimensions  
  **SOURCE FILE:** `src/preprocessing/preprocess.py`, `results/preprocessing/preprocessing_report.txt`  
  **LOCATION:** `preprocess.py:58, 286-298`, `preprocessing_report.txt:17`  
  **VALUE:** 224 × 224 pixels, 3 channels (RGB/BGR), isotropic aspect-ratio preserved scaling (fill scale = 0.90) with centered background padding

- **FACT:** Mean Preprocessing Latency  
  **SOURCE FILE:** `results/preprocessing/preprocessing_report.txt`  
  **LOCATION:** `preprocessing_report.txt:11`  
  **VALUE:** 5.44 ms per image

- **FACT:** Sample Original Image Dimensions (Observed)  
  **SOURCE FILE:** `results/preprocessing/preprocessing_report.txt`  
  **LOCATION:** `preprocessing_report.txt:14-16`  
  **VALUE:** Minimum: 206 × 135 px, Maximum: 328 × 301 px, Mean: 261.5 × 222.2 px

---

## 4. Feature Extraction & Engineering

- **FACT:** Number of Handcrafted Morphological/Shape Features  
  **SOURCE FILE:** `src/features/handcrafted_features.py`  
  **LOCATION:** `handcrafted_features.py:31-125`  
  **VALUE:** 14 features (`shape_area`, `shape_perimeter`, `shape_width`, `shape_height`, `shape_aspect_ratio`, `shape_extent`, `shape_solidity`, `shape_circularity`, `shape_eccentricity`, `shape_major_axis_length`, `shape_minor_axis_length`, `shape_equivalent_diameter`, `shape_convex_hull_area`, `shape_bbox_area`)

- **FACT:** Number of Handcrafted GLCM Texture Features  
  **SOURCE FILE:** `src/features/handcrafted_features.py`  
  **LOCATION:** `handcrafted_features.py:131-284`  
  **VALUE:** 12 features (6 descriptors × 2 statistics [mean, std]: contrast, dissimilarity, homogeneity, energy, correlation, angular second moment [ASM]; computed over distances 1 and 2, angles 0°, 45°, 90°, 135°, quantized to 32 intensity levels exclusively on grain foreground)

- **FACT:** Number of Handcrafted Colour Features  
  **SOURCE FILE:** `src/features/handcrafted_features.py`  
  **LOCATION:** `handcrafted_features.py:290-330`  
  **VALUE:** 36 features (9 channels across RGB, HSV, LAB spaces × 4 statistics [mean, std, min, max] computed exclusively on grain foreground)

- **FACT:** Total Handcrafted Feature Dimension  
  **SOURCE FILE:** `src/features/handcrafted_features.py`, `results/features/`  
  **LOCATION:** `handcrafted_features.py:348`  
  **VALUE:** 62 dimensions (14 + 12 + 36 = 62)

- **FACT:** Deep Learning Feature Backbone  
  **SOURCE FILE:** `src/features/efficientnet_features.py`  
  **LOCATION:** `efficientnet_features.py:38-46`  
  **VALUE:** Pretrained EfficientNet-B0 (`torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)`)

- **FACT:** Deep Feature Extraction Head Configuration  
  **SOURCE FILE:** `src/features/efficientnet_features.py`  
  **LOCATION:** `efficientnet_features.py:41-44`  
  **VALUE:** Classifier replaced with `torch.nn.Identity()`, evaluation mode (`model.eval()`), inference under `torch.inference_mode()`, feature dimension = 1280

- **FACT:** Total Fused Hybrid Feature Dimension  
  **SOURCE FILE:** `src/models/train_hybrid_xgboost.py`, `results/models/hybrid_xgboost/hybrid_reproducibility.json`  
  **LOCATION:** `train_hybrid_xgboost.py:8`, `hybrid_reproducibility.json:8`  
  **VALUE:** 1,342 dimensions (62 handcrafted + 1280 EfficientNet-B0 = 1342)

- **FACT:** Feature Scaler  
  **SOURCE FILE:** `src/models/train_hybrid_xgboost.py`, `results/models/hybrid_xgboost/best_parameters.json`  
  **LOCATION:** `train_hybrid_xgboost.py:284-290`, `best_parameters.json:8`  
  **VALUE:** `StandardScaler` (Fit strictly on 24,767 training samples; applied to validation and test samples)

---

## 5. Model Hyperparameters & Optimization

- **FACT:** Final Hybrid XGBoost Hyperparameters  
  **SOURCE FILE:** `results/models/hybrid_xgboost/best_parameters.json`, `hybrid_best_params.json`  
  **LOCATION:** `best_parameters.json:1-10`  
  **VALUE:**  
  - `n_estimators`: 100  
  - `max_depth`: 4  
  - `learning_rate`: 0.1  
  - `subsample`: 0.8  
  - `colsample_bytree`: 0.8  
  - `min_child_weight`: 1  
  - `objective`: `multi:softprob`  
  - `tree_method`: `hist`  
  - `eval_metric`: `mlogloss`  
  - `scaler_name`: `StandardScaler`  
  - `balanced_sample_weight`: False  
  - `random_state`: 42  

- **FACT:** Final Baseline SVM Hyperparameters  
  **SOURCE FILE:** `results/models/svm/svm_baseline_report.txt`  
  **LOCATION:** `svm_baseline_report.txt:31`  
  **VALUE:** Kernel = RBF, C = 50.0, gamma = 'scale', class_weight = None, Scaler = StandardScaler

- **FACT:** Final EfficientNet-B0 + XGBoost Hyperparameters  
  **SOURCE FILE:** `results/models/efficientnet_xgboost/best_parameters.json`, `efficientnet_xgboost_report.txt`  
  **LOCATION:** `efficientnet_xgboost_report.txt:19`  
  **VALUE:** `n_estimators`: 100, `max_depth`: 4, `learning_rate`: 0.1, `subsample`: 0.8, `colsample_bytree`: 0.8, `min_child_weight`: 1, `balanced_sample_weight`: False

---

## 6. Model Benchmark & Comparison Metrics

- **FACT:** Final Test Evaluation Metrics Comparison  
  **SOURCE FILE:** `results/models/model_comparison_final.csv`, `results/models/hybrid_xgboost/hybrid_test_metrics.json`, `results/models/svm/svm_baseline_report.txt`, `results/models/efficientnet_xgboost/efficientnet_xgboost_report.txt`  
  **LOCATION:** `model_comparison_final.csv:1-4`  
  **VALUE:**  
  - **Baseline 1 (SVM + 62 Handcrafted):**  
    - Accuracy: 0.9116 (91.16%)  
    - Macro F1: 0.8494 (84.94%)  
    - Weighted F1: 0.9161 (91.61%)  
    - Macro Precision: 0.8206 (82.06%)  
    - Macro Recall: 0.8860 (88.60%)  
    - Inference Latency: 1.374 s total (0.44 ms/image)  
  - **Baseline 2 (XGBoost + 1280 EfficientNet-B0):**  
    - Accuracy: 0.9064516 (90.65%)  
    - Macro F1: 0.8399332 (83.99%)  
    - Weighted F1: 0.9098666 (90.99%)  
    - Macro Precision: 0.8180 (81.80%)  
    - Macro Recall: 0.8679 (86.79%)  
    - Inference Latency: 0.0511 s total (0.0165 ms/image)  
  - **Proposed Hybrid Framework (XGBoost + 1342 Fused):**  
    - Accuracy: 0.9209677 (92.10%)  
    - Macro F1: 0.8660282 (86.60%)  
    - Weighted F1: 0.9244098 (92.44%)  
    - Macro Precision: 0.8454377 (84.54%)  
    - Macro Recall: 0.8940000 (89.40%)  
    - Inference Latency: 0.028199 s total (0.0091 ms/image)  

- **FACT:** Hybrid Model Validation Performance  
  **SOURCE FILE:** `results/models/hybrid_xgboost/hybrid_validation_metrics.json`, `hybrid_model_report.txt`  
  **LOCATION:** `hybrid_validation_metrics.json:1-5`, `hybrid_model_report.txt:48-52`  
  **VALUE:**  
  - Validation Accuracy: 0.9583199 (95.83%)  
  - Validation Macro F1: 0.9071344 (90.71%)  
  - Validation Weighted F1: 0.9576935 (95.77%)  

- **FACT:** Accuracy and Macro F1 Gains of Proposed Hybrid Model  
  **SOURCE FILE:** `results/models/model_comparison_final.csv`, `hybrid_model_report.txt:121-128`  
  **LOCATION:** `model_comparison_final.csv:4`  
  **VALUE:**  
  - vs. Handcrafted SVM: Accuracy Change = +0.94 percentage points (+0.009368), Macro F1 Change = +1.66 percentage points (+0.016628)  
  - vs. EfficientNet XGBoost: Accuracy Change = +1.45 percentage points (+0.014516), Macro F1 Change = +2.61 percentage points (+0.026095)  

---

## 7. Per-Class Test Performance (Hybrid Model)

**SOURCE FILE:** `results/models/hybrid_xgboost/hybrid_per_class_metrics.csv`  
**LOCATION:** Lines 1-10

| Class ID | Class Label | Precision | Recall | F1-Score | Support |
|:---|:---|:---:|:---:|:---:|:---:|
| 0 | `0_NOR` | 0.9858 (98.58%) | 0.9370 (93.70%) | 0.9608 (96.08%) | 2000 |
| 1 | `1_F&S` | 0.7824 (78.24%) | 0.8867 (88.67%) | 0.8313 (83.13%) | 150 |
| 2 | `2_SD`  | 0.6703 (67.03%) | 0.8133 (81.33%) | 0.7349 (73.49%) | 150 |
| 3 | `3_MY`  | 0.6440 (64.40%) | 0.8200 (82.00%) | 0.7214 (72.14%) | 150 |
| 4 | `4_AP`  | 0.9343 (93.43%) | 0.8533 (85.33%) | 0.8920 (89.20%) | 150 |
| 5 | `5_BN`  | 0.9653 (96.53%) | 0.9267 (92.67%) | 0.9456 (94.56%) | 150 |
| 6 | `6_UN`  | 0.7966 (79.66%) | 0.9400 (94.00%) | 0.8624 (86.24%) | 150 |
| 7 | `7_IM`  | 0.9848 (98.48%) | 0.9750 (97.50%) | 0.9799 (97.99%) | 200 |

- **FACT:** Highest Performing Classes: `7_IM` (F1: 0.9799), `0_NOR` (F1: 0.9608), `5_BN` (F1: 0.9456)  
- **FACT:** Lowest Performing Classes: `3_MY` (F1: 0.7214), `2_SD` (F1: 0.7349)

---

## 8. Confusion Matrix Breakdown (Hybrid Model on 3,100 Test Samples)

**SOURCE FILE:** `results/models/hybrid_xgboost/hybrid_confusion_matrix.csv`  
**LOCATION:** Lines 1-10

- **True `0_NOR` (2,000 total):** 1,874 correctly classified; 3 to `1_F&S`, 42 to `2_SD`, 52 to `3_MY`, 5 to `4_AP`, 0 to `5_BN`, 24 to `6_UN`, 0 to `7_IM`
- **True `1_F&S` (150 total):** 133 correctly classified; 1 to `0_NOR`, 11 to `2_SD`, 2 to `3_MY`, 0 to `4_AP`, 1 to `5_BN`, 2 to `6_UN`, 0 to `7_IM`
- **True `2_SD` (150 total):** 122 correctly classified; 0 to `0_NOR`, 26 to `1_F&S`, 0 to `3_MY`, 0 to `4_AP`, 0 to `5_BN`, 2 to `6_UN`, 0 to `7_IM`
- **True `3_MY` (150 total):** 123 correctly classified; 13 to `0_NOR`, 6 to `1_F&S`, 3 to `2_SD`, 1 to `4_AP`, 0 to `5_BN`, 4 to `6_UN`, 0 to `7_IM`
- **True `4_AP` (150 total):** 128 correctly classified; 13 to `0_NOR`, 0 to `1_F&S`, 3 to `2_SD`, 1 to `3_MY`, 1 to `5_BN`, 4 to `6_UN`, 0 to `7_IM`
- **True `5_BN` (150 total):** 139 correctly classified; 0 to `0_NOR`, 0 to `1_F&S`, 0 to `2_SD`, 7 to `3_MY`, 1 to `4_AP`, 0 to `6_UN`, 3 to `7_IM`
- **True `6_UN` (150 total):** 141 correctly classified; 0 to `0_NOR`, 2 to `1_F&S`, 0 to `2_SD`, 4 to `3_MY`, 2 to `4_AP`, 1 to `5_BN`, 0 to `7_IM`
- **True `7_IM` (200 total):** 195 correctly classified; 0 to `0_NOR`, 0 to `1_F&S`, 1 to `2_SD`, 2 to `3_MY`, 0 to `4_AP`, 2 to `5_BN`, 0 to `6_UN`
- **Total Correct Predictions:** 1,874 + 133 + 122 + 123 + 128 + 139 + 141 + 195 = 2,855 / 3,100 = 92.0968%

---

## 9. Feature Ablation Study

**SOURCE FILE:** `results/models/hybrid_xgboost/ablation_results.csv`  
**LOCATION:** Lines 1-9

| Configuration | Dim | Train Sec | Val Acc | Val Macro F1 | Test Acc | Test Macro Prec | Test Macro Rec | Test Macro F1 | Test Wtd F1 | Test Infer Sec |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A. Shape only | 14 | 0.726 | 0.8126 | 0.5607 | 0.7761 | 0.5671 | 0.5558 | 0.5560 | 0.7715 | 0.0036 |
| B. GLCM Texture only | 12 | 0.630 | 0.8019 | 0.5568 | 0.7584 | 0.5505 | 0.5219 | 0.5241 | 0.7512 | 0.0045 |
| C. Colour only | 36 | 1.094 | 0.8653 | 0.7113 | 0.8558 | 0.7435 | 0.7123 | 0.7237 | 0.8516 | 0.0045 |
| D. All handcrafted | 62 | 2.249 | 0.9373 | 0.8580 | 0.9116 | 0.8258 | 0.8775 | 0.8479 | 0.9151 | 0.0043 |
| E. EfficientNet only | 1280 | 170.226 | 0.9396 | 0.8630 | 0.9065 | 0.8187 | 0.8686 | 0.8413 | 0.9098 | 0.0355 |
| F. Handcrafted + EfficientNet | 1342 | 173.104 | 0.9570 | 0.9038 | 0.9248 | 0.8515 | 0.9018 | 0.8735 | 0.9277 | 0.0356 |
| G. Full Hybrid (saved final) | 1342 | 129.193 | 0.9583 | 0.9071 | 0.9210 | 0.8454 | 0.8940 | 0.8660 | 0.9244 | 0.0282 |

- **Key Ablation Finding:** Combining Handcrafted + EfficientNet boosts Test Macro F1 from 0.8479 (handcrafted) and 0.8413 (deep only) to 0.8660 (saved final) and 0.8735 (retrained).

---

## 10. Explainability: SHAP vs. XGBoost Gain Feature Importance

- **FACT:** SHAP Explainer Configuration  
  **SOURCE FILE:** `src/models/run_explainability_ablation.py`, `results/models/hybrid_xgboost/shap_reproducibility.json`  
  **LOCATION:** `run_explainability_ablation.py:64`, `shap_reproducibility.json:2-3`  
  **VALUE:** TreeExplainer (`shap.TreeExplainer(model)`), Explanation Subset = 300 test samples, Random State = 42

- **FACT:** SHAP Feature-Group Attribution Breakdown  
  **SOURCE FILE:** `results/models/hybrid_xgboost/shap_feature_group_importance.csv`  
  **LOCATION:** Lines 1-6  
  **VALUE:**  
  - `EfficientNet-B0` (1280 features): Sum Mean |SHAP| = 3.81515, Attribution Share = **68.5942%**  
  - `Shape/Morphological` (14 features): Sum Mean |SHAP| = 0.75218, Attribution Share = **13.5237%**  
  - `Colour` (36 features): Sum Mean |SHAP| = 0.71549, Attribution Share = **12.8641%**  
  - `GLCM Texture` (12 features): Sum Mean |SHAP| = 0.27910, Attribution Share = **5.0180%**  
  *(Note: Strictly represents mean-absolute-SHAP feature attribution, not model accuracy, confidence, or defect percentage)*

- **FACT:** XGBoost Model-Based Gain Group Importance  
  **SOURCE FILE:** `results/models/hybrid_xgboost/hybrid_feature_group_importance.csv`  
  **LOCATION:** Lines 1-6  
  **VALUE:**  
  - `EfficientNet Deep` (1280 features): Total Gain Importance = 0.85502, Share = **85.5024%**, Top-20 Count = 13  
  - `Shape` (14 features): Total Gain Importance = 0.05772, Share = **5.7722%**, Top-20 Count = 4  
  - `Colour` (36 features): Total Gain Importance = 0.05458, Share = **5.4577%**, Top-20 Count = 2  
  - `GLCM Texture` (12 features): Total Gain Importance = 0.03268, Share = **3.2677%**, Top-20 Count = 1  

- **FACT:** Top Individual Features by Mean Absolute SHAP Attribution  
  **SOURCE FILE:** `results/models/hybrid_xgboost/shap_top20_features.csv`  
  **LOCATION:** Lines 1-22  
  **VALUE:**  
  1. `shape_height` (0.1883)  
  2. `color_lab_b_mean` (0.1763)  
  3. `shape_major_axis_length` (0.1498)  
  4. `shape_eccentricity` (0.1295)  
  5. `deep_feature_766` (0.1084)  
  6. `shape_area` (0.0993)  
  7. `color_hsv_s_mean` (0.0952)  
  8. `deep_feature_475` (0.0857)  
  9. `color_lab_b_std` (0.0807)  
  10. `deep_feature_203` (0.0722)  
  11. `deep_feature_484` (0.0667)  
  12. `deep_feature_262` (0.0610)  
  13. `deep_feature_104` (0.0552)  
  14. `glcm_contrast_std` (0.0533)  
  15. `color_lab_a_mean` (0.0532)  
  16. `deep_feature_574` (0.0487)  
  17. `glcm_contrast_mean` (0.0434)  
  18. `shape_aspect_ratio` (0.0421)  
  19. `deep_feature_979` (0.0392)  
  20. `glcm_homogeneity_mean` (0.0377)  

---

## 11. Reproducibility & Software Environment

- **FACT:** Operating System Platform  
  **SOURCE FILE:** `results/models/hybrid_xgboost/hybrid_reproducibility.json`  
  **LOCATION:** Line 10  
  **VALUE:** Windows 10 (AMD64, build 10.0.19045-SP0)

- **FACT:** Python Version Recorded in Experiment Metadata  
  **SOURCE FILE:** `results/models/hybrid_xgboost/hybrid_reproducibility.json`, `shap_reproducibility.json`  
  **LOCATION:** Line 9 (hybrid), Line 4 (shap)  
  **VALUE:** Python 3.14.2 (Dec 5 2025, MSC v.1944 64 bit AMD64)

- **FACT:** Core Library Versions  
  **SOURCE FILE:** `results/models/hybrid_xgboost/shap_reproducibility.json`, `hybrid_reproducibility.json`  
  **LOCATION:** `shap_reproducibility.json:5-8`, `hybrid_reproducibility.json:11-12`  
  **VALUE:**  
  - XGBoost: `3.1.3`  
  - SHAP: `0.50.0`  
  - scikit-learn: `1.8.0`  
  - Pandas: `3.0.5`  
  - NumPy: `2.5.3` (in hybrid train metadata), `2.4.1` (in SHAP metadata)  
  *(Note: Both NumPy versions are documented transparently for full reproducibility disclosure)*

- **FACT:** Hardware Environment  
  **SOURCE FILE:** Experimental Execution Records  
  **LOCATION:** N/A  
  **VALUE:** **[NOT EXPLICITLY RECORDED IN REPOSITORY LOGS]**

- **FACT:** Rice / Non-Rice Binary Classifier Gate  
  **SOURCE FILE:** `src/models/train_rice_gate.py`  
  **LOCATION:** `train_rice_gate.py:20-35`  
  **VALUE:** Heuristic/skeleton placeholder implementation; a dedicated binary classifier trained on true negative non-rice imagery was not trained in the primary pipeline. Documented as a known limitation.
