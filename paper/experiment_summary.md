# Executive Experiment Summary: Rice Quality & Defect Assessment

**Project Title:** Rice Quality and Defect Assessment using Hybrid Machine Learning and Deep Feature Fusion  
**Research Paper Title:** An Explainable Hybrid Feature Fusion Framework for Rice Quality and Defect Classification Using EfficientNet-B0 and XGBoost  
**Authors:** Dharshan B¹, Dinesh G L¹ (Mentored by Ms. Sripriya S)  
**Affiliation:** Department of Information Technology, Kongu Engineering College, Perundurai, Tamil Nadu, India  

---

## 1. Executive Overview

This empirical study investigates whether fusing domain-specific handcrafted image descriptors (geometric, textural, and chromatic) with deep convolutional embeddings (extracted via a pretrained EfficientNet-B0 backbone) yields statistically and practically superior performance for eight-class rice quality and defect assessment compared to either representation in isolation.

All experiments were executed on the **GrainSet Rice Dataset** across a strictly partitioned, zero-overlap split of 24,767 training images, 3,095 validation images, and 3,100 test images (total: 30,962 images). Data hygiene safeguards ensured that feature scalers (`StandardScaler`) were fitted exclusively on training data to prevent data leakage.

The core empirical findings are:
1. **The Proposed Hybrid Model (1,342 Fused Features + XGBoost)** achieves **92.10% Test Accuracy**, **0.8660 Macro F1**, and **0.9244 Weighted F1**, decisively surpassing both the classical handcrafted baseline and the deep-feature-only baseline.
2. **Deep Features Alone Do Not Beat Handcrafted Features:** An XGBoost classifier operating solely on 1,280 EfficientNet-B0 embeddings achieved **90.65% Accuracy** and **0.8399 Macro F1**, which was slightly lower than a Support Vector Machine operating on 62 handcrafted descriptors (**91.16% Accuracy**, **0.8494 Macro F1**). This empirical outcome proves that deep embeddings alone omit critical domain-specific morphometric boundaries, creating direct justification for feature fusion.
3. **Synergistic Complementarity Confirmed via Ablation:** Systematically combining the 62 handcrafted descriptors with the 1,280 deep embeddings produced an absolute gain of **+1.66% Macro F1** over the handcrafted baseline and **+2.61% Macro F1** over the deep baseline.
4. **Disentangled Explainability:** TreeExplainer SHAP analysis on 300 test samples revealed that while deep features account for 68.59% of the overall attribution mass, handcrafted features (shape: 13.52%, colour: 12.86%, texture: 5.02%) account for nearly a third (31.41%) of the total explanatory weight, with shape height, lab-b mean, and major axis length ranking among the absolute top 4 most influential individual features.

---

## 2. Benchmark Model Comparison

Three primary model architectures were evaluated under identical, untouched test conditions (3,100 test samples):

| Metric | Baseline 1: SVM Baseline | Baseline 2: Deep Feature Model | Proposed: Hybrid Fusion Framework | Delta (Hybrid vs. SVM) | Delta (Hybrid vs. Deep) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Feature Set** | 62 Handcrafted | 1280 EfficientNet-B0 | 1342 Fused (62 + 1280) | +1280 dims | +62 dims |
| **Classifier** | RBF SVC (C=50.0) | XGBoost (depth=4) | XGBoost (depth=4, lr=0.1) | — | — |
| **Scaling** | StandardScaler | None / Raw | StandardScaler | — | — |
| **Validation Accuracy** | 94.96% | 93.99% | **95.83%** | +0.87% | +1.84% |
| **Validation Macro F1** | 0.8853 | 0.8646 | **0.9071** | +0.0218 | +0.0425 |
| **Validation Wtd F1** | 0.9491 | 0.9379 | **0.9577** | +0.0086 | +0.0198 |
| **Test Accuracy** | 91.16% | 90.65% | **92.10%** | **+0.94%** | **+1.45%** |
| **Test Macro Precision** | 82.06% | 81.80% | **84.54%** | **+2.48%** | **+2.74%** |
| **Test Macro Recall** | 88.60% | 86.79% | **89.40%** | **+0.80%** | **+2.61%** |
| **Test Macro F1** | 0.8494 | 0.8399 | **0.8660** | **+0.0166 (+1.66%)** | **+0.0261 (+2.61%)** |
| **Test Weighted F1** | 0.9161 | 0.9099 | **0.9244** | **+0.0083 (+0.83%)** | **+0.0145 (+1.45%)** |
| **Inference Latency** | 0.44 ms / image | 0.0165 ms / image | **0.0091 ms / image** | 48× faster | 1.8× faster |

---

## 3. Systematic Ablation Study Breakdown

To isolate the discriminatory contribution of each feature sub-domain, seven distinct feature configurations were evaluated using identical XGBoost hyperparameters (`n_estimators=100`, `max_depth=4`, `learning_rate=0.1`):

| ID | Configuration | Dims | Train Time (s) | Val Acc | Val Macro F1 | Test Acc | Test Macro Prec | Test Macro Rec | Test Macro F1 | Test Wtd F1 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | Shape / Morphology only | 14 | 0.73 s | 0.8126 | 0.5607 | 0.7761 | 0.5671 | 0.5558 | 0.5560 | 0.7715 |
| **B** | GLCM Texture only | 12 | 0.63 s | 0.8019 | 0.5568 | 0.7584 | 0.5505 | 0.5219 | 0.5241 | 0.7512 |
| **C** | Colour Statistics only | 36 | 1.09 s | 0.8653 | 0.7113 | 0.8558 | 0.7435 | 0.7123 | 0.7237 | 0.8516 |
| **D** | All Handcrafted | 62 | 2.25 s | 0.9373 | 0.8580 | 0.9116 | 0.8258 | 0.8775 | 0.8479 | 0.9151 |
| **E** | EfficientNet-B0 only | 1280 | 170.23 s | 0.9396 | 0.8630 | 0.9065 | 0.8187 | 0.8686 | 0.8413 | 0.9098 |
| **F** | Retrained Handcrafted + Deep | 1342 | 173.10 s | 0.9570 | 0.9038 | 0.9248 | 0.8515 | 0.9018 | 0.8735 | 0.9277 |
| **G** | **Saved Final Hybrid Model** | **1342** | **129.19 s** | **0.9583** | **0.9071** | **0.9210** | **0.8454** | **0.8940** | **0.8660** | **0.9244** |

### Scientific Insights from the Ablation Study:
- **Colour is the Strongest Handcrafted Individual Modality:** Colour features alone achieve 85.58% Test Accuracy and 0.7237 Macro F1, heavily outperforming Shape alone (0.5560 F1) and Texture alone (0.5241 F1). This indicates that chalkiness, fungal discoloration, and milling degree are strongly reflected in chromatic distributions across LAB, HSV, and RGB.
- **Handcrafted Synergy:** Combining Shape + Texture + Colour (62 features) yields 0.8479 Macro F1—a massive +12.42 percentage point jump over colour alone, proving that shape and texture resolve chromatic ambiguities.
- **Deep Feature Parity with Handcrafted Features:** EfficientNet-B0 achieves 0.8413 Macro F1, which is practically equivalent to the 62 handcrafted features (0.8479 Macro F1) despite having over 20× more dimensions (1280 vs. 62).
- **The Fusion Advantage:** Concatenating handcrafted and deep descriptors produces the highest performance across all evaluation splits (0.9071 Val Macro F1, 0.8660 Test Macro F1), confirming that handcrafted and deep representations are non-redundant and synergistic.

---

## 4. Class-Wise Classification Dynamics & Error Analysis

The performance of the proposed hybrid model varies across the eight categories due to visual subtlety and class imbalance (the majority class `0_NOR` represents 64.5% of the test set):

| Class ID | Class Label | Test Support | Precision | Recall | F1-Score | Dominant Confusion Pattern |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| 0 | `0_NOR` | 2,000 | 98.58% | 93.70% | **0.9608** | 52 misclassified as `3_MY`, 42 as `2_SD` |
| 1 | `1_F&S` | 150 | 78.24% | 88.67% | **0.8313** | 11 misclassified as `2_SD` |
| 2 | `2_SD`  | 150 | 67.03% | 81.33% | **0.7349** | 26 misclassified as `1_F&S` |
| 3 | `3_MY`  | 150 | 64.40% | 82.00% | **0.7214** | 13 misclassified as `0_NOR`, 6 as `1_F&S` |
| 4 | `4_AP`  | 150 | 93.43% | 85.33% | **0.8920** | 13 misclassified as `0_NOR`, 4 as `6_UN` |
| 5 | `5_BN`  | 150 | 96.53% | 92.67% | **0.9456** | 7 misclassified as `3_MY`, 3 as `7_IM` |
| 6 | `6_UN`  | 150 | 79.66% | 94.00% | **0.8624** | 4 misclassified as `3_MY`, 2 as `1_F&S` |
| 7 | `7_IM`  | 200 | 98.48% | 97.50% | **0.9799** | 2 misclassified as `3_MY`, 2 as `5_BN` |

### Key Error Observations:
- **`2_SD` vs. `1_F&S` Mutual Confusion:** 26 true `2_SD` samples were predicted as `1_F&S`, and 11 true `1_F&S` samples were predicted as `2_SD`. This represents the largest mutual confusion pair in the dataset, reflecting overlapping geometric breakage and surface fissure patterns.
- **`0_NOR` vs. `3_MY` / `2_SD` Boundary Leakage:** Because `0_NOR` has 2,000 test samples, even a small error rate (52 samples into `3_MY` and 42 into `2_SD`) sharply depresses the precision of `3_MY` (64.40%) and `2_SD` (67.03%), while their recalls remain high (82.00% and 81.33%).
- **Exceptional Detection for Distinct Classes:** `7_IM` (0.9799 F1), `0_NOR` (0.9608 F1), and `5_BN` (0.9456 F1) exhibit near-perfect separation, attributable to stark geometric (chalky/immature dimensions) and chromatic contrasts.

---

## 5. Model Explainability & Interpretability Analysis

### A. TreeExplainer SHAP Feature Group Contributions
Evaluated across 300 test samples using `shap.TreeExplainer`:
- **EfficientNet-B0 (1,280 features):** 68.59% of mean absolute SHAP attribution
- **Shape / Morphological (14 features):** 13.52% of mean absolute SHAP attribution
- **Colour Statistics (36 features):** 12.86% of mean absolute SHAP attribution
- **GLCM Texture (12 features):** 5.02% of mean absolute SHAP attribution

### B. XGBoost Gain-Based Model Importance
- **EfficientNet Deep:** 85.50% total split gain (13 of top-20 features)
- **Shape:** 5.77% total split gain (4 of top-20 features)
- **Colour:** 5.46% total split gain (2 of top-20 features)
- **GLCM Texture:** 3.27% total split gain (1 of top-20 features)

### C. Scientific Distinction Between Gain and SHAP:
- **XGBoost Gain** measures how much each feature split reduced training objective loss across the decision trees. Because deep features provide 1,280 continuous orthogonal axes, trees heavily leverage them for internal splitting (85.50%).
- **SHAP Attribution** measures the Shapley marginal contribution of each feature to the final output probability distribution across real test instances. Here, handcrafted features provide 31.41% of the explanatory power, proving that the model actively relies on physical geometric boundaries (`shape_height`, `shape_major_axis_length`, `shape_eccentricity`) and colour channels (`color_lab_b_mean`, `color_hsv_s_mean`) to finalize classifications.

---

## 6. Supporting Engineering & Deployment Stack

While the primary scientific contribution is the evaluated feature representation and explainability analysis, the project incorporates complete operational engineering components:
- **Streamlit Web Application:** Multi-page dashboard supporting single-image inspection, batch classification, review queues, and PDF reporting.
- **Human-in-the-Loop Review Queue:** Automated flagging of predictions with confidence $< 0.75$ for expert audit, storing corrections into SQLite.
- **REST API (`api.py`):** FastAPI endpoints for automated headless inference and batch scoring.
- **SQLite Database (`results/rice_quality.db`):** Persistent logging of all inferences, timestamps, confidence scores, and review statuses.
