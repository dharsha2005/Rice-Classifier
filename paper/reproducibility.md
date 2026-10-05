# Reproducibility Protocol: Rice Quality & Defect Assessment

This protocol provides complete, step-by-step specifications to reproduce every experiment, feature extraction step, model training routine, ablation benchmark, and explainability output presented in:  
*"An Explainable Hybrid Feature Fusion Framework for Rice Quality and Defect Classification Using EfficientNet-B0 and XGBoost"*

---

## 1. Environment & Software Specifications

The original experiments were conducted under the following verified environment:

- **Operating System:** Windows 10 (AMD64, build 10.0.19045-SP0)
- **Python Version:** 3.14.2 (64-bit AMD64)
- **Machine Learning & Data Libraries:**
  - `xgboost`: 3.1.3
  - `shap`: 0.50.0
  - `scikit-learn`: 1.8.0
  - `torch` & `torchvision`: PyTorch with torchvision 0.20+ (EfficientNet-B0 default weights)
  - `opencv-python`: 4.10+
  - `numpy`: 2.5.3 (hybrid training metadata) / 2.4.1 (SHAP metadata)
  - `pandas`: 3.0.5
  - `joblib`: 1.4+
  - `matplotlib`: 3.9+
  - `seaborn`: 0.13+

To install the exact core dependencies:
```bash
pip install numpy pandas scikit-learn xgboost shap opencv-python matplotlib seaborn joblib torch torchvision
```

---

## 2. Directory Layout & Dataset Verification

The project enforces a strict, modular layout:
```
Rice classifier final project/
├── datasets/
│   ├── rice_train.txt        # 24,767 entries
│   ├── rice_val.txt          # 3,095 entries
│   ├── rice_test.txt         # 3,100 entries
│   └── rice_train_bal.txt    # 9,528 entries (balanced train sub-sample)
├── rice/                     # Root folder containing raw images organized by class folder
│   ├── 0_NOR/
│   ├── 1_F&S/
│   ├── 2_SD/
│   ├── 3_MY/
│   ├── 4_AP/
│   ├── 5_BN/
│   ├── 6_UN/
│   └── 7_IM/
├── src/
│   ├── preprocessing/preprocess.py
│   ├── features/
│   │   ├── handcrafted_features.py
│   │   ├── efficientnet_features.py
│   │   ├── run_feature_extraction.py
│   │   └── run_efficientnet_extraction.py
│   └── models/
│       ├── train_svm_baseline.py
│       ├── train_xgboost_efficientnet.py
│       ├── train_hybrid_xgboost.py
│       └── run_explainability_ablation.py
└── results/
    ├── preprocessing/
    ├── features/
    └── models/
```

### Step 1: Verify Dataset Integrity & Split Hygiene
Before running feature extraction, confirm that no missing files or split overlaps exist:
```bash
python -m src.data.dataset_inspection
```
*Verification Check:* Ensure 0 overlapping filenames across train, val, and test splits. The expected split counts must match: Train = 24,767; Val = 3,095; Test = 3,100 (Total: 30,962).

---

## 3. Preprocessing & Grain Segmentation

The preprocessing pipeline isolates the primary rice grain, eliminates background illumination artifacts, and standardizes image resolution to 224×224 pixels with isotropic aspect-ratio padding:
```bash
python src/preprocessing/preprocess.py --split all
```

**Key Parameters Applied in `src/preprocessing/preprocess.py`:**
- Gaussian Blur Kernel: `(5, 5)`, `sigma = 0`
- Thresholding: Automatic Otsu thresholding (`cv2.THRESH_BINARY + cv2.THRESH_OTSU`)
- Morphological Kernel: Elliptical structuring element (`cv2.MORPH_ELLIPSE`), size `(5, 5)`
- Morphological Operations: Close (bridge internal voids) followed by Open (remove stray noise)
- Contour Filtering: External contour extraction (`cv2.RETR_EXTERNAL`), largest contour selected with area threshold $> 500$ px
- Bounding Box Padding: Margin ratio = 0.05 (5%)
- Standardized Image Size: $224 \times 224 \times 3$, isotropic scaling with fill scale = 0.90, background black padding

---

## 4. Feature Extraction Workflows

### Step 2: Extract 62 Handcrafted Features
Extract 14 morphological, 12 GLCM texture, and 36 colour statistics from the segmented grain region:
```bash
python src/features/run_feature_extraction.py
```
- Outputs saved to: `results/features/{train,val,test}_handcrafted_features.csv`
- Total columns: 66 (4 metadata columns: `image_path`, `split`, `class_id`, `class_name` + 62 numerical features)

### Step 3: Extract 1,280 EfficientNet-B0 Deep Features
Pass standardized 224×224 images through ImageNet-pretrained EfficientNet-B0 with classification head replaced by `Identity()`:
```bash
python src/features/run_efficientnet_extraction.py
```
- Transforms: ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- Evaluation Mode: `model.eval()`, `torch.inference_mode()`
- Batch Size: 32
- Outputs saved to: `results/features/deep/{train,val,test}_efficientnet_features.csv`

---

## 5. Model Training & Evaluation Protocols

### Step 4: Train & Evaluate Baseline 1 (Handcrafted SVM)
```bash
python src/models/train_svm_baseline.py
```
- Pipeline: `StandardScaler` (fit on train only) + `sklearn.svm.SVC`
- Validation Grid: $C \in [0.1, 1.0, 5.0, 10.0, 50.0]$, $\gamma \in [\text{'scale'}, \text{'auto'}, 0.01, 0.05, 0.1]$, `class_weight` $\in [\text{None}, \text{'balanced'}]$
- Selected Best Parameters: $C = 50.0$, $\gamma = \text{'scale'}$, `class_weight` = None
- Output: `results/models/svm/svm_baseline.joblib` and `svm_baseline_report.txt`

### Step 5: Train & Evaluate Baseline 2 (EfficientNet-B0 + XGBoost)
```bash
python src/models/train_xgboost_efficientnet.py
```
- Model: `xgboost.XGBClassifier(objective='multi:softprob', tree_method='hist', random_state=42)`
- Selected Best Parameters: `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=1`, `balanced_sample_weight=False`
- Output: `results/models/efficientnet_xgboost/xgboost_efficientnet.joblib`

### Step 6: Train & Evaluate Proposed Main Hybrid Model (1,342 Features + XGBoost)
```bash
python src/models/train_hybrid_xgboost.py
```
- Feature Join: Explicit key join on `image_path`, verifying metadata alignment
- Scaling: `StandardScaler` fitted strictly on the 24,767 training samples and saved to `hybrid_scaler.joblib`
- XGBoost Objective: `multi:softprob` across 8 classes (`mlogloss`)
- Optimal Hyperparameters:
  - `n_estimators`: 100
  - `max_depth`: 4
  - `learning_rate`: 0.1
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `min_child_weight`: 1
  - `balanced_sample_weight`: False
  - `random_state`: 42
- Saved Model Artifact: `results/models/hybrid_xgboost/hybrid_xgboost_model.joblib`

---

## 6. Ablation Study & SHAP Explainability Protocols

Execute the combined ablation benchmark and SHAP explanation suite:
```bash
python src/models/run_explainability_ablation.py
```

### Ablation Configuration:
Trains and evaluates 7 configurations across exact train/val/test splits using identical XGBoost hyperparameters:
1. Shape only (14-D)
2. GLCM Texture only (12-D)
3. Colour only (36-D)
4. All Handcrafted (62-D)
5. EfficientNet-B0 only (1,280-D)
6. Retrained Hybrid (1,342-D)
7. Saved Final Hybrid Model (1,342-D)

### SHAP Configuration:
- Explainer: `shap.TreeExplainer(model)`
- Explanation Dataset: 300 test samples sampled with `random_state=42`
- Value Aggregation: Mean absolute SHAP values averaged across 300 instances and 8 output classes
- Group Summation: Sum of mean absolute SHAP values for Shape, Texture, Colour, and Deep groups normalized to 100%

---

## 7. Data Hygiene & Leakage Prevention Verification

1. **Scaler Hygiene:** No feature scaling is ever applied to the entire dataset before splitting. `StandardScaler.fit()` is strictly executed on training features `X_train`. `StandardScaler.transform()` is subsequently applied to `X_val` and `X_test`.
2. **Split Isolation:** Train, validation, and test files are pre-partitioned in `datasets/` and cross-checked for zero filename overlap.
3. **Hyperparameter Selection:** Hyperparameter selection is strictly conducted using the Validation split Macro F1 score. The Test split is evaluated exactly once on the selected final model to assess generalization.
