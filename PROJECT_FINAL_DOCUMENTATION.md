# Rice Quality and Defect Assessment

## 1. Project Summary

This project develops a computer-vision system for classifying rice grains into eight quality and defect categories. It combines interpretable handcrafted image descriptors with deep visual representations from EfficientNet-B0, then classifies the fused feature vector using XGBoost.

The system is delivered through a Streamlit application and includes prediction probabilities, SHAP-based local explanations, batch analysis, review workflows, reports, persistent history, model comparison, and a REST API.

## 2. Research Problem

Manual rice inspection can be slow, subjective, and difficult to scale. The objective is to build a reproducible image-based system that supports consistent rice quality assessment while exposing confidence and feature contributions for human review.

## 3. Dataset Classes

| Label | Class |
|---|---|
| `0_NOR` | Normal rice |
| `1_F&S` | Fusarium and spot |
| `2_SD` | Seed damage |
| `3_MY` | Mycotoxin |
| `4_AP` | Aflatoxin |
| `5_BN` | Broken rice |
| `6_UN` | Underdeveloped or unclassified |
| `7_IM` | Immature rice |

## 4. Methodology

### Preprocessing

1. Load the image.
2. Convert to grayscale and additional colour spaces.
3. Apply Gaussian denoising.
4. Segment the grain using Otsu thresholding.
5. Refine the mask with morphological operations.
6. Select the primary grain contour.
7. Remove the background.
8. Crop and standardize the grain to `224 x 224` pixels.

### Handcrafted Features

The system extracts 62 handcrafted features:

- 14 shape and morphological features
- 12 GLCM texture features
- 36 colour features

### Deep Features

EfficientNet-B0 produces a 1,280-dimensional deep feature representation from the standardized grain image.

### Feature Fusion

```text
62 handcrafted features + 1280 deep features = 1342 fused features
```

The fused vector is scaled using the saved training scaler and classified using the saved hybrid XGBoost model.

## 5. Model Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| SVM + Handcrafted Features | 91.16% | 84.94% | 91.61% |
| XGBoost + EfficientNet Features | 90.65% | 83.99% | 90.99% |
| XGBoost + Hybrid Features | 92.10% | 86.60% | 92.44% |

The hybrid model produced the strongest saved evaluation result. Its accuracy was `92.0968%`, with a macro F1 of `86.6028%` and weighted F1 of `92.4410%`.

## 6. Explainability

SHAP TreeExplainer is used for local explanations of individual XGBoost predictions. The interface separates:

- positive contributors, which support the selected class
- negative contributors, which oppose the selected class

Handcrafted features such as shape, colour, and texture are easier to interpret directly. EfficientNet features are learned representations and should be described as learned visual contributions rather than assigned unsupported human meanings.

## 7. Application Features

The Streamlit application provides these pages:

- Dashboard
- Analyze Rice (with Segmentation, Deep Visual Saliency Heatmap, and Agronomic Diagnosis)
- Multi-Grain Inspection (Bulk segmentation, defect bounding boxes, and commercial quality grading)
- Batch Analysis
- Prediction History
- Review Queue
- SHAP / Explainability (with Agronomic Feature Dictionary)
- Reports
- Webcam
- Model Performance
- System / API Status
- About Project

Supporting functionality includes CSV export, PDF export, SQLite persistence, manual review correction, confidence-based review status, and a full REST API with upload validation, batch inference (`/predict/batch`), bulk grading (`/multi-grain`), and explanation endpoints (`/explain`).

## 8. Validation Evidence

Verified checks include:

- Python compilation of application and UI modules
- Streamlit application import
- API health endpoint
- invalid API file rejection
- PDF generation
- SQLite save and reload
- blank-frame rejection
- obvious non-rice object rejection
- realtime monitoring tests
- one-sample-per-class inference diagnostic
- 16-image validation sample: 15 correct, 1 misclassified (`93.75%` sample accuracy)

The 16-image sample is a smoke check, not a replacement for the official test-set metrics.

## 9. Webcam Limitation

The binary rice-versus-non-rice gate is prepared but not trained because the repository does not contain genuine non-rice training images. The webcam page therefore blocks analysis and displays setup instructions until real data is added.

Required structure:

```text
datasets/rice_gate/
├── rice/
└── non_rice/
```

The binary gate should only be trained after collecting real examples such as hands, leaves, stones, other grains, paper, cloth, blank backgrounds, and household objects. No fabricated negatives should be used.

## 10. Limitations

- Some visually similar defect classes can be confused.
- Deep feature names do not directly describe physical defects.
- Confidence values should be calibrated before high-stakes deployment.
- The webcam gate requires real non-rice data and independent validation.
- Performance may change under different lighting, cameras, backgrounds, and grain arrangements.
- The displayed evaluation metrics are based on the saved project artifacts and should be interpreted with the dataset split and experimental protocol documented in the final report.

## 11. Paper Contribution Statement

The contribution should not claim that the project invented SHAP, EfficientNet, or XGBoost. A defensible contribution statement is:

> This work presents a reproducible hybrid rice-quality classification framework that combines handcrafted shape, texture, and colour descriptors with EfficientNet-B0 deep features and XGBoost classification. The framework compares feature families, preserves a verified training-to-inference feature contract, and provides confidence-aware, explainable inspection support through a deployable application.

## 12. Demonstration Flow

1. Open the Streamlit dashboard.
2. Navigate to Analyze Rice.
3. Upload a rice image.
4. Show the segmented image beside the original.
5. Click Analyze Rice.
6. Explain the predicted class and probability table.
7. Click Explain Prediction and discuss positive and negative SHAP contributors.
8. Show Batch Analysis and CSV export.
9. Show Review Queue and manual correction.
10. Show Model Performance and compare the three models.
11. Show Reports and System / API Status.
12. Explain the webcam gate limitation honestly.

## 13. Run Commands

### Streamlit

```bash
source ".venv/Scripts/activate"
python -m streamlit run app.py
```

Open `http://localhost:8501`.

### REST API

```bash
source ".venv/Scripts/activate"
uvicorn api:app --reload --port 8000
```

Open `http://127.0.0.1:8000/docs`.

## 14. Final Status

The core classification system, inference contract, UI workflow, reporting, explainability, persistence, and API are implemented and validated. Final submission work should focus on the paper, figures, screenshots, error analysis, and presentation rather than adding unrelated administrative features.
