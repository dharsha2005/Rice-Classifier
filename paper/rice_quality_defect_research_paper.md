# An Explainable Hybrid Feature Fusion Framework for Rice Quality and Defect Classification Using EfficientNet-B0 and XGBoost

**Dharshan B¹, Dinesh G L¹**  
**Mentored by:** Ms. Sripriya S  
**¹Department of Information Technology, Kongu Engineering College, Perundurai, Tamil Nadu, India**

## Abstract

Automated rice-image analysis can support consistent inspection of visually distinct grain categories, but performance may depend on whether the representation captures complementary geometric, textural, chromatic, and learned visual information. This paper presents an explainable hybrid feature-fusion framework for eight-class rice image classification. The implemented pipeline denoises and segments the primary grain, standardizes it to a three-channel 224 × 224 representation, extracts 62 handcrafted descriptors, and obtains a 1,280-dimensional representation from a pretrained EfficientNet-B0 backbone. The two representations are concatenated into a 1,342-dimensional vector, standardized using a training-fitted StandardScaler, and classified with multiclass XGBoost. SVM with handcrafted features and XGBoost with EfficientNet-B0 features are used as comparison experiments. On the saved test evaluation, the final hybrid model achieved 92.0968% accuracy, 0.8660 macro F1, and 0.9244 weighted F1. The corresponding validation accuracy and macro F1 were 95.8320% and 0.9071. A controlled ablation study shows that individual feature groups are weaker than the combined representation, while SHAP analysis on a reproducible 300-sample test subset indicates that EfficientNet-B0 features contribute 68.5942% of the mean-absolute-SHAP group summary, followed by shape, colour, and GLCM groups. The project also provides a Streamlit interface, review workflow, reports, persistence, and API support as deployment components. The results support hybrid representation as a useful experimentally evaluated direction, while the dataset-specific scope, latent deep-feature semantics, probability calibration status, and untrained rice/non-rice gate remain limitations.

**Index Terms—** Rice image classification, rice quality assessment, computer vision, handcrafted features, feature fusion, EfficientNet-B0, XGBoost, explainable AI, SHAP.

## I. Introduction

Rice inspection involves visual distinctions that may appear in grain dimensions, surface texture, colour distribution, and other appearance patterns. An image-based classifier can assist inspection by producing a repeatable class estimate, but a system intended for research use should also preserve a defensible train-to-inference contract and expose evidence about model behaviour.

This work investigates a hybrid representation for the eight raw dataset labels `0_NOR`, `1_F&S`, `2_SD`, `3_MY`, `4_AP`, `5_BN`, `6_UN`, and `7_IM`. The repository documentation records these as raw labels and explicitly states that their abbreviations are not interpreted. Consequently, this paper preserves the labels without assigning expanded class meanings. Any expanded semantic interpretation is **[NOT VERIFIED FROM PROJECT]**.

The research objective is to determine whether a representation that combines handcrafted shape, GLCM texture, and colour descriptors with EfficientNet-B0 features improves classification over the implemented handcrafted-only and deep-feature-only comparisons. The framework also uses SHAP TreeExplainer for local and group-level interpretation. The application interface, database, API, reporting, and review functions are supporting deployment components; the primary research contribution is the experimentally evaluated representation and its analysis.

The contributions are:

1. A reproducible 1,342-dimensional hybrid feature pipeline combining 62 handcrafted descriptors and 1,280 EfficientNet-B0 features.
2. A controlled comparison with an SVM handcrafted baseline and an EfficientNet-B0 plus XGBoost baseline.
3. A feature-group ablation study covering shape, GLCM texture, colour, all handcrafted features, EfficientNet-B0, and fused configurations.
4. SHAP-based analysis that distinguishes local contributions and group-level summaries from model probabilities and gain-based importance.
5. A deployable application workflow that preserves the saved model, scaler, feature schema, prediction history, and review artifacts.

## II. Related Work

Research in rice image analysis commonly uses combinations of image preprocessing, morphology, colour statistics, texture descriptors, and machine learning classifiers. Handcrafted descriptors can provide directly interpretable measurements, while convolutional neural networks can learn higher-level representations from images. Feature fusion is a natural way to investigate whether these representation families provide complementary information. Tree-based classifiers such as XGBoost can then operate on tabular fused features, and explainability methods such as SHAP can expose feature contributions for individual outputs.

A publication-grade bibliography for these areas requires external academic verification. The repository does not contain a bibliography or cited source list. Therefore, the reference section below is marked **[REFERENCES REQUIRE EXTERNAL VERIFICATION]** rather than containing invented citations. Before submission, the authors should add verified papers on rice quality assessment, GLCM texture analysis, EfficientNet, XGBoost, SHAP, and hybrid feature fusion.

The research gap addressed here is deliberately stated conservatively: existing approaches may emphasize handcrafted descriptors or learned representations, while this work experimentally evaluates a combined representation under a fixed train/validation/test protocol and analyzes feature-group contributions. A stronger claim such as “first” is not supported by the repository alone.

## III. Dataset and Problem Formulation

### A. Dataset

The project uses the GrainSet rice dataset as identified in the repository reports. The standard split contains 24,767 training images, 3,095 validation images, and 3,100 test images, for 30,962 images in total. A balanced training subset of 9,528 images is also recorded, but the final hybrid experiment's reproducibility artifacts use the standard train, validation, and test counts. The test split contains 2,000 `0_NOR` images, 150 images for each of `1_F&S` through `6_UN`, and 200 `7_IM` images. The train split is strongly imbalanced, with `0_NOR` representing 64.5% of training images. This makes macro F1 and class-wise metrics important alongside accuracy and weighted F1.

No missing files or corrupt images were reported. The dataset report records zero filename overlap between train, validation, and test splits. It also records 9,528 identical filename/content pairs between the ordinary training tree and the balanced training subset; this is an expected relationship between those two training resources and is not reported as train-test leakage.

### B. Labels

The model predicts the following raw labels:

| Class ID | Raw label |
|---:|---|
| 0 | `0_NOR` |
| 1 | `1_F&S` |
| 2 | `2_SD` |
| 3 | `3_MY` |
| 4 | `4_AP` |
| 5 | `5_BN` |
| 6 | `6_UN` |
| 7 | `7_IM` |

Expanded class meanings are **[NOT VERIFIED FROM PROJECT]** and are intentionally not asserted here.

### C. Problem Formulation

Given a rice image $I$, the system computes a standardized grain representation, derives a handcrafted vector $h \in \mathbb{R}^{62}$ and an EfficientNet representation $d \in \mathbb{R}^{1280}$, and forms:

$$
 z = h \oplus d \in \mathbb{R}^{1342}.
$$

The training-fitted StandardScaler transforms $z$ into $\tilde{z}$, and XGBoost estimates an eight-class probability vector $p(y \mid I)$. The predicted class is the class with the largest model probability.

## IV. Proposed Methodology

### A. Image Preprocessing and Segmentation

The implemented preprocessor converts images into RGB, grayscale, HSV, and LAB representations. A Gaussian blur with a `(5, 5)` kernel and zero sigma is applied to the denoised grayscale image. Otsu thresholding separates foreground and background, after which an elliptical `(5, 5)` morphological closing operation bridges small internal gaps and an opening operation removes small stray regions. External contours are extracted with `cv2.RETR_EXTERNAL`, and the largest contour meeting the configured minimum area is selected as the primary grain.

A filled binary mask is generated from the selected contour. The grain is masked, cropped using its bounding box with a 5% margin, isotropically resized, and centered with background padding to a 224 × 224 three-channel standardized image. The same preprocessing contract is reused during inference.

### B. Handcrafted Feature Extraction

The project extracts 62 handcrafted features in three groups. The 14 shape/morphological descriptors include area, perimeter, width, height, aspect ratio, extent, solidity, circularity, eccentricity, major and minor axis lengths, equivalent diameter, convex-hull area, and bounding-box area. The 12 GLCM descriptors are mean and standard-deviation summaries for contrast, dissimilarity, homogeneity, energy, correlation, and ASM over distances 1 and 2 and angles 0°, 45°, 90°, and 135°. The 36 colour descriptors summarize mean, standard deviation, minimum, and maximum for the R, G, and B channels, the H, S, and V channels, and the L, A, and B channels of RGB, HSV, and LAB representations.

All handcrafted extraction records were reported as successful for the 30,962 standard-split images, with no missing or infinite values in the audit report.

### C. EfficientNet-B0 Deep Feature Extraction

The standardized 224 × 224 grain image is passed through the implemented EfficientNet-B0 feature extractor. The classifier uses the resulting 1,280-dimensional representation rather than an EfficientNet classification head. These dimensions encode learned visual patterns and complement manually defined shape, texture, and colour descriptors. Individual dimensions such as `deep_feature_766` are latent representations; the project does not provide evidence for assigning them direct human meanings.

### D. Hybrid Feature Fusion

The handcrafted and deep representations are concatenated in the verified training order:

$$
 z = [h_1,\ldots,h_{62},d_1,\ldots,d_{1280}].
$$

Therefore, $62 + 1280 = 1342$ features are passed to the scaling and classification stage. The feature schema and saved training CSV column order are checked during inference to reduce feature-order drift. StandardScaler is fitted on training data only and applied to validation, test, and inference samples.

### E. XGBoost Classification

The selected classifier is multiclass XGBoost with objective `multi:softprob`, eight classes, histogram tree construction, and `mlogloss` evaluation. The saved configuration uses 100 estimators, maximum depth 4, learning rate 0.1, subsample 0.8, column subsample 0.8, minimum child weight 1, and random state 42. The final selected configuration is unweighted because the repository's validation comparison reports a higher validation macro F1 for unweighted training than for the weighted alternative.

### F. Explainability Using SHAP

The project uses `shap.TreeExplainer` on the frozen hybrid XGBoost model. The reproducibility artifact states that explanations were generated for a 300-sample test subset with random state 42. For an individual explained class, a positive SHAP contribution pushes the model output toward that class, while a negative contribution pushes away from it. A negative contribution does not mean that the prediction is wrong.

Mean-absolute-SHAP group summaries are distinct from XGBoost gain importance. The SHAP group summary is 68.5942% EfficientNet-B0, 13.5237% Shape/Morphological, 12.8641% Colour, and 5.0180% GLCM Texture. These are contribution summaries over the explanation set, not probabilities, accuracy, or percentages of defective rice.

## V. Experimental Setup

The predefined train/validation/test split is retained. Models are selected using validation performance, with the final test split treated as the untouched evaluation split in the hybrid experiment report. The principal evaluation measures are accuracy, macro precision, macro recall, macro F1, weighted F1, per-class metrics, and confusion matrices.

The reported final environment metadata includes Python 3.14.2, XGBoost 3.1.3, SHAP 0.50.0, and scikit-learn 1.8.0 in the SHAP reproducibility record. The hybrid reproducibility record reports Windows 10, Python 3.14.2, NumPy 2.5.3, and pandas 3.0.5. The NumPy versions differ between saved metadata files and should be reconciled before publication. Hardware used for the recorded experiments is **[NOT VERIFIED FROM PROJECT]**.

The selected metrics are defined as:

$$
\text{Accuracy} = \frac{TP+TN}{\text{all samples}},
$$

with multiclass accuracy interpreted as the fraction of correctly classified samples. For a class $c$:

$$
\text{Precision}_c = \frac{TP_c}{TP_c+FP_c}, \quad
\text{Recall}_c = \frac{TP_c}{TP_c+FN_c},
$$

and:

$$
F1_c = \frac{2\,\text{Precision}_c\,\text{Recall}_c}{\text{Precision}_c+\text{Recall}_c}.
$$

Macro F1 is the unweighted mean of per-class F1 values, while weighted F1 weights each class by its support. Because the dataset is imbalanced, macro F1 and per-class recall are particularly informative.

## VI. Results and Discussion

### A. Model Comparison

**Table I. Model comparison on the saved test evaluation.**

| Model | Representation | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| SVM + Handcrafted | 62 handcrafted | 91.16% | 0.8494 | 0.9161 |
| EfficientNet-B0 + XGBoost | 1,280 deep | 90.65% | 0.8399 | 0.9099 |
| Full Hybrid XGBoost | 1,342 fused | **92.10%** | **0.8660** | **0.9244** |

The full hybrid model exceeds the handcrafted SVM baseline by 0.9368 percentage points in accuracy and 0.0166 in macro F1. It exceeds the EfficientNet-only comparison by 1.4516 percentage points in accuracy and 0.0261 in macro F1. These observations apply to the saved project experiments and do not establish universal superiority beyond this dataset and protocol.

The validation metrics for the selected hybrid model were 95.83199% accuracy, 0.907134 macro F1, and 0.957693 weighted F1. The final test metrics were lower, which is expected when validation is used for selection and test is held out for final assessment.

### B. Class-wise Performance

**Table II. Final hybrid class-wise test performance.**

| Raw class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `0_NOR` | 0.9858 | 0.9370 | 0.9608 | 2000 |
| `1_F&S` | 0.7824 | 0.8867 | 0.8313 | 150 |
| `2_SD` | 0.6703 | 0.8133 | 0.7349 | 150 |
| `3_MY` | 0.6440 | 0.8200 | 0.7214 | 150 |
| `4_AP` | 0.9343 | 0.8533 | 0.8920 | 150 |
| `5_BN` | 0.9653 | 0.9267 | 0.9456 | 150 |
| `6_UN` | 0.7966 | 0.9400 | 0.8624 | 150 |
| `7_IM` | 0.9848 | 0.9750 | 0.9799 | 200 |

The strongest F1 values are observed for `7_IM`, `0_NOR`, and `5_BN`, while the lowest are observed for `3_MY` and `2_SD`. The confusion matrix records, among other entries, 42 `0_NOR` samples predicted as `2_SD`, 52 `0_NOR` samples predicted as `3_MY`, 11 `1_F&S` samples predicted as `2_SD`, and 7 `5_BN` samples predicted as `3_MY`. These are observed error patterns; the available experiment does not provide sufficient evidence to establish their biological cause.

### C. Ablation Study

**Table III. Feature-group ablation and controlled configurations.**

| Configuration | Dimensions | Val. accuracy | Val. macro F1 | Test accuracy | Test macro F1 | Test weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Shape only | 14 | 0.8126 | 0.5607 | 0.7761 | 0.5560 | 0.7715 |
| GLCM texture only | 12 | 0.8019 | 0.5568 | 0.7584 | 0.5241 | 0.7512 |
| Colour only | 36 | 0.8653 | 0.7113 | 0.8558 | 0.7237 | 0.8516 |
| All handcrafted | 62 | 0.9373 | 0.8580 | 0.9116 | 0.8479 | 0.9151 |
| EfficientNet only | 1280 | 0.9396 | 0.8630 | 0.9065 | 0.8413 | 0.9098 |
| Handcrafted + EfficientNet | 1342 | 0.9570 | 0.9038 | 0.9248 | 0.8735 | 0.9277 |
| Full Hybrid saved final | 1342 | 0.9583 | 0.9071 | 0.9210 | 0.8660 | 0.9244 |

The ablation results show that individual groups are weaker than the complete handcrafted representation. Colour alone is stronger than shape or GLCM alone in this experiment. The fused configurations produce the strongest validation macro F1 and competitive final test performance. The “Handcrafted + EfficientNet” row and “Full Hybrid saved final” row are both retained because they are separate saved configurations in the ablation artifact; the paper does not silently merge them.

### D. SHAP Analysis

**Table IV. SHAP group contribution on the 300-sample explanation subset.**

| Feature group | Feature count | Mean absolute SHAP summary | Contribution summary |
|---|---:|---:|---:|
| EfficientNet-B0 | 1280 | 3.815151 | 68.5942% |
| Shape/Morphological | 14 | 0.752179 | 13.5237% |
| Colour | 36 | 0.715490 | 12.8641% |
| GLCM Texture | 12 | 0.279099 | 5.0180% |

The largest group-level SHAP summary belongs to the EfficientNet-B0 representation. This should not be interpreted as a quality percentage or probability. It indicates the relative mean absolute contribution within the analyzed explanation subset. The global top features include `shape_height`, `color_lab_b_mean`, `shape_major_axis_length`, `shape_eccentricity`, `deep_feature_766`, `shape_area`, `color_hsv_s_mean`, and `deep_feature_475`. Latent deep features are not assigned human concepts without supporting evidence.

For comparison, XGBoost gain-based group importance assigns 85.5024% to EfficientNet Deep, 5.7722% to Shape, 5.4577% to Colour, and 3.2677% to GLCM Texture. Gain importance and SHAP contribution answer different questions and should not be mixed.

### E. Error Analysis

The final test confusion matrix contains 3,100 samples and shows high performance for the dominant `0_NOR` class as well as strong performance for `5_BN` and `7_IM`. Lower F1 values occur for `2_SD` and `3_MY`. The matrix also shows cross-class errors involving `0_NOR`, `2_SD`, and `3_MY`, while some minority-class errors occur between `1_F&S` and `2_SD` and between `5_BN` and `3_MY`.

The dataset imbalance means overall accuracy is influenced strongly by the 2,000 `0_NOR` test examples. Macro F1, macro recall, and class-wise support therefore provide a more balanced view. The saved experiment compared unweighted and inverse-frequency weighted training; the unweighted hybrid model achieved the higher validation macro F1 and was selected. The evidence does not establish that weighting solves the observed class-specific errors.

## VII. Deployment and Application Support

The repository includes a Streamlit application organized into Dashboard, Analyze Rice, Batch Analysis, Prediction History, Review Queue, SHAP/Explainability, Reports, Webcam, Model Performance, System/API Status, and About Project pages. These pages are supporting deployment components rather than the central research novelty.

The application can display class probabilities, local SHAP tables, batch results, review records, CSV reports, PDF reports, saved model comparison artifacts, and system status. SQLite persistence stores prediction history and manual corrections. The REST API exposes `/health` and `/predict`, validates uploaded files, and supports optional `X-API-Key` validation when `RICE_API_KEY` is configured.

The webcam component includes a rice/non-rice gate interface, but the repository README states that real non-rice images must be added and the gate trained before the validator can be used. A fully trained and validated rice/non-rice classifier is therefore **not yet available**.

## VIII. Limitations

1. The evaluation is based on one project dataset and its predefined split; external-dataset performance is not verified.
2. The class abbreviations are preserved as raw labels because expanded meanings are not verified in the dataset report.
3. The dataset is imbalanced, and minority-class F1 values are lower than the dominant-class result.
4. EfficientNet features are latent dimensions and are not directly human-interpretable.
5. Model probabilities are not described as calibrated confidence because probability calibration was not verified.
6. The webcam rice/non-rice classifier requires real non-rice data and has not been trained and validated in the repository.
7. The saved reproducibility files report different NumPy versions, which should be reconciled before publication.
8. Hardware used for the recorded experiments is not verified from the repository.
9. A small manual inference smoke sample is not a substitute for the official test evaluation.

## IX. Future Work

Future work should include independent external validation, calibrated probabilities, collection and training of the rice/non-rice gate, robustness tests under illumination and background changes, feature-selection experiments for the 1,280-dimensional deep representation, mobile or edge optimization, and additional explanation methods. These are proposed extensions and are not reported as completed work.

## X. Conclusion

This paper presented an explainable hybrid feature-fusion framework for eight-class rice image classification. The implemented pipeline combines 14 shape descriptors, 12 GLCM texture descriptors, 36 colour descriptors, and 1,280 EfficientNet-B0 features into a 1,342-dimensional representation. After leakage-safe StandardScaler transformation, an XGBoost classifier achieved 92.0968% test accuracy, 0.8660 macro F1, and 0.9244 weighted F1 on the saved final evaluation. The hybrid model improved over the implemented handcrafted SVM and EfficientNet-only comparisons in the principal saved results, while the ablation study showed the value of combining complementary feature groups. SHAP analysis provided an evidence-based view of local and group-level model contributions, with the important distinction that contribution summaries are not class probabilities or quality percentages. The Streamlit application, persistence, reports, review workflow, and API provide practical support around the research pipeline. The conclusions remain limited to the verified dataset and artifacts, and the untrained rice/non-rice gate and absent external validation should be addressed before claims of broader deployment reliability.

## References

**[REFERENCES REQUIRE EXTERNAL VERIFICATION]**

The repository does not contain verified bibliographic references. Before submission, add externally verified IEEE-style references for: the GrainSet dataset; EfficientNet; XGBoost; SHAP; GLCM texture analysis; rice quality or defect classification; and relevant hybrid feature-fusion studies. Do not add author names, titles, venues, years, DOIs, or URLs without checking reliable academic sources.
