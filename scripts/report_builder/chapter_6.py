"""
chapter_6.py
============
Generates Chapter 6: Results and Discussion for the B.Tech project report:
- 6.1 Dataset Summary and Partitioning
- 6.2 Image Preprocessing Verification
- 6.3 Handcrafted Feature Extraction Integrity
- 6.4 Baseline Model Evaluation
- 6.5 EfficientNet-B0 + XGBoost Baseline
- 6.6 Final Hybrid Model Evaluation
- 6.7 Class-Wise Classification Dynamics
- 6.8 Confusion Matrix Analysis
- 6.9 Systematic Feature Ablation Study
- 6.10 SHAP Interpretability and Group Contributions
- 6.11 XGBoost Split-Gain vs. SHAP Attributions
- 6.12 Detailed Error and Boundary Leakage Analysis
- 6.13 Deployment Stack and Interactive Application
- 6.14 Output Probability Interpretation and Calibration Nuances
"""

from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from .styles import (
    FONT_NAME, COLOR_BLACK, COLOR_NAVY, COLOR_DARK_GRAY,
    add_chapter_heading, add_section_heading, add_subsection_heading,
    add_subsubsection_heading, add_body_p, add_bullet_p, add_numbered_p,
    add_callout_box, add_table_data, add_figure
)

FIG_DIR = Path("c:/Rice classifier final project/paper/figures")

def build_chapter_6(doc):
    add_chapter_heading(doc, "CHAPTER 6", "RESULTS AND DISCUSSION")
    
    # 6.1 DATASET SUMMARY
    add_section_heading(doc, "6.1 DATASET SUMMARY AND PARTITIONING")
    p_data1 = (
        "All experiments in this research were conducted on the GrainSet Rice Dataset, comprising 30,962 single-grain digital "
        "images partitioned into strictly non-overlapping, zero-leakage training, validation, and test splits. The original "
        "partitioning manifest was rigorously preserved to guarantee experimental reproducibility: 24,767 images (79.99%) for "
        "training, 3,095 images (10.00%) for validation, and 3,100 images (10.01%) for untouched final testing. An automated hash audit "
        "confirmed zero filename or image overlap across the three partitions."
    )
    add_body_p(doc, p_data1, indent=0.3)

    p_data2 = (
        "The dataset comprises eight distinct classes represented by their official raw labels: 0_NOR, 1_F&S, 2_SD, 3_MY, 4_AP, "
        "5_BN, 6_UN, and 7_IM. To maintain strict scientific authenticity, these raw identifiers are preserved without inventing "
        "unverified semantic expansions. A dominant characteristic of the dataset is severe natural class imbalance: the normal class "
        "(0_NOR) accounts for 15,980 samples in the training split (64.52%), 2,020 samples in the validation split (65.27%), and "
        "2,000 samples in the test split (64.52%). Table 6.1 details the class distributions across all splits."
    )
    add_body_p(doc, p_data2, indent=0.3)

    tbl_ds_headers = ["CLASS ID", "RAW CLASS LABEL", "TRAIN COUNT (%)", "VAL COUNT (%)", "TEST COUNT (%)", "TOTAL IMAGES"]
    tbl_ds_data = [
        ["0", "0_NOR", "15,980 (64.52%)", "2,020 (65.27%)", "2,000 (64.52%)", "20,000 (64.59%)"],
        ["1", "1_F&S", "1,191 (4.81%)", "145 (4.68%)", "150 (4.84%)", "1,486 (4.80%)"],
        ["2", "2_SD", "1,194 (4.82%)", "156 (5.04%)", "150 (4.84%)", "1,500 (4.84%)"],
        ["3", "3_MY", "1,223 (4.94%)", "127 (4.10%)", "150 (4.84%)", "1,500 (4.84%)"],
        ["4", "4_AP", "1,208 (4.88%)", "142 (4.59%)", "150 (4.84%)", "1,500 (4.84%)"],
        ["5", "5_BN", "1,198 (4.84%)", "143 (4.62%)", "150 (4.84%)", "1,491 (4.82%)"],
        ["6", "6_UN", "1,185 (4.78%)", "150 (4.85%)", "150 (4.84%)", "1,485 (4.80%)"],
        ["7", "7_IM", "1,588 (6.41%)", "212 (6.85%)", "200 (6.45%)", "2,000 (6.46%)"],
        ["TOTAL", "8 Classes", "24,767 (100.0%)", "3,095 (100.0%)", "3,100 (100.0%)", "30,962 (100.0%)"]
    ]
    add_table_data(doc, tbl_ds_headers, tbl_ds_data, col_widths=[0.8, 1.2, 1.4, 1.3, 1.3, 1.2], caption="Table 6.1: GrainSet Rice Dataset Distribution across Partitioned Zero-Leakage Splits")

    add_callout_box(
        doc,
        "METHODOLOGICAL SIGNIFICANCE OF CLASS IMBALANCE: Because the majority class (0_NOR) represents ~64.5% of the evaluation data, "
        "standard classification accuracy is an insufficient metric. A naive trivial classifier assigning all samples to 0_NOR would achieve "
        "64.52% accuracy while completely failing to detect any defective grains. Therefore, Macro-Averaged F1-Score (Macro F1)—which calculates "
        "the arithmetic mean of individual class F1-scores with equal weight—serves as the primary performance benchmark alongside accuracy.",
        title="EVALUATION METRIC JUSTIFICATION"
    )

    # 6.2 PREPROCESSING RESULTS
    add_section_heading(doc, "6.2 IMAGE PREPROCESSING EVALUATION")
    p_prep_res = (
        "The automated OpenCV preprocessing pipeline was validated through a controlled verification experiment across 20 representative "
        "grain samples selected evenly across all eight defect categories. Across all 20 test instances, the pipeline achieved 20/20 "
        "successful grain isolations (100% success rate, 0 segmentation failures). The observed image dimensions prior to preprocessing "
        "ranged from 206×135 to 328×301 pixels, which were reliably converted to standardized 224×224×3 pixel arrays. The average "
        "preprocessing latency was measured at 5.44 ms per image. This metric is documented as a verified sample validation experiment "
        "rather than a generalized claim across industrial line setups."
    )
    add_body_p(doc, p_prep_res, indent=0.3)

    add_figure(
        doc,
        FIG_DIR / "fig3_preprocessing_grid.png",
        "Figure 6.2.1: Preprocessing Verification Summary Grid Displaying Representative Sample Grains across All Eight Classes "
        "undergoing Denoising, Otsu Segmentation, Morphological Cleaning, and Standardized 224×224 Cropping.",
        width_in=5.8
    )

    # 6.3 HANDCRAFTED FEATURE RESULTS
    add_section_heading(doc, "6.3 HANDCRAFTED FEATURE EXTRACTION INTEGRITY")
    p_hc_res = (
        "Batch extraction of the 62 handcrafted descriptors was executed across the entire 30,962-image dataset. Feature integrity checks "
        "confirmed flawless execution across all partitions (24,767 train, 3,095 val, 3,100 test), recording 30,962/30,962 successful "
        "extractions (0 failures). Rigorous data hygiene auditing verified that the resulting feature matrix contained 0 NaN values, "
        "0 infinite values, and 0 constant (zero-variance) columns. The total execution time for the full 30,962-image corpus was 126.2 "
        "seconds, representing an impressive extraction throughput of 245.4 images per second on the multi-threaded CPU workstation."
    )
    add_body_p(doc, p_hc_res, indent=0.3)

    # 6.4 BASELINE RESULTS
    add_section_heading(doc, "6.4 BASELINE MODEL EVALUATION")
    p_base = (
        "To rigorously contextualize the contribution of hybrid feature fusion, three foundational architectures were evaluated "
        "under identical test conditions on the untouched 3,100-sample test split: (1) Baseline 1: RBF Support Vector Machine operating "
        "on 62 handcrafted features; (2) Baseline 2: XGBoost operating exclusively on 1,280 EfficientNet-B0 deep features; and (3) The "
        "Proposed Hybrid Framework: XGBoost operating on 1,342 fused features. Table 6.2 summarizes this comparative benchmark."
    )
    add_body_p(doc, p_base, indent=0.3)

    tbl_bench_headers = ["MODEL ARCHITECTURE", "FEATURE SET", "DIMS", "ACCURACY", "MACRO PREC", "MACRO REC", "MACRO F1", "WEIGHTED F1"]
    tbl_bench_data = [
        ["Baseline 1: SVM Classifier", "Handcrafted Only", "62", "91.16%", "82.06%", "88.60%", "0.8494", "0.9161"],
        ["Baseline 2: Deep XGBoost", "EfficientNet-B0 Only", "1,280", "90.65%", "81.80%", "86.79%", "0.8399", "0.9099"],
        ["Proposed: Hybrid XGBoost", "Fused (Handcrafted + Deep)", "1,342", "92.10%", "84.54%", "89.40%", "0.8660", "0.9244"],
        ["Delta (Hybrid vs. SVM)", "Fused vs. Handcrafted", "+1,280", "+0.94%", "+2.48%", "+0.80%", "+1.66%", "+0.83%"],
        ["Delta (Hybrid vs. Deep)", "Fused vs. Deep", "+62", "+1.45%", "+2.74%", "+2.61%", "+2.61%", "+1.45%"]
    ]
    add_table_data(doc, tbl_bench_headers, tbl_bench_data, col_widths=[1.5, 1.2, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8], caption="Table 6.2: Benchmark Model Performance Evaluation on 3,100 Test Samples")

    add_figure(
        doc,
        FIG_DIR / "fig5_model_comparison_bars.png",
        "Figure 6.4.1: Benchmark Model Comparison Chart: Handcrafted SVM vs. EfficientNet-B0 XGBoost vs. Proposed Hybrid XGBoost "
        "across Test Accuracy, Macro F1, and Weighted F1.",
        width_in=5.6
    )

    # 6.5 EFFICIENTNET-B0 + XGBOOST
    add_subsection_heading(doc, "6.5 EfficientNet-B0 + XGBoost Baseline Analysis")
    p_eff_xgb = (
        "A critical empirical discovery from Table 6.2 is that deep features alone do NOT surpass handcrafted features. Operating on "
        "1,280 EfficientNet-B0 latent embeddings, XGBoost achieved a validation accuracy of 93.99% and validation Macro F1 of 0.8646. "
        "However, on the untouched test split, performance dropped to 90.65% accuracy and 0.8399 Macro F1 (Weighted F1: 0.9099). "
        "In comparison, the classical SVM operating on only 62 handcrafted features achieved 91.16% test accuracy and 0.8494 Macro F1. "
        "This empirical outcome demonstrates that while pretrained convolutional networks capture broad visual hierarchies, global pooling "
        "dilutes sharp morphological aspect boundaries and exact dimensional ratios. This finding provides direct, unambiguous empirical "
        "justification for fusing handcrafted domain descriptors with deep neural embeddings."
    )
    add_body_p(doc, p_eff_xgb, indent=0.3)

    # 6.6 FINAL HYBRID MODEL
    add_subsection_heading(doc, "6.6 Final Hybrid Model Performance")
    p_hyb_res = (
        "The proposed hybrid framework, combining 62 handcrafted features with 1,280 deep embeddings under an optimized XGBoost classifier, "
        "achieved the highest performance across all evaluation splits. On the validation split (3,095 images), the model achieved 95.83% "
        "accuracy, 0.9071 Macro F1, and 0.9577 Weighted F1. On the untouched 3,100-sample test split, the final serialized hybrid model "
        "achieved a headline test accuracy of 92.10% (exact: 92.0968%), Macro Precision of 0.8454 (84.54%), Macro Recall of 0.8940 (89.40%), "
        "Macro F1 of 0.8660 (86.60%), and Weighted F1 of 0.9244 (92.44%). Out of 3,100 test samples, the model correctly classified exactly "
        "2,855 kernels. Table 6.3 details the validation and test performance metrics."
    )
    add_body_p(doc, p_hyb_res, indent=0.3)

    tbl_hyb_headers = ["EVALUATION SPLIT", "SUPPORT", "ACCURACY", "MACRO PRECISION", "MACRO RECALL", "MACRO F1", "WEIGHTED F1"]
    tbl_hyb_data = [
        ["Validation Split", "3,095 images", "95.83%", "88.62%", "93.18%", "0.9071", "0.9577"],
        ["Test Split (Untouched)", "3,100 images", "92.10%", "84.54%", "89.40%", "0.8660", "0.9244"]
    ]
    add_table_data(doc, tbl_hyb_headers, tbl_hyb_data, col_widths=[1.5, 1.1, 1.0, 1.1, 1.0, 1.0, 1.0], caption="Table 6.3: Final Hybrid XGBoost Model Validation and Test Performance Summary")

    # 6.7 CLASS-WISE PERFORMANCE
    add_section_heading(doc, "6.7 CLASS-WISE CLASSIFICATION DYNAMICS")
    p_cw = (
        "The classification fidelity of the proposed hybrid model varies across the eight categories due to distinct physical "
        "and chromatic characteristics. Table 6.4 catalogues the exact per-class metrics evaluated on the 3,100 test samples."
    )
    add_body_p(doc, p_cw, indent=0.3)

    tbl_cw_headers = ["CLASS ID", "RAW LABEL", "PRECISION", "RECALL", "F1-SCORE", "TEST SUPPORT", "DOMINANT MISCLASSIFICATION PATTERN"]
    tbl_cw_data = [
        ["0", "0_NOR", "98.58%", "93.70%", "0.9608", "2,000", "52 predicted as 3_MY, 42 predicted as 2_SD"],
        ["1", "1_F&S", "78.24%", "88.67%", "0.8313", "150", "11 predicted as 2_SD, 2 as 6_UN, 2 as 3_MY"],
        ["2", "2_SD", "67.03%", "81.33%", "0.7349", "150", "26 predicted as 1_F&S, 2 as 6_UN"],
        ["3", "3_MY", "64.40%", "82.00%", "0.7214", "150", "13 predicted as 0_NOR, 6 as 1_F&S, 4 as 6_UN"],
        ["4", "4_AP", "93.43%", "85.33%", "0.8920", "150", "13 predicted as 0_NOR, 4 as 6_UN, 3 as 2_SD"],
        ["5", "5_BN", "96.53%", "92.67%", "0.9456", "150", "7 predicted as 3_MY, 3 as 7_IM, 1 as 4_AP"],
        ["6", "6_UN", "79.66%", "94.00%", "0.8624", "150", "4 predicted as 3_MY, 2 as 4_AP, 2 as 1_F&S"],
        ["7", "7_IM", "98.48%", "97.50%", "0.9799", "200", "2 predicted as 3_MY, 2 as 5_BN, 1 as 2_SD"],
        ["MACRO", "Average", "84.54%", "89.40%", "0.8660", "3,100", "Arithmetic mean across all 8 classes"],
        ["WEIGHTED", "Average", "92.74%", "92.10%", "0.9244", "3,100", "Support-weighted mean across 3,100 instances"]
    ]
    add_table_data(doc, tbl_cw_headers, tbl_cw_data, col_widths=[0.6, 1.0, 0.9, 0.9, 0.8, 0.9, 2.3], caption="Table 6.4: Class-Wise Classification Dynamics for the Final Hybrid XGBoost Model")

    add_figure(
        doc,
        FIG_DIR / "fig10_per_class_f1_comparison.png",
        "Figure 6.7.1: Per-Class F1-Score Comparison across Benchmark Models: Handcrafted SVM, EfficientNet-B0 XGBoost, "
        "and Proposed Hybrid Framework.",
        width_in=5.8
    )

    # 6.8 CONFUSION MATRIX
    add_section_heading(doc, "6.8 CONFUSION MATRIX ANALYSIS")
    p_cm = (
        "To evaluate error distributions, the complete 8×8 confusion matrix evaluated on the 3,100 test samples is presented in "
        "Table 6.5. Exactly 2,855 samples lie along the principal diagonal, representing correct classifications (92.0968% accuracy)."
    )
    add_body_p(doc, p_cm, indent=0.3)

    tbl_cm_headers = ["TRUE CLASS", "0_NOR", "1_F&S", "2_SD", "3_MY", "4_AP", "5_BN", "6_UN", "7_IM", "TOTAL"]
    tbl_cm_data = [
        ["0_NOR", "1874", "3", "42", "52", "5", "0", "24", "0", "2000"],
        ["1_F&S", "1", "133", "11", "2", "0", "1", "2", "0", "150"],
        ["2_SD", "0", "26", "122", "0", "0", "0", "2", "0", "150"],
        ["3_MY", "13", "6", "3", "123", "1", "0", "4", "0", "150"],
        ["4_AP", "13", "0", "3", "1", "128", "1", "4", "0", "150"],
        ["5_BN", "0", "0", "0", "7", "1", "139", "0", "3", "150"],
        ["6_UN", "0", "2", "0", "4", "2", "1", "141", "0", "150"],
        ["7_IM", "0", "0", "1", "2", "0", "2", "0", "195", "200"],
        ["TOTAL PRED", "1901", "170", "182", "191", "137", "144", "177", "198", "3100"]
    ]
    add_table_data(doc, tbl_cm_headers, tbl_cm_data, col_widths=[1.1, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7], caption="Table 6.5: Complete 8×8 Test Confusion Matrix on 3,100 Evaluation Samples")

    add_figure(
        doc,
        FIG_DIR / "fig4_confusion_matrix_raw.png",
        "Figure 6.8.1: Raw Count Confusion Matrix of the Proposed Hybrid XGBoost Model on 3,100 Test Samples.",
        width_in=5.4
    )

    add_figure(
        doc,
        FIG_DIR / "fig4_confusion_matrix_normalized.png",
        "Figure 6.8.2: Normalized Confusion Matrix Demonstrating Class-Recall Rates across All Eight Grain Categories.",
        width_in=5.4
    )

    # 6.9 ABLATION STUDY
    add_section_heading(doc, "6.9 SYSTEMATIC FEATURE ABLATION STUDY")
    p_abl1 = (
        "To rigorously quantify the individual and collective contributions of each feature modality, a systematic seven-configuration "
        "ablation study was conducted. All configurations were trained on identical training partitions (24,767 samples) and evaluated on "
        "identical validation (3,095 samples) and test (3,100 samples) sets using identical XGBoost hyperparameters (n_estimators=100, "
        "max_depth=4, learning_rate=0.1). Table 6.6 documents the complete experimental ablation breakdown."
    )
    add_body_p(doc, p_abl1, indent=0.3)

    tbl_abl_headers = ["ID", "FEATURE CONFIGURATION", "DIMS", "TRAIN TIME", "VAL ACC", "VAL F1", "TEST ACC", "TEST PREC", "TEST REC", "TEST F1", "WTD F1"]
    tbl_abl_data = [
        ["A", "Shape / Morphology Only", "14", "0.73 s", "0.8126", "0.5607", "0.7761", "0.5671", "0.5558", "0.5560", "0.7715"],
        ["B", "GLCM Texture Only", "12", "0.63 s", "0.8019", "0.5568", "0.7584", "0.5505", "0.5219", "0.5241", "0.7512"],
        ["C", "Colour Statistics Only", "36", "1.09 s", "0.8653", "0.7113", "0.8558", "0.7435", "0.7123", "0.7237", "0.8516"],
        ["D", "All Handcrafted (A+B+C)", "62", "2.25 s", "0.9373", "0.8580", "0.9116", "0.8258", "0.8775", "0.8479", "0.9151"],
        ["E", "EfficientNet-B0 Only", "1,280", "170.23 s", "0.9396", "0.8630", "0.9065", "0.8187", "0.8686", "0.8413", "0.9098"],
        ["F", "Handcrafted + Deep (Retrained)", "1,342", "173.10 s", "0.9570", "0.9038", "0.9248", "0.8515", "0.9018", "0.8735", "0.9277"],
        ["G", "Full Hybrid (Saved Final Model)", "1,342", "129.19 s", "0.9583", "0.9071", "0.9210", "0.8454", "0.8940", "0.8660", "0.9244"]
    ]
    add_table_data(doc, tbl_abl_headers, tbl_abl_data, col_widths=[0.4, 1.8, 0.5, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7], caption="Table 6.6: Systematic Seven-Configuration Feature Ablation Study Results")

    add_callout_box(
        doc,
        "EXPERIMENTAL DISTINCTION BETWEEN CONFIGURATIONS F AND G: Configurations F and G represent two separate experimental runs "
        "of the 1,342-dimensional hybrid architecture. Configuration F represents a retrained ablation run (achieving 92.48% test accuracy "
        "and 0.8735 Macro F1). Configuration G represents the serialized, final deployed model artifact (achieving 92.10% test accuracy and "
        "0.8660 Macro F1) utilized for all headline benchmarks, SHAP attributions, confusion matrices, and web deployment. In strict adherence "
        "to data hygiene, F and G are documented as distinct runs and are NEVER silently merged.",
        title="METHODOLOGICAL DISCLOSURE: CONFIGURATION F VS. G"
    )

    add_figure(
        doc,
        FIG_DIR / "fig6_ablation_comparison.png",
        "Figure 6.9.1: Systematic Feature Ablation Study: Validation and Test Performance across 7 Feature Configurations.",
        width_in=5.8
    )

    p_abl_disc = (
        "Key scientific insights derived from Table 6.6 include:\n"
        "1. Colour is the Strongest Handcrafted Modality: Configuration C (36 colour features) achieves 85.58% test accuracy and 0.7237 Macro F1, "
        "substantially outperforming Shape alone (0.5560 F1) and Texture alone (0.5241 F1). This confirms that fungal spots, chalky endosperms, "
        "and pericarp maturation are strongly reflected in chromatic distributions across LAB, HSV, and RGB channels.\n"
        "2. Handcrafted Synergy: Combining Shape, Texture, and Colour (Configuration D, 62 features) produces a massive jump to 0.8479 Macro F1 "
        "(+12.42 percentage points over colour alone), proving that geometric and textural boundaries resolve chromatic ambiguities.\n"
        "3. Deep Feature Parity: Configuration E (1,280 EfficientNet embeddings) achieves 0.8413 Macro F1, which is on par with the 62 handcrafted "
        "features (0.8479 F1) despite having over 20× more dimensions.\n"
        "4. The Fusion Dividend: Concatenating handcrafted and deep descriptors (Configurations F & G) yields the highest performance across all "
        "metrics (0.8660-0.8735 Macro F1), empirically confirming that explicit physical descriptors and deep latent manifolds provide "
        "mutually complementary discriminatory information."
    )
    add_body_p(doc, p_abl_disc, indent=0.3)

    # 6.10 SHAP RESULTS
    add_section_heading(doc, "6.10 SHAP INTERPRETABILITY AND GROUP CONTRIBUTIONS")
    p_shap_res = (
        "To deliver operational transparency, SHAP analysis was conducted on an evaluation subset of 300 test samples using TreeExplainer "
        "(reproducibility seed random_state = 42). The mean absolute SHAP values were aggregated across the four feature families to quantify "
        "the relative attribution share of each modality. Table 6.7 summarizes the SHAP group contributions."
    )
    add_body_p(doc, p_shap_res, indent=0.3)

    tbl_shap_headers = ["FEATURE GROUP", "FEATURE COUNT", "SUM MEAN |SHAP|", "ATTRIBUTION SHARE (%)", "TOP-RANKED INDIVIDUAL DESCRIPTORS"]
    tbl_shap_data = [
        ["EfficientNet-B0 Deep", "1,280", "3.81515", "68.5942%", "deep_feature_766 (5th), deep_feature_475 (8th), deep_feature_203 (10th)"],
        ["Shape / Morphological", "14", "0.75218", "13.5237%", "shape_height (1st), shape_major_axis_length (3rd), shape_eccentricity (4th)"],
        ["Multichannel Colour", "36", "0.71549", "12.8641%", "color_lab_b_mean (2nd), color_hsv_s_mean (7th), color_lab_b_std (9th)"],
        ["GLCM Texture", "12", "0.27910", "5.0180%", "glcm_contrast_std (14th), glcm_contrast_mean (17th), glcm_homogeneity_mean (20th)"],
        ["TOTAL FUSED VECTOR", "1,342", "5.56192", "100.0000%", "Top 4 features include 3 Handcrafted: height, lab_b_mean, major_axis"]
    ]
    add_table_data(doc, tbl_shap_headers, tbl_shap_data, col_widths=[1.5, 0.8, 1.1, 1.3, 2.5], caption="Table 6.7: SHAP Feature Group Contribution Summary across 300 Test Instances")

    add_figure(
        doc,
        FIG_DIR / "fig7a_shap_top20_bar.png",
        "Figure 6.10.1: Top 20 Most Influential Features Ranked by Mean Absolute SHAP Attribution on 300 Test Instances.",
        width_in=5.6
    )

    add_figure(
        doc,
        FIG_DIR / "fig7b_shap_beeswarm_summary.png",
        "Figure 6.10.2: SHAP Beeswarm Summary Plot Displaying Feature Value Magnitudes (High vs. Low) and Directional Impact on Class Log-Odds.",
        width_in=5.8
    )

    add_figure(
        doc,
        FIG_DIR / "fig8_shap_group_contribution.png",
        "Figure 6.10.3: SHAP Feature Group Attribution Share: EfficientNet-B0 (68.59%), Shape (13.52%), Colour (12.86%), and Texture (5.02%).",
        width_in=5.2
    )

    p_top_shap = (
        "The verified ranking of the top 20 individual features evaluated by mean absolute SHAP value is:\n"
        "1. shape_height (0.1883) — Dominant physical descriptor capturing kernel vertical elongation.\n"
        "2. color_lab_b_mean (0.1763) — CIELAB b* yellow-blue balance, sensitive to endosperm opalescence and fungal jaundice.\n"
        "3. shape_major_axis_length (0.1498) — Primary elliptical axis, segregating whole kernels from broken pieces.\n"
        "4. shape_eccentricity (0.1295) — Degree of kernel slender ratio.\n"
        "5. deep_feature_766 (0.1084) — High-level convolutional pattern learned by EfficientNet-B0.\n"
        "6. shape_area (0.0993) — Projected kernel size.\n"
        "7. color_hsv_s_mean (0.0952) — Chromatic purity and saturation.\n"
        "8. deep_feature_475 (0.0857) — Deep latent representation.\n"
        "9. color_lab_b_std (0.0807) — Spatial variance of yellow-blue tones across the grain surface.\n"
        "10. deep_feature_203 (0.0722) — Deep latent representation.\n"
        "11. deep_feature_484 (0.0667) — Deep latent representation.\n"
        "12. deep_feature_262 (0.0610) — Deep latent representation.\n"
        "13. deep_feature_104 (0.0552) — Deep latent representation.\n"
        "14. glcm_contrast_std (0.0533) — Textural variation in surface fissures.\n"
        "15. color_lab_a_mean (0.0532) — CIELAB a* red-green axis, detecting Fusarium pink spots.\n"
        "16. deep_feature_574 (0.0487) — Deep latent representation.\n"
        "17. glcm_contrast_mean (0.0434) — Baseline surface roughness.\n"
        "18. shape_aspect_ratio (0.0421) — Height-to-width ratio.\n"
        "19. deep_feature_979 (0.0392) — Deep latent representation.\n"
        "20. glcm_homogeneity_mean (0.0377) — Vitreous surface smoothness."
    )
    add_body_p(doc, p_top_shap, indent=0.2)

    # 6.11 FEATURE IMPORTANCE VS SHAP
    add_section_heading(doc, "6.11 XGBOOST SPLIT-GAIN VS. SHAP ATTRIBUTIONS")
    p_gain_shap = (
        "A vital methodological contribution of this research is the empirical and theoretical distinction between model-based "
        "training split-gain importance and post-hoc instance-level SHAP attributions. Table 6.8 contrasts the two paradigms."
    )
    add_body_p(doc, p_gain_shap, indent=0.3)

    tbl_imp_headers = ["FEATURE GROUP", "XGBOOST TREE-GAIN SHARE (%)", "SHAP ATTRIBUTION SHARE (%)", "DISCREPANCY RATIONALE"]
    tbl_imp_data = [
        ["EfficientNet-B0 Deep (1280)", "85.5024%", "68.5942%", "Deep embeddings provide 1,280 continuous orthogonal axes heavily utilized during training tree splits."],
        ["Shape / Morphological (14)", "5.7722%", "13.5237%", "Handcrafted geometric thresholds anchor real test instances (e.g. height, major axis rank top 3 in SHAP)."],
        ["Multichannel Colour (36)", "5.4577%", "12.8641%", "Colour channels (lab_b_mean, hsv_s_mean) provide critical marginal contributions on test specimens."],
        ["GLCM Texture (12)", "3.2677%", "5.0180%", "Fine surface texture resolves borderline ambiguous classifications on real test kernels."]
    ]
    add_table_data(doc, tbl_imp_headers, tbl_imp_data, col_widths=[1.5, 1.5, 1.5, 2.5], caption="Table 6.8: Comparative Breakdown: XGBoost Tree-Gain Split Importance vs. SHAP Attribution")

    add_figure(
        doc,
        FIG_DIR / "fig9_gain_vs_shap_comparison.png",
        "Figure 6.11.1: Dual Comparison of XGBoost Tree-Gain Split Importance vs. Instance-Level SHAP Attribution Share.",
        width_in=5.6
    )

    p_gain_disc = (
        "The scientific explanation for this discrepancy is profound: XGBoost Gain measures how much each feature split reduced "
        "the training objective loss summed across all trees during training. Because the 1,280 deep embeddings offer high-dimensional "
        "continuous manifolds, the greedy tree-building algorithm disproportionately selects them to partition training clusters (85.50% gain). "
        "Conversely, SHAP evaluates the marginal Shapley contribution of each feature to the output log-odds on real, unseen test instances. "
        "In real testing, physical geometric boundaries (shape height, major axis, eccentricity) and chromatic metrics account for nearly "
        "one-third (31.41%) of the explanatory mass, proving that the model actively relies on physical handcrafted features to finalize "
        "boundary decisions."
    )
    add_body_p(doc, p_gain_disc, indent=0.3)

    # 6.12 ERROR ANALYSIS
    add_section_heading(doc, "6.12 DETAILED ERROR AND BOUNDARY LEAKAGE ANALYSIS")
    p_err1 = (
        "A rigorous audit of the 245 misclassified test samples (7.90% error rate) reveals two dominant error modalities:\n"
        "1. Mutual Morphological Confusion between 1_F&S and 2_SD: The largest pairwise confusion in the dataset occurs between 1_F&S "
        "and 2_SD: 26 true 2_SD samples were predicted as 1_F&S, and 11 true 1_F&S samples were predicted as 2_SD (37 samples total). "
        "Inspection confirms that severe seed cracks (2_SD) often create dark transverse shadows that visually mimic elongated fungal lesions (1_F&S).\n"
        "2. Majority Class Boundary Leakage: Because the normal class (0_NOR) contains 2,000 test instances, even a minor classification "
        "leakage (52 samples into 3_MY and 42 samples into 2_SD) severely depresses the precision of the minority classes: 3_MY precision "
        "drops to 64.40% and 2_SD precision drops to 67.03%, despite both classes maintaining robust recalls of 82.00% and 81.33% respectively. "
        "Conversely, classes with distinct macroscopic geometry—such as 7_IM (0.9799 F1), 0_NOR (0.9608 F1), and 5_BN (0.9456 F1)—exhibit "
        "near-perfect separation."
    )
    add_body_p(doc, p_err1, indent=0.3)

    # 6.13 DEPLOYMENT
    add_section_heading(doc, "6.13 DEPLOYMENT STACK AND INTERACTIVE APPLICATION")
    p_dep1 = (
        "While the primary academic contribution is the verified feature-fusion framework and explainability analysis, the project "
        "incorporates a comprehensive operational deployment stack to demonstrate practical industrial feasibility:"
    )
    add_body_p(doc, p_dep1, indent=0.3)

    dep_points = [
        ("Streamlit Multi-Page Graphical Dashboard: ", "Provides an operator-facing web application featuring single-grain image upload, real-time segmentation previews, class probability displays, and deep visual saliency heatmaps."),
        ("Explainability Interface: ", "Displays interactive SHAP attribution waterfalls separating positive contributors (supporting the prediction) from negative contributors (opposing it), supported by an agronomic feature dictionary."),
        ("Human-in-the-Loop Review Queue: ", "Predictions with maximum softmax probability below a configurable threshold (<0.75) are automatically routed to an audit queue, allowing human inspectors to review, verify, or manually correct ambiguous classifications."),
        ("SQLite Historical Logging: ", "All inferences, image paths, predicted labels, confidence values, review statuses, and timestamps are transactionally persisted into `results/rice_quality.db`."),
        ("Batch Analysis & CSV Export: ", "Enables bulk ingestion of grain image folders, computing batch defect statistics and exporting structured CSV reports for enterprise inventory integration."),
        ("Automated Commercial PDF Reports: ", "Generates publication-quality commercial inspection certificates containing grain imagery, defect breakdowns, and grading compliance."),
        ("Headless FastAPI REST Service: ", "Exposes high-throughput endpoints (`/predict`, `/predict/batch`, `/explain`) enabling automated integration with optical sortation chutes and programmable logic controllers (PLCs).")
    ]
    for title, desc in dep_points:
        add_bullet_p(doc, desc, bold_prefix=title)

    # 6.14 MODEL OUTPUT / PROBABILITY
    add_section_heading(doc, "6.14 OUTPUT PROBABILITY INTERPRETATION AND CALIBRATION NUANCES")
    p_prob_caveat = (
        "In industrial deployment and operational reports, model outputs must be communicated with strict technical precision. "
        "For example, when the system outputs:\n\n"
        "   Predicted Class: 2_SD  |  Model Probability: 75.63%\n\n"
        "the metric '75.63%' denotes strictly the uncalibrated softmax probability generated by the XGBoost multi:softprob objective function. "
        "It MUST NOT be interpreted as:\n"
        "• 75.63% physical grain quality or purity;\n"
        "• 75.63% extent of physical grain defect;\n"
        "• 75.63% mathematical certainty that the prediction is ground-truth correct.\n\n"
        "Because post-hoc probability calibration (such as Platt scaling or isotonic regression) was not independently fitted on the test split, "
        "raw softmax outputs reflect relative ranking scores rather than true frequentist probabilities. Operational documentation strictly "
        "enforces the verified phrasing: 'Predicted Class: 2_SD; Model Probability: 75.63%' to prevent erroneous commercial assumptions."
    )
    add_body_p(doc, p_prob_caveat, indent=0.3)
