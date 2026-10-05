"""
chapter_5.py
============
Generates Chapter 5: System Implementation for the B.Tech project report:
- 5.1 Proposed System
  - 5.1.1 Image Preprocessing and Segmentation
  - 5.1.2 Handcrafted Feature Extraction Pipeline (14 shape, 12 GLCM, 36 colour = 62)
  - 5.1.3 GLCM Texture Features
  - 5.1.4 Multichannel Colour Features
  - 5.1.5 SVM Baseline Model
  - 5.1.6 StandardScaler and Data Leakage Prevention
  - 5.1.7 EfficientNet-B0 Deep Feature Backbone
  - 5.1.8 Feature Fusion Architecture
  - 5.1.9 Extreme Gradient Boosting (XGBoost) Classifier
  - 5.1.10 SHAP Explainability Framework
- 5.2 System Architecture
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

def build_chapter_5(doc):
    add_chapter_heading(doc, "CHAPTER 5", "SYSTEM IMPLEMENTATION")
    
    # 5.1 PROPOSED SYSTEM
    add_section_heading(doc, "5.1 PROPOSED SYSTEM")
    p_prop = (
        "The proposed explainable hybrid feature-fusion framework is engineered as a multi-stage, modular computer-vision "
        "pipeline designed to overcome the individual deficiencies of purely handcrafted and purely deep learning approaches. "
        "The overarching architectural flow adheres strictly to the following conceptual pipeline:\n\n"
        "Input Raw Rice Grain Image\n"
        "         │\n"
        "         ▼\n"
        "Preprocessing & Segmentation (Denoising, Otsu Thresholding, Morphological Cleaning, Isotropic 224×224 Standard.)\n"
        "         │\n"
        "         ├───────────────────────────────────────────────┐\n"
        "         ▼                                               ▼\n"
        "Branch 1: Handcrafted Feature Extraction (62-D)         Branch 2: EfficientNet-B0 Deep Backbone (1280-D)\n"
        "• 14 Geometric / Morphological Descriptors             • Excised Classification Head (Identity)\n"
        "• 12 GLCM Multi-Offset Texture Descriptors             • ImageNet Pretrained Weights\n"
        "• 36 Multichannel Colour Statistics (RGB, HSV, LAB)     • Global Average Pooling Latent Embeddings\n"
        "         │                                               │\n"
        "         └───────────────────────┬───────────────────────┘\n"
        "                                 ▼\n"
        "                     Feature Concatenation (1,342-D Fused Representation)\n"
        "                                 │\n"
        "                                 ▼\n"
        "                     StandardScaler Transformation (Fitted Strictly on Training Split)\n"
        "                                 │\n"
        "                                 ▼\n"
        "                     Extreme Gradient Boosting (XGBoost) Multiclass Classifier\n"
        "                                 │\n"
        "                                 ▼\n"
        "                     Eight-Class Output Prediction & Softmax Probabilities\n"
        "                                 │\n"
        "                                 ▼\n"
        "                     TreeExplainer SHAP Local & Global Attributions\n"
        "                                 │\n"
        "                                 ▼\n"
        "                     Interactive Deployment & Decision Support (Streamlit, SQLite, REST API)"
    )
    add_body_p(doc, p_prop, indent=0.0)

    # 5.1.1 Image Preprocessing
    add_subsection_heading(doc, "5.1.1 Image Preprocessing and Grain Segmentation")
    p_prep1 = (
        "Digital images of agricultural specimens frequently exhibit high-frequency sensor noise, uneven ambient lighting, "
        "and minor background artifacts. In raw GrainSet imagery, captured grain dimensions vary substantially, exhibiting "
        "observed dimensions ranging from 206×135 pixels to 328×301 pixels (mean: 261.5×222.2 pixels). Directly ingesting "
        "raw unaligned images into feature extractors causes severe dimensional inconsistency and background noise contamination. "
        "To establish a pristine, standardized input representation, an OpenCV-based preprocessing pipeline is executed through "
        "the following sequential stages:"
    )
    add_body_p(doc, p_prep1, indent=0.3)

    stages = [
        ("RGB and Multichannel Conversion: ", "The input BGR image is loaded and converted to RGB for visual fidelity, Grayscale for thresholding, HSV (Hue, Saturation, Value) for chromatic segmentation, and CIELAB (L*, a*, b*) for perceptual lightness-chroma decoupling."),
        ("Gaussian Spatial Filtering: ", "To eliminate high-frequency electronic noise and sensor speckle while preserving sharp grain boundaries, a 2D Gaussian smoothing filter is applied to the grayscale image using a 5×5 kernel with standard deviation sigma = 0, mathematically expressed as:\n"
         "G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))."),
        ("Automatic Otsu Thresholding: ", "The smoothed grayscale image is segmented into foreground and background using Otsu's bimodal thresholding [7]. Otsu's method exhaustively searches for the optimal threshold T* that maximizes the between-class variance sigma_B^2(T) = w0(T)*w1(T)*[mu0(T) - mu1(T)]^2, separating the bright grain foreground from the dark background without manual parameter tuning."),
        ("Morphological Mask Refinement: ", "The raw binary mask undergoes two-stage morphological filtering using an elliptical structuring element (cv2.MORPH_ELLIPSE) of size 5×5 pixels. First, Morphological Closing (dilation followed by erosion) bridges small internal micro-voids, cracks, or shadows inside the grain boundary. Second, Morphological Opening (erosion followed by dilation) eliminates isolated background speckles, dust particles, and stray noise."),
        ("Primary Grain Contour Extraction & Area Filtering: ", "External contours are extracted using cv2.RETR_EXTERNAL and approximated via cv2.CHAIN_APPROX_SIMPLE. The contours are ranked by enclosed pixel area. To eliminate stray dust or chaff fragments, a strict area filter threshold (minimum grain area = 500 pixels) is enforced, selecting the single dominant contour representing the true grain."),
        ("Cropping with Proportional Margin: ", "A tight upright bounding rectangle is computed around the isolated grain contour. To ensure boundary pixels, pericarp edges, and awn remnants are not clipped, a 5% margin ratio (margin = 0.05) is added symmetrically around the bounding box."),
        ("Isotropic Resizing and Centered Background Padding: ", "To standardize the image for deep neural network ingestion without introducing artificial geometric distortion, anisotropic stretching is strictly avoided. The cropped grain is isotropically scaled such that its maximum dimension occupies 90% of the target canvas (fill scale = 0.90), preserving the true physical aspect ratio. The scaled kernel is then centered within a standardized 224×224×3 pixel canvas with zero-intensity black padding.")
    ]
    for title, desc in stages:
        add_bullet_p(doc, desc, bold_prefix=title)

    # Embed Figure 5.1.1 / 5.1.2
    add_figure(
        doc,
        FIG_DIR / "fig2_preprocessing_stages.png",
        "Figure 5.1.1: Four-Stage Preprocessing Workflow Demonstrated on Representative Grain Sample: "
        "(a) Original BGR Image, (b) Gaussian Denoised Grayscale, (c) Refined Binary Mask, and (d) Segmented 224×224 Standardized Grain.",
        width_in=5.6
    )

    # 5.1.2 Handcrafted Feature Extraction
    add_subsection_heading(doc, "5.1.2 Handcrafted Feature Extraction Pipeline")
    p_hc1 = (
        "While deep learning architectures automatically learn latent visual patterns, agricultural scientists and commercial "
        "grain inspectors rely heavily on explicit, standardized physical measurements. To retain direct domain interpretability, "
        "the first feature extraction branch extracts a comprehensive 62-dimensional handcrafted feature vector, systematically "
        "partitioned into three complementary feature families:\n\n"
        "62 Handcrafted Features = 14 Shape/Morphological + 12 GLCM Texture + 36 Multichannel Colour\n\n"
        "The mathematical formulation and physical rationale of each family are detailed below."
    )
    add_body_p(doc, p_hc1, indent=0.3)

    add_subsubsection_heading(doc, "Morphological and Geometric Descriptors (14 Features)")
    p_shape = (
        "Shape features capture the macroscopic dimensions, elongation, edge smoothness, and structural completeness of the grain, "
        "which are vital for identifying broken fragments (5_BN), immature shriveled grains (7_IM), and underdeveloped seeds (6_UN). "
        "Fourteen explicit geometric parameters are computed directly from the isolated binary contour C and its convex hull H:\n"
        "1. Area (A): Total count of foreground pixels enclosed by the grain contour.\n"
        "2. Perimeter (P): Arc length of the continuous boundary contour.\n"
        "3. Width (W) & Height (H): Horizontal and vertical extents of the upright bounding box.\n"
        "4. Bounding Box Area (A_bbox): Product of width and height (W × H).\n"
        "5. Convex Hull Area (A_hull): Area enclosed by the smallest convex polygon enclosing contour C.\n"
        "6. Aspect Ratio: Ratio of kernel height to kernel width (H / W), reflecting grain elongation.\n"
        "7. Extent: Proportion of bounding box occupied by the grain (A / A_bbox).\n"
        "8. Solidity: Proportion of convex hull occupied by the grain (A / A_hull), sensitive to notches and broken corners.\n"
        "9. Circularity (Form Factor): Compactness measure defined as 4 * pi * A / P^2 (equals 1.0 for a perfect circle).\n"
        "10. Equivalent Diameter: Diameter of a circle with equivalent area: sqrt(4 * A / pi).\n"
        "11. Major Axis Length: Length of the primary axis of the best-fit ellipse to contour C.\n"
        "12. Minor Axis Length: Length of the secondary axis of the best-fit ellipse.\n"
        "13. Eccentricity: Degree of elliptical elongation: sqrt(1 - (MinorAxis / MajorAxis)^2).\n"
        "14. Bounding Box Area: Upright area descriptor capturing physical footprint."
    )
    add_body_p(doc, p_shape, indent=0.2)

    # 5.1.3 GLCM Texture Features
    add_subsection_heading(doc, "5.1.3 Gray-Level Co-occurrence Matrix (GLCM) Texture Features")
    p_glcm1 = (
        "Surface texture provides essential discriminatory information for distinguishing healthy, smooth endosperms from chalky, "
        "shriveled, or fungal-damaged surfaces. The Gray-Level Co-occurrence Matrix (GLCM), formulated by Haralick et al. [6], "
        "quantifies the spatial distribution of gray-level intensity combinations between pairs of pixels separated by specified "
        "spatial displacements d and angular orientations theta."
    )
    add_body_p(doc, p_glcm1, indent=0.3)

    p_glcm2 = (
        "In this pipeline, texture analysis is executed exclusively across the foreground grain pixels, preventing background black "
        "pixels from distorting the co-occurrence statistics. To optimize computational throughput while retaining fine textural "
        "gradation, the grayscale intensities [0, 255] are quantized into N_g = 32 discrete gray levels. GLCM matrices P(i, j | d, theta) "
        "are computed across two spatial distances (d in {1, 2} pixels) and four angular directions (theta in {0°, 45°, 90°, 135°}), "
        "yielding 8 co-occurrence matrices. Each matrix is normalized such that sum_{i, j} P(i, j) = 1."
    )
    add_body_p(doc, p_glcm2, indent=0.3)

    p_glcm3 = (
        "From the normalized matrices, six foundational Haralick statistical descriptors are evaluated:\n"
        "1. Contrast: Measures local intensity variations and sharpness of surface fissures:\n"
        "   Contrast = sum_{i, j} |i - j|^2 * P(i, j).\n"
        "2. Dissimilarity: Linear measure of local variations:\n"
        "   Dissimilarity = sum_{i, j} |i - j| * P(i, j).\n"
        "3. Homogeneity (Inverse Difference Moment): Measures smoothness and local uniformity:\n"
        "   Homogeneity = sum_{i, j} P(i, j) / (1 + |i - j|^2).\n"
        "4. Energy (Uniformity): Sum of squared elements, reflecting structural order:\n"
        "   Energy = sqrt(sum_{i, j} P(i, j)^2).\n"
        "5. Angular Second Moment (ASM): Measures textural orderliness: ASM = sum_{i, j} P(i, j)^2.\n"
        "6. Correlation: Measures linear dependency of gray levels between neighboring pixels:\n"
        "   Correlation = sum_{i, j} (i - mu_x) * (j - mu_y) * P(i, j) / (sigma_x * sigma_y).\n\n"
        "To achieve rotational invariance, each descriptor is aggregated across all 8 spatial offsets, computing both the Mean and "
        "Standard Deviation, yielding exactly 6 × 2 = 12 GLCM texture features."
    )
    add_body_p(doc, p_glcm3, indent=0.2)

    # 5.1.4 Colour Features
    add_subsection_heading(doc, "5.1.4 Multichannel Colour Features")
    p_col1 = (
        "Chromatic variation is the most direct indicator of fungal infestation (such as Fusarium pink spots or Aspergillus flavus "
        "greenish-yellow mycotoxins), thermal discolouration, and immature green pericarp pigments. However, RGB colour representation "
        "heavily couples chromaticity with intensity. To construct a comprehensive chromatic profile, the segmented grain foreground "
        "is projected into three complementary colour representations spanning nine distinct channels:\n"
        "• Red, Green, Blue (RGB): Standard primary additive colour space.\n"
        "• Hue, Saturation, Value (HSV): Perceptual colour space decoupling dominant wavelength (Hue) and purity (Saturation) from brightness (Value).\n"
        "• CIELAB (L*, a*, b*): Perceptually uniform colour space decoupling perceptual lightness (L*) from red-green (a*) and blue-yellow (b*) axes."
    )
    add_body_p(doc, p_col1, indent=0.3)

    p_col2 = (
        "Across each of the 9 channels, four statistical moments are computed strictly across the masked foreground grain pixels:\n"
        "1. Mean (mu): Represents the central tendency and baseline chromatic tone.\n"
        "2. Standard Deviation (sigma): Captures chromatic dispersion, highlighting localized fungal spots or chalky mottling.\n"
        "3. Minimum (min): Detects localized dark necrosis, insect punctures, or severe lesions.\n"
        "4. Maximum (max): Detects localized specular reflections, bright chalky deposits, or bleached endosperm.\n\n"
        "This formulation produces exactly 9 channels × 4 statistical moments = 36 colour features. Table 5.1 summarizes the complete "
        "inventory of all 62 handcrafted features."
    )
    add_body_p(doc, p_col2, indent=0.3)

    tbl_hc_headers = ["FEATURE FAMILY", "COUNT", "EXTRACTED METRICS & DESCRIPTORS", "DISCRIMINATIVE ROLE"]
    tbl_hc_data = [
        ["Morphological / Shape", "14", "Area, perimeter, width, height, aspect ratio, extent, solidity, circularity, eccentricity, major axis, minor axis, equiv diameter, hull area, bbox area", "Segregates broken fragments (5_BN), immature slender kernels (7_IM), and underdeveloped grains (6_UN)."],
        ["GLCM Texture", "12", "Contrast, dissimilarity, homogeneity, energy, correlation, ASM (mean and std across 8 spatial offsets: d in {1, 2}, theta in {0°, 45°, 90°, 135°})", "Captures endosperm surface roughness, internal fissures, and differentiates vitreous from chalky endosperms."],
        ["Chromatic / Colour", "36", "4 statistics (mean, std, min, max) across 9 channels (R, G, B, H, S, V, L*, a*, b*) strictly on foreground mask", "Identifies fungal spots (1_F&S), aflatoxin/mycotoxin discolouration (3_MY, 4_AP), and green pigments."],
        ["Total Handcrafted", "62", "14 Shape + 12 Texture + 36 Colour Descriptors", "Provides an interpretable, domain-specific physical representation."]
    ]
    add_table_data(doc, tbl_hc_headers, tbl_hc_data, col_widths=[1.5, 0.8, 2.5, 2.2], caption="Table 5.1: Complete Inventory and Mathematical Description of 62 Handcrafted Features")

    # 5.1.5 SVM Baseline
    add_subsection_heading(doc, "5.1.5 Support Vector Machine (SVM) Baseline")
    p_svm1 = (
        "To establish a rigorous classical machine-learning benchmark, a Support Vector Machine (SVM) classifier [8] was trained "
        "and evaluated exclusively on the 62 handcrafted features. SVM constructs an optimal separating hyperplane in a "
        "high-dimensional feature space by maximizing the geometric margin between nearest support vectors of competing classes. "
        "Because agricultural visual features exhibit non-linear boundaries, a non-linear Radial Basis Function (RBF) kernel is employed:\n"
        "K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2).\n\n"
        "Hyperparameter optimization on the training split identified an optimal penalty parameter C = 50.0 and gamma = 'scale'. "
        "Because SVM is inherently a binary classifier, multiclass categorization is achieved through a One-vs-Rest (OvR) decision strategy. "
        "On the untouched 3,100-sample test split, the SVM baseline achieved a strong test accuracy of 91.16%, a Macro F1 of 0.8494, and a "
        "Weighted F1 of 0.9161. This proves that 62 handcrafted domain features possess substantial discriminatory capacity, serving as "
        "an indispensable comparative baseline for evaluating deep and hybrid representations."
    )
    add_body_p(doc, p_svm1, indent=0.3)

    # 5.1.6 StandardScaler
    add_subsection_heading(doc, "5.1.6 StandardScaler and Data Leakage Prevention")
    p_scl1 = (
        "The fused hybrid feature space combines heterogeneous numerical quantities with vastly disparate dynamic ranges. For instance, "
        "shape area spans 10,000 to 25,000 pixels, aspect ratio ranges between 1.5 and 4.0, GLCM homogeneity ranges between 0.05 and 0.40, "
        "and deep neural embeddings exhibit unbounded floating-point distributions. Without standardization, unscaled features with "
        "large numerical magnitudes artificially dominate distance metrics and gradient calculations, while smaller but highly "
        "informative texture descriptors are suppressed."
    )
    add_body_p(doc, p_scl1, indent=0.3)

    p_scl2 = (
        "To ensure numerical stability, feature standardization is executed using the StandardScaler transformation, mapping each "
        "feature dimension to zero mean and unit variance:\n"
        "z = (x - mu) / sigma\n"
        "where mu is the empirical mean and sigma is the standard deviation computed across the training dataset."
    )
    add_body_p(doc, p_scl2, indent=0.3)

    add_callout_box(
        doc,
        "A critical methodological integrity safeguard enforced in this work is the strict prevention of Data Leakage. "
        "The StandardScaler parameters (mu and sigma) were fitted EXCLUSIVELY on the 24,767 training samples (scaler.fit(X_train)). "
        "The validation (3,095 samples) and test (3,100 samples) feature sets were transformed strictly using the pre-fitted training "
        "parameters (scaler.transform(X_val) and scaler.transform(X_test)). The scaler was NEVER fit or refit on validation or test "
        "splits. Re-fitting a scaler on test data constitutes data snooping, artificially deflates variance, and generates overly "
        "optimistic, unreplicable performance metrics.",
        title="DATA LEAKAGE PREVENTION & STANDARDIZATION INTEGRITY"
    )

    # 5.1.7 EfficientNet-B0
    add_subsection_heading(doc, "5.1.7 EfficientNet-B0 Deep Feature Backbone")
    p_eff1 = (
        "To complement explicit handcrafted features, the second extraction branch employs a deep convolutional neural network "
        "backbone. Conventional CNN architectures scale arbitrarily along depth (ResNet), width (WideResNet), or resolution. "
        "Tan & Le [2] introduced EfficientNet, demonstrating that scaling network depth (d = alpha^phi), width (w = beta^phi), and "
        "input resolution (r = gamma^phi) concurrently under a fixed compound coefficient phi maintains an optimal balance between "
        "representational capacity and computational efficiency."
    )
    add_body_p(doc, p_eff1, indent=0.3)

    p_eff2 = (
        "In this work, EfficientNet-B0 is selected as the deep visual extractor. EfficientNet-B0 is constructed from Mobile Inverted "
        "Bottleneck Convolutional blocks (MBConv) featuring depthwise separable convolutions, expansion layers, squeeze-and-excitation (SE) "
        "channel attention mechanisms, and residual connections. The network is initialized with weights pretrained on ImageNet (1.28 million "
        "images across 1,000 visual categories) [10]. Each standardized 224×224×3 grain image is normalized using standard ImageNet channel "
        "statistics (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])."
    )
    add_body_p(doc, p_eff2, indent=0.3)

    p_eff3 = (
        "Crucially, EfficientNet-B0 is NOT utilized as an end-to-end classifier in this project. The final 1,000-class dense classification "
        "layer is removed and replaced with a pass-through identity mapping (model.classifier[1] = torch.nn.Identity()). The network is placed "
        "in permanent evaluation mode (model.eval()), and inference is executed under torch.inference_mode(). Forward propagation through the "
        "convolutional backbone and global average pooling layer yields a rich, 1,280-dimensional continuous visual embedding vector for "
        "each grain."
    )
    add_body_p(doc, p_eff3, indent=0.3)

    # 5.1.8 Feature Fusion
    add_subsection_heading(doc, "5.1.8 Feature Fusion Architecture")
    p_fus1 = (
        "The core scientific thesis of this project is that explicit geometric/textural descriptors and learned deep latent "
        "embeddings provide complementary, non-redundant discriminative information. To synthesize these modalities, an early feature-fusion "
        "strategy is implemented via horizontal vector concatenation:\n\n"
        "x_fused = [ x_shape (14), x_glcm (12), x_colour (36), x_deep (1280) ] in R^1342\n\n"
        "Total Fused Dimension = 62 Handcrafted + 1,280 Deep = 1,342 Dimensions."
    )
    add_body_p(doc, p_fus1, indent=0.3)

    p_fus2 = (
        "Handcrafted features explicitly represent macroscopic aspect ratios, millimeter dimensions, and localized pixel variances "
        "that convolutional downsampling layers tend to smooth out. Conversely, deep convolutional embeddings capture subtle, abstract "
        "hierarchical spatial relationships, endosperm opalescence, and micro-lesion contours that defy simple mathematical formulation. "
        "Concatenating both representations creates a high-dimensional descriptor space wherein gradient boosted decision trees can dynamically "
        "exploit linear geometric thresholds alongside high-capacity visual manifolds. The empirical ablation study in Chapter 6 confirms "
        "that this fusion yields statistically superior classification performance across all evaluation splits."
    )
    add_body_p(doc, p_fus2, indent=0.3)

    # 5.1.9 XGBoost
    add_subsection_heading(doc, "5.1.9 Extreme Gradient Boosting (XGBoost) Classifier")
    p_xgb1 = (
        "While deep neural networks excel at continuous feature learning, decision tree ensembles remain the state-of-the-art classifier "
        "for structured, heterogeneous numerical vectors. Extreme Gradient Boosting (XGBoost) [3] is selected as the terminal classification "
        "engine for the 1,342-dimensional fused representation. XGBoost iteratively constructs an additive ensemble of K regression trees "
        "f_k(x) to minimize a regularized objective function:\n"
        "Obj(theta) = sum_{i=1}^N L(y_i, y_hat_i) + sum_{k=1}^K Omega(f_k)\n"
        "where L is the multiclass cross-entropy loss and Omega(f_k) = gamma * T + 0.5 * lambda * sum_{j=1}^T w_j^2 penalizes tree complexity "
        "(number of terminal leaves T and leaf weights w_j)."
    )
    add_body_p(doc, p_xgb1, indent=0.3)

    p_xgb2 = (
        "XGBoost employs a second-order Taylor expansion of the loss function, incorporating both first-order gradients (g_i) and "
        "second-order hessians (h_i) to achieve rapid, exact split optimization. In this work, the histogram-based tree building algorithm "
        "(tree_method = 'hist') is utilized, binning continuous features into discrete bins to dramatically accelerate training across "
        "24,767 training instances while maintaining exact split quality."
    )
    add_body_p(doc, p_xgb2, indent=0.3)

    p_xgb3 = (
        "Exhaustive hyperparameter tuning on the validation split established the final verified model configuration:\n"
        "• n_estimators = 100: Number of boosting iterations, balancing convergence against overfitting.\n"
        "• max_depth = 4: Maximum tree depth, constraining individual trees to 4 levels (at most 16 leaves) to prevent overfitting in the high-dimensional space.\n"
        "• learning_rate (eta) = 0.1: Step size shrinkage applied to each tree update to ensure conservative, robust convergence.\n"
        "• subsample = 0.8: Randomly samples 80% of training instances per tree, introducing stochastic regularization.\n"
        "• colsample_bytree = 0.8: Randomly subsamples 80% of feature columns per tree, decorrelating trees and preventing deep features from monopolizing early splits.\n"
        "• min_child_weight = 1: Minimum sum of instance weight (hessian) needed in a child leaf.\n"
        "• objective = 'multi:softprob': Multi-class softmax cross-entropy, outputting class probability distributions.\n"
        "• balanced_sample_weight = False: Unweighted training, matching the natural empirical distribution. Table 5.2 contrasts the hyperparameter parameters of the evaluated models."
    )
    add_body_p(doc, p_xgb3, indent=0.3)

    tbl_param_headers = ["HYPERPARAMETER", "BASELINE SVM", "DEEP-ONLY XGBOOST", "FINAL HYBRID XGBOOST", "OPERATIONAL RATIONALE"]
    tbl_param_data = [
        ["Input Dimensions", "62 (Handcrafted)", "1,280 (EfficientNet-B0)", "1,342 (Fused)", "Evaluates isolated vs. fused representational capacity."],
        ["Classifier Family", "Support Vector Machine", "Gradient Boosted Trees", "Gradient Boosted Trees", "XGBoost captures complex non-linear feature interactions."],
        ["Scaling Method", "StandardScaler", "None (Raw Features)", "StandardScaler (Train-Fit)", "Normalizes heterogeneous handcrafted and deep scales."],
        ["Estimators / Trees", "N/A (Kernel SVM)", "100 Trees", "100 Trees", "Ensures sufficient ensemble capacity without overtraining."],
        ["Max Tree Depth", "N/A", "4 Levels", "4 Levels", "Constrains tree depth to limit variance in high-D space."],
        ["Learning Rate (eta)", "N/A", "0.1", "0.1", "Conservative step shrinkage prevents premature convergence."],
        ["Subsample Ratio", "N/A", "0.8 (80% instances)", "0.8 (80% instances)", "Stochastic row sampling regularizes tree ensembles."],
        ["Colsample by Tree", "N/A", "0.8 (80% features)", "0.8 (80% features)", "Forces trees to sample across handcrafted and deep subsets."],
        ["Loss Objective", "Hinge Loss (RBF C=50)", "multi:softprob", "multi:softprob", "Outputs normalized 8-class softmax probability distributions."]
    ]
    add_table_data(doc, tbl_param_headers, tbl_param_data, col_widths=[1.3, 1.2, 1.2, 1.2, 1.5], caption="Table 5.2: Verified Hyperparameter Configuration for Baseline and Final Hybrid Models")

    # 5.1.10 SHAP Explainability
    add_subsection_heading(doc, "5.1.10 SHAP Explainability Framework")
    p_shap1 = (
        "In regulated agricultural sortation and commercial grain procurement, black-box predictions undermine operational confidence. "
        "A system that rejects a grain consignment without justification cannot be audited or trusted by human inspectors. To deliver "
        "mathematically rigorous explainability, this project incorporates SHapley Additive exPlanations (SHAP) [4, 5]."
    )
    add_body_p(doc, p_shap1, indent=0.3)

    p_shap2 = (
        "SHAP grounds model interpretability in cooperative game theory, where each feature acts as a player in a game whose payout is "
        "the model's prediction. The unique additive attribution phi_i that satisfies the desirable axioms of Efficiency, Symmetry, "
        "Dummy, and Additivity is the Shapley value:\n"
        "phi_i(x) = sum_{S subseteq N \\ {i}} [ (|S|! * (|N| - |S| - 1)!) / |N|! ] * [ f_x(S union {i}) - f_x(S) ]\n"
        "where N is the complete feature set, S is a feature subset excluding feature i, and f_x(S) is the conditional expectation."
    )
    add_body_p(doc, p_shap2, indent=0.3)

    p_shap3 = (
        "For tree-based models like XGBoost, Lundberg et al. [5] developed TreeExplainer, an algorithm that evaluates exact conditional "
        "expectations by traversing decision tree paths in polynomial time O(TLD^2) (where T is trees, L is leaves, D is depth), "
        "completely avoiding exponential sampling approximations. In this system, SHAP computes local attribution vectors for "
        "every inference sample: positive SHAP values (phi_i > 0) push the prediction toward the selected class, while negative SHAP "
        "values (phi_i < 0) oppose it."
    )
    add_body_p(doc, p_shap3, indent=0.3)

    add_callout_box(
        doc,
        "CRITICAL SCIENTIFIC DISTINCTION: SHAP attributions represent local feature contributions to the model's decision function. "
        "SHAP values do NOT represent: (a) classification accuracy, (b) prediction probability calibration, (c) physical percentage of "
        "grain defects, or (d) formal proof of model correctness. Furthermore, a deep feature attribution (such as deep_feature_766) "
        "represents a high-level mathematical projection learned by EfficientNet-B0 and must NOT be assigned unsupported human biological "
        "meanings without independent spectroscopic validation.",
        title="SCIENTIFIC INTERPRETABILITY CAVEAT"
    )

    # 5.2 SYSTEM ARCHITECTURE
    add_section_heading(doc, "5.2 SYSTEM ARCHITECTURE")
    p_arch1 = (
        "The end-to-end system architecture integrates data ingestion, preprocessing, parallel feature extraction, feature scaling, "
        "ensemble classification, explainability, and user deployment into a cohesive, decoupled pipeline. Figure 5.2.1 illustrates "
        "the complete technical architecture."
    )
    add_body_p(doc, p_arch1, indent=0.3)

    # Embed Figure 5.2.1
    add_figure(
        doc,
        FIG_DIR / "fig1_overall_framework.png",
        "Figure 5.2.1: End-to-End Architectural Pipeline of the Proposed Explainable Hybrid Feature-Fusion Framework "
        "Showing Input Ingestion, Preprocessing, Dual-Branch Feature Extraction, Standardization, XGBoost Classification, "
        "and SHAP Interpretability Integration.",
        width_in=5.8
    )

    p_arch2 = (
        "The architecture is organized into four distinct operational layers:\n"
        "1. Image Acquisition & Preprocessing Layer: Handles file validation, format decoding, Gaussian smoothing, Otsu segmentation, morphological refinement, and standardized 224×224 cropping.\n"
        "2. Dual-Branch Feature Engineering Layer: Concurrently computes 62 handcrafted shape, texture, and colour descriptors alongside 1,280 EfficientNet-B0 deep latent embeddings, producing the 1,342-dimensional vector.\n"
        "3. Inference & Explainability Layer: Applies the training-fitted StandardScaler, queries the trained XGBoost ensemble, evaluates softmax probabilities across the 8 classes, and invokes TreeExplainer SHAP for local attributions.\n"
        "4. Deployment & Persistence Layer: Wraps the core pipeline in a Streamlit graphical application, automated PDF report generation engine, human-in-the-loop review queue (<0.75 threshold), and an SQLite transactional database (`results/rice_quality.db`)."
    )
    add_body_p(doc, p_arch2, indent=0.3)
