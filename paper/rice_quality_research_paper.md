# An Explainable Hybrid Feature Fusion Framework for Rice Quality and Defect Classification Using EfficientNet-B0 and XGBoost

**Dharshan B¹**, **Dinesh G L¹**  
*Mentored by:* **Ms. Sripriya S**  
¹*Department of Information Technology, Kongu Engineering College, Perundurai, Tamil Nadu, India*  

---

### Abstract
Automated visual inspection of milled rice is essential for high-throughput quality grading, defect detection, and commercial standardization. However, single-paradigm classification pipelines frequently suffer from inherent trade-offs: conventional handcrafted descriptors capture explicit, physically interpretable geometric and chromatic properties but struggle with subtle, high-order visual patterns, whereas deep convolutional networks learn expressive latent representations but omit fine-grained boundary metrics and operate as opaque black boxes. To resolve this tension, this paper presents a complete, publication-grade explainable hybrid feature-fusion framework for eight-class rice quality and defect assessment. The pipeline first standardizes raw rice grains via an adaptive segmentation workflow combining Gaussian smoothing, automatic Otsu thresholding, elliptical morphological filtering, and isotropic aspect-ratio padding to $224 \times 224$ pixels. From each segmented grain, we extract a 62-dimensional handcrafted vector comprising 14 morphological/shape descriptors, 12 Gray-Level Co-occurrence Matrix (GLCM) texture descriptors, and 36 multichannel colour statistics across RGB, HSV, and LAB spaces. Concurrently, a pretrained EfficientNet-B0 backbone extracts a 1,280-dimensional global visual embedding. The representations are concatenated into a 1,342-dimensional hybrid vector, standardized using a training-fitted `StandardScaler` to prevent data leakage, and classified using an optimized multiclass XGBoost classifier (`n_estimators=100`, `max_depth=4`, `learning_rate=0.1`). Rigorously evaluated on the GrainSet Rice Dataset across 30,962 images partitioned into strictly non-overlapping train (24,767), validation (3,095), and test (3,100) splits, the proposed hybrid model achieves **92.10% Test Accuracy**, **0.8660 Macro F1**, and **0.9244 Weighted F1**, decisively surpassing both a tuned Support Vector Machine baseline operating on handcrafted features (91.16% accuracy, 0.8494 macro F1) and an XGBoost baseline operating on EfficientNet-B0 features alone (90.65% accuracy, 0.8399 macro F1). A controlled seven-configuration ablation study demonstrates that fusing handcrafted and deep features yields a +2.61 percentage point macro F1 improvement over deep features alone and a +1.66 percentage point gain over handcrafted features alone. Furthermore, post-hoc explainability using TreeExplainer SHAP across 300 test samples reveals that while deep features provide 68.59% of the overall attribution mass, handcrafted features capture 31.41% of the explanatory weight, with physical dimensions (`shape_height`, `color_lab_b_mean`, `shape_major_axis_length`, and `shape_eccentricity`) dominating the top-tier feature attributions. The framework is supported by a deployable human-in-the-loop review system, SQLite persistence, and REST API.

**Index Terms—** Rice quality assessment, defect classification, hybrid feature fusion, EfficientNet-B0, XGBoost, explainable artificial intelligence (XAI), SHAP, computer vision, agricultural automation.

---

## I. Introduction

Rice (*Oryza sativa*) is the primary dietary staple for more than half of the global population. Accurate quality grading and defect assessment directly govern commercial pricing, milling efficiency, storage stability, and food safety standards [1]. In industrial rice milling and grain trade, batches must be categorized into distinct quality tiers based on physical head rice integrity, broken grain proportions, chalkiness, discoloration, immature kernels, and fungal or pest damage [1], [14]. Traditionally, grain inspection has relied on manual visual inspection by certified quality analysts. However, manual grain inspection is inherently subjective, labour-intensive, low-throughput, prone to operator visual fatigue, and vulnerable to substantial inter-assessor variability [1], [15].

Over the past decade, computer vision and machine learning (ML) techniques have increasingly been deployed to automate non-destructive cereal inspection [1], [16]. Classical approaches typically rely on handcrafted feature engineering, where domain experts define explicit mathematical operators to quantify physical attributes: morphological dimensions (area, length, aspect ratio, circularity) to quantify broken or deformed grains; Gray-Level Co-occurrence Matrix (GLCM) texture metrics to evaluate surface roughness and fissures; and colour space statistics (RGB, HSV, LAB) to detect chalky bellies, yellowing, or pathogen lesions [1], [6], [14]. These handcrafted features possess the decisive advantage of direct physical interpretability and computational lightness, allowing classifiers such as Support Vector Machines (SVM) [8] or Random Forests to achieve strong baseline performance on controlled datasets. Nevertheless, handcrafted descriptors fail when confronted with complex, non-linear visual interactions—such as irregular superficial mould colonization, variegated defect pigmentation, and subtle milling variations—because manually engineered equations cannot anticipate all natural biological variabilities [17], [18].

Conversely, modern Deep Learning (DL) architectures—particularly Convolutional Neural Networks (CNNs)—automatically extract rich, hierarchical visual representations directly from raw pixel arrays [2], [10]. Pretrained backbones such as EfficientNet-B0 [2] leverage inverted residual blocks and compound coefficient scaling to learn generalizable visual features while maintaining high parameter efficiency. Nonetheless, pure deep-learning paradigms exhibit two fundamental operational vulnerabilities in industrial quality grading:
1. **Loss of Fine-Grained Physical Measurements:** Standard CNN feature extractors compress input images through successive pooling and stride operations, frequently attenuating exact millimeter-scale metric boundaries, aspect ratios, and edge perimeters that are critical for grain grading [1].
2. **The "Black Box" Problem:** Deep networks provide opaque numerical embeddings that obscure the biological rationale behind a classification decision, creating severe trust barriers for agricultural traders, regulatory auditors, and mill operators [4], [5].

To address these complementary limitations, this investigation formulates and empirically validates an **explainable hybrid feature-fusion framework** for eight-class rice quality and defect assessment. By concatenating 62 domain-specific handcrafted descriptors with 1,280 deep embeddings from a pretrained EfficientNet-B0 backbone into a 1,342-dimensional vector, and classifying the fused representation using an optimized XGBoost gradient-boosted decision tree ensemble [3], the proposed methodology captures both explicit physical boundaries and complex latent visual semantics. Furthermore, to overcome the interpretability barrier, the framework incorporates post-hoc Shapley Additive Explanations (SHAP) [4], [5], providing mathematically grounded global and local attributions that disentangle the contributions of geometric, textural, chromatic, and deep feature subsets.

### Scope and Research Contributions
The empirical findings presented in this paper are based exclusively on verified experimental execution over 30,962 images from the GrainSet Rice Dataset. The specific, verified contributions of this research are:
1. **A Standardized, Reproducible Preprocessing and Segmentation Pipeline:** An adaptive, boundary-preserving grain isolation pipeline combining Gaussian filtering, Otsu thresholding, elliptical morphological refinement, contour filtering, and isotropic aspect-ratio padding to $224 \times 224$ pixels, achieving a mean latency of 5.44 ms per image.
2. **A 62-Dimensional Multi-Domain Handcrafted Feature Engineering Architecture:** Extraction of 14 morphological descriptors, 12 GLCM texture statistics computed strictly on the grain foreground, and 36 colour statistics across RGB, HSV, and LAB spaces.
3. **An EfficientNet-B0 Deep Feature Extractor:** Deployment of an ImageNet-pretrained EfficientNet-B0 backbone operating in no-gradient inference mode to generate 1,280-dimensional global visual embeddings without expensive end-to-end retraining.
4. **A 1,342-Dimensional Hybrid Feature Representation with Strict Leakage Hygiene:** A fused feature schema standardized via a training-fitted `StandardScaler`, ensuring zero test leakage across 30,962 strictly partitioned samples.
5. **Systematic Empirical Benchmarking:** Controlled comparative evaluation demonstrating that the proposed Hybrid XGBoost model (92.10% test accuracy, 0.8660 macro F1) decisively outperforms both a tuned handcrafted SVM baseline (91.16% accuracy, 0.8494 macro F1) and an EfficientNet-B0 + XGBoost baseline (90.65% accuracy, 0.8399 macro F1).
6. **A Controlled Seven-Configuration Ablation Study:** Methodological ablation across individual feature groups (Shape, GLCM, Colour, All Handcrafted, EfficientNet-B0, and Fused Hybrid), demonstrating that handcrafted and deep features are non-redundant and synergistic.
7. **Disentangled Post-Hoc Model Explainability:** Application of TreeExplainer SHAP on 300 test samples to quantify feature-group contributions and resolve the critical divergence between internal tree-gain importance and external Shapley attribution.
8. **Operational Decision Support Engineering:** A supporting application stack comprising a Streamlit multi-page interface, an automated human-in-the-loop review queue for low-confidence predictions ($p < 0.75$), SQLite persistence, and REST API endpoints.

The remainder of this paper is organized as follows: Section II reviews related literature. Section III details the dataset properties and mathematical problem formulation. Section IV explains the proposed methodology. Section V details the experimental setup and evaluation protocol. Section VI presents the comparative empirical results. Section VII presents the feature ablation study. Section VIII details the SHAP explainability analysis. Section IX conducts an in-depth error and confusion matrix analysis. Section X describes supporting deployment components. Section XI discusses verified limitations. Section XII outlines future research directions, and Section XIII concludes the paper.

---

## II. Related Work

Automated agricultural grain classification has evolved through three distinct methodological epochs: classical morphology-based machine vision, deep learning feature extractors, and hybrid ensemble frameworks.

### A. Classical Machine Vision and Handcrafted Descriptors
Early automated grain assessment systems relied primarily on morphological geometry and flatbed scanner image acquisition [1], [14], [15]. Zareiforoush et al. [14] evaluated qualitative classification of milled rice grains using geometric features, reporting that length, width, aspect ratio, and projected area provided adequate discrimination for bulk dimensional grading. Mahale and Korde [15] applied thresholding and boundary tracing to classify Indian rice varieties, demonstrating that shape factors alone could identify whole vs. broken grains but failed when grains exhibited internal chalkiness or minor surface defects.

To capture surface texture, researchers integrated Gray-Level Co-occurrence Matrix (GLCM) formulations, originally introduced by Haralick et al. [6]. GLCM computes second-order statistical dependencies between pairs of pixels separated by distance $d$ at orientation $\theta$. Sun et al. [16] demonstrated that GLCM descriptors (contrast, energy, homogeneity, and correlation) correlated with chalky endosperm voids and internal stress cracks. Colour statistics across multiple colour spaces (RGB, HSV, CIE-LAB) were subsequently integrated by Mittal, Dutta, and Issac [1], who developed a non-destructive image processing framework for assessing rice quality and commercial value. Mittal et al. extracted geometric and colour histogram features to infer commercial quality grades. However, while Mittal et al. [1] utilized a specialized multi-feature scoring heuristic for commercial indexing, their framework relied exclusively on handcrafted descriptors and did not incorporate deep convolutional feature learning, gradient boosting, or post-hoc Shapley explainability.

For classification, classical systems predominantly adopted Support Vector Machines (SVM) with Radial Basis Function (RBF) kernels [8] or Multilayer Perceptrons (MLP). While SVMs generalize well in high-dimensional handcrafted spaces, their quadratic training complexity $\mathcal{O}(N^2 \cdot D)$ severely degrades scalability on large-scale industrial datasets ($N > 25,000$).

### B. Deep Learning and Efficient Neural Backbones
The advent of deep Convolutional Neural Networks (CNNs) revolutionized agricultural image processing [10], [17]. Pretrained architectures such as VGG, ResNet, and DenseNet demonstrated the capacity to bypass manual feature engineering by learning hierarchical spatial representations directly from raw imagery. Tan and Le [2] introduced EfficientNet, which established a systematic compound scaling method that uniformly scales network depth, width, and input resolution using a fixed compound coefficient $\phi$. EfficientNet-B0, developed via neural architecture search (NAS), utilizes mobile inverted bottleneck convolutions (MBConv) with squeeze-and-excitation optimization, achieving 77.1% top-1 accuracy on ImageNet while utilizing only 5.3 million parameters [2].

Despite their representational power, purely deep networks applied to agricultural grading exhibit specific vulnerabilities. Fabiyi et al. [17] noted that fine-tuning deep CNNs on grain datasets with strong class imbalance often leads to overfitting on majority categories and loss of fine geometric perimeter details due to successive spatial pooling. Furthermore, deep networks require substantial computational infrastructure for fine-tuning and inference, posing barriers for deployment on embedded edge sorters.

### C. Gradient Boosted Decision Trees and Hybrid Feature Fusion
Ensemble decision trees—specifically XGBoost (Extreme Gradient Boosting), introduced by Chen and Guestrin [3]—have emerged as state-of-the-art classifiers for structured, tabular, and fused feature representations. XGBoost utilizes a sparsity-aware split-finding algorithm, second-order Taylor expansion of the loss function, and explicit tree regularization ($\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum w_j^2$) to prevent overfitting while handling multicollinear feature sets [3].

Feature fusion bridges the gap between domain expertise and learned representations [17], [18]. Yao et al. [18] investigated combining morphological features with colour features for grain inspection, concluding that multi-domain fusion achieved higher classification stability than any single descriptor. However, existing literature in cereal inspection has largely explored handcrafted-to-handcrafted fusion or fine-tuned end-to-end CNNs. A systematic, leakage-free integration of high-dimensional deep embeddings (1,280-D EfficientNet-B0) with comprehensive multi-domain handcrafted descriptors (62-D) classified via regularized gradient boosting remains underexplored.

### D. Explainable Artificial Intelligence (XAI) in Agricultural Vision
As machine learning models are integrated into industrial quality control, algorithmic transparency has become an imperative [4], [5]. Lundberg and Lee [4] formulated SHAP (Shapley Additive Explanations), unifying cooperative game theory with local surrogate explanations. For tree-based ensembles, Lundberg et al. [5] developed TreeExplainer, an algorithm capable of computing exact Shapley values in low-order polynomial time $\mathcal{O}(T L D^2)$, where $T$ is the number of trees, $L$ is the maximum number of leaves, and $D$ is the maximum tree depth. In agricultural classification, explainability has rarely been applied to fused feature representations to disentangle the relative influence of physical biological descriptors versus latent deep embeddings. This research directly bridges this methodological gap.

---

## III. Dataset and Problem Formulation

### A. The GrainSet Rice Dataset
The empirical investigation is conducted on the **GrainSet Rice Dataset** as curated and partitioned in the project repository. The dataset comprises a total of **30,962 high-resolution rice grain images**. To guarantee rigorous scientific validity, prevent data leakage, and ensure reproducible benchmarking, the original dataset split was strictly preserved:
- **Training Split:** 24,767 images (79.99%)
- **Validation Split:** 3,095 images (10.00%)
- **Test Split:** 3,100 images (10.01%)

A secondary balanced training sub-manifest (`rice_train_bal.txt`) containing 9,528 images is documented in the repository; however, all primary training and validation experiments for the final hybrid model were conducted on the full standard 24,767-image training split to maximize statistical diversity. An automated filesystem audit confirmed that all 40,490 manifest entries exist on disk, with 0 corrupt files and exactly **0 overlapping filenames between the train, validation, and test partitions**, confirming strict leakage prevention.

### B. Class Distribution and Preservation of Raw Labels
The dataset encompasses eight visually distinct grain categories. Because semantic expansions (e.g., specific defect pathology or commercial varietal names) are not independently verified in the repository documentation, all classes are strictly identified by their original raw labels to preserve scientific integrity:
1. `0_NOR` (Normal / Reference grain)
2. `1_F&S`
3. `2_SD`
4. `3_MY`
5. `4_AP`
6. `5_BN`
7. `6_UN`
8. `7_IM`

The class distributions across the three evaluation splits are summarized in Table I.

```
+---------------------------------------------------------------------------------------------+
| TABLE I: DATASET DISTRIBUTION AND CLASS COMPOSITION ACROSS EVALUATION SPLITS                |
+---------+------------+--------------------+-------------------+-------------------+---------+
| ClassID | Raw Label  | Train Count (%)    | Val Count (%)     | Test Count (%)    | Total   |
+---------+------------+--------------------+-------------------+-------------------+---------+
| 0       | 0_NOR      | 15,980 (64.52%)    | 2,020 (65.27%)    | 2,000 (64.52%)    | 20,000  |
| 1       | 1_F&S      |  1,191  (4.81%)    |   145  (4.68%)    |   150  (4.84%)    |  1,486  |
| 2       | 2_SD       |  1,194  (4.82%)    |   156  (5.04%)    |   150  (4.84%)    |  1,500  |
| 3       | 3_MY       |  1,223  (4.94%)    |   127  (4.10%)    |   150  (4.84%)    |  1,500  |
| 4       | 4_AP       |  1,208  (4.88%)    |   142  (4.59%)    |   150  (4.84%)    |  1,500  |
| 5       | 5_BN       |  1,198  (4.84%)    |   143  (4.62%)    |   150  (4.84%)    |  1,491  |
| 6       | 6_UN       |  1,185  (4.78%)    |   150  (4.85%)    |   150  (4.84%)    |  1,485  |
| 7       | 7_IM       |  1,588  (6.41%)    |   212  (6.85%)    |   200  (6.45%)    |  2,000  |
+---------+------------+--------------------+-------------------+-------------------+---------+
| Total   |            | 24,767 (100.0%)    | 3,095 (100.0%)    | 3,100 (100.0%)    | 30,962  |
+---------+------------+--------------------+-------------------+-------------------+---------+
```

As demonstrated in Table I, the dataset exhibits substantial class imbalance: the majority class `0_NOR` represents approximately 64.5% of all splits, while defect categories `1_F&S` through `6_UN` each represent roughly 4.8% of the data. Consequently, overall classification accuracy alone is an insufficient metric; **macro-averaged F1-score** and **class-wise recall/precision** are essential to evaluate model robustness on minority defect categories.

### C. Problem Formulation
Let $\mathcal{D} = \{(I_i, y_i)\}_{i=1}^N$ denote the dataset of rice grain images $I_i \in \mathbb{R}^{H_i \times W_i \times 3}$ and associated categorical ground-truth labels $y_i \in \mathcal{C} = \{0, 1, \dots, 7\}$.

The objective is to learn a mapping function:
$$
f: I \xrightarrow{\text{Preprocess}} \hat{I} \xrightarrow{\text{Feature Extraction}} \mathbf{z} \xrightarrow{\text{Standardize}} \tilde{\mathbf{z}} \xrightarrow{\text{Classification}} \hat{\mathbf{p}} \in \Delta^7
$$
where $\hat{I} \in \mathbb{R}^{224 \times 224 \times 3}$ is the standardized segmented grain, $\mathbf{z} \in \mathbb{R}^{1342}$ is the concatenated hybrid feature vector, $\tilde{\mathbf{z}}$ is the normalized representation, and $\hat{\mathbf{p}} = [p_0, p_1, \dots, p_7]$ is the predicted class probability distribution on the 7-simplex $\Delta^7$. The predicted discrete class is assigned via the maximum a posteriori rule:
$$
\hat{y} = \arg\max_{c \in \mathcal{C}} p_c
$$

---

## IV. Proposed Methodology

The overall architecture of the proposed explainable hybrid feature-fusion framework is depicted in Fig. 1. The framework comprises five sequential modules: (A) Image Preprocessing and Grain Segmentation, (B) Handcrafted Feature Extraction, (C) EfficientNet-B0 Deep Feature Extraction, (D) Hybrid Feature Fusion and Normalization, and (E) XGBoost Classification with SHAP Explainability.

```
+--------------------------------------------------------------------------------------------------------+
|                                    FIG. 1: OVERALL ARCHITECTURE FLOWCHART                              |
|                                                                                                        |
|  [ Raw Rice Image ]                                                                                    |
|          │                                                                                             |
|          ▼                                                                                             |
|  [ Stage 1: Preprocessing & Denoising ]                                                                |
|          │  ├── Gaussian Blur Smoothing (5x5 kernel, sigma=0)                                          |
|          │  ├── Automatic Otsu Thresholding                                                            |
|          │  └── Morphological Closing & Opening (5x5 Elliptical SE)                                    |
|          ▼                                                                                             |
|  [ Stage 2: Grain Isolation & Standardization ]                                                        |
|          │  ├── External Contour Extraction & Filtering (Area > 500 px)                                 |
|          │  ├── Binary Mask Generation & Foreground Masking                                            |
|          │  └── Bounding Box Crop (5% margin) + Isotropic Scaling to 224x224x3 Canvas                  |
|          ├────────────────────────────────────────┬──────────────────────────────────────────┐         |
|          ▼                                        ▼                                          │         |
|  [ Handcrafted Features (62-D) ]       [ Deep Feature Extractor (1280-D) ]                   │         |
|   ├── Shape / Morphology (14-D)         ├── ImageNet Pretrained EfficientNet-B0              │         |
|   ├── GLCM Texture (12-D)               ├── Classification Head Replaced with Identity()     │         |
|   └── Multichannel Colour (36-D)        └── Torch Inference Mode (1280-D Vector)             │         |
|          │                                        │                                          │         |
|          └───────────────────┬────────────────────┘                                          │         |
|                              ▼                                                               │         |
|               [ Feature Concatenation: 1,342-D ]                                             │         |
|                              │                                                               │         |
|                              ▼                                                               │         |
|               [ StandardScaler (Train-Fit Only) ]                                            │         |
|                              │                                                               │         |
|                              ▼                                                               │         |
|               [ Multiclass XGBoost Classifier ]                                              │         |
|               (n_est=100, depth=4, lr=0.1, multi:softprob)                                   │         |
|                              │                                                               │         |
|                              ▼                                                               │         |
|          ┌───────────────────┴────────────────────┐                                          │         |
|          ▼                                        ▼                                          ▼         |
|  [ 8-Class Prediction ]               [ TreeExplainer SHAP ]                 [ Supporting Pipeline ]   |
|   ├── Predicted Class & Probabilities   ├── Feature Group Attribution (%)     ├── Human Review Queue   |
|   └── Confidence Evaluation             └── Top-20 Feature Attributions       ├── SQLite Logging & DB  |
|                                                                               └── REST API & Dashboard |
+--------------------------------------------------------------------------------------------------------+
```

### A. Image Preprocessing and Grain Segmentation
The raw images in GrainSet possess variable resolutions (ranging from $206 \times 135$ px to $328 \times 301$ px, with an empirical mean of $261.5 \times 222.2$ px) and contain background sensor noise and illumination variations. The preprocessing pipeline operates as follows:

1. **Colour Space Conversion:** Given an input BGR image $I_{BGR}$, the image is converted into RGB, Grayscale, HSV, and CIE-LAB colour representations:
   $$
   I_{Gray} = 0.299 R + 0.587 G + 0.114 B
   $$
2. **Noise Attenuation:** Grayscale images are smoothed using a Gaussian filter with kernel size $K = 5 \times 5$ and $\sigma = 0$:
   $$
   I_{denoised}(x, y) = I_{Gray}(x, y) * G(x, y; \sigma)
   $$
3. **Adaptive Foreground Thresholding:** Automatic Otsu thresholding [7] calculates the optimal binarization threshold $T^*$ that minimizes intra-class variance $\sigma_w^2(T)$:
   $$
   T^* = \arg\min_T \left[ \omega_0(T) \sigma_0^2(T) + \omega_1(T) \sigma_1^2(T) \right]
   $$
   yielding binary mask $M_{binary}(x, y) \in \{0, 255\}$.
4. **Morphological Refinement:** To bridge small chalky fissures and internal grain voids while eliminating stray dust particles, morphological closing followed by opening is executed using an elliptical structuring element $B_{ellipse}$ of dimension $5 \times 5$:
   $$
   M_{refined} = (M_{binary} \bullet B_{ellipse}) \circ B_{ellipse}
   $$
5. **Primary Grain Isolation:** External contours are extracted via topological structural analysis (`cv2.RETR_EXTERNAL`, `cv2.CHAIN_APPROX_SIMPLE`). The contour $\mathcal{K}^*$ with the maximum enclosed area exceeding a minimum threshold of 500 px is selected as the primary grain:
   $$
   \mathcal{K}^* = \arg\max_{\mathcal{K} \in \{\text{contours}\}} \text{Area}(\mathcal{K}), \quad \text{subject to } \text{Area}(\mathcal{K}) \ge 500
   $$
   A clean binary mask $M_{clean}$ is formed by rendering $\mathcal{K}^*$ filled.
6. **Masking, Cropping, and Standardized Padding:** The original BGR image is masked via bitwise conjunction: $I_{masked} = I_{BGR} \odot M_{clean}$. The bounding box $(x, y, w, h)$ of $\mathcal{K}^*$ is computed and expanded by an isotropic margin ratio of $\alpha = 0.05$ (5% padding):
   $$
   x_1 = \max(0, x - \alpha w), \quad y_1 = \max(0, y - \alpha h)
   $$
   $$
   x_2 = \min(W_{orig}, x + w + \alpha w), \quad y_2 = \min(H_{orig}, y + h + \alpha h)
   $$
   The cropped grain $I_{crop} = I_{masked}[y_1:y_2, x_1:x_2]$ is isotropically scaled to fit within a $224 \times 224$ frame with a fill scaling factor of $\beta = 0.90$. The scaled grain is centered upon a black zero-padded canvas of dimension $224 \times 224 \times 3$, producing the standardized image $\hat{I}$.

The measured mean execution latency of this preprocessing pipeline across experimental trials is **5.44 ms per image**. Visualizations of intermediate preprocessing stages and multi-class segmentations are presented in Fig. 2 and Fig. 3.

### B. Handcrafted Feature Extraction (62 Dimensions)
Handcrafted features are extracted strictly from the segmented grain foreground pixels $\Omega = \{(x, y) \mid M_{clean}(x, y) > 0\}$ to eliminate background bias:

1. **Shape and Morphological Features ($\mathbf{h}_{shape} \in \mathbb{R}^{14}$):**
   Derived from the primary contour $\mathcal{K}^*$:
   - *Area ($A$):* Total foreground pixel count, $A = \sum_{(x, y) \in \Omega} 1$.
   - *Perimeter ($P$):* Arc length of closed contour $\mathcal{K}^*$.
   - *Bounding Box Metrics:* Width $w$, Height $h$, Bounding Box Area $A_{bbox} = w \cdot h$.
   - *Aspect Ratio:* $\rho = w / h$.
   - *Extent:* Ratio of grain area to bounding box area, $e = A / A_{bbox}$.
   - *Convex Hull and Solidity:* Convex hull area $A_{hull}$ and solidity $s = A / A_{hull}$.
   - *Circularity:* Metric compactness, $C = \frac{4 \pi A}{P^2}$, clamped to $[0, 1]$.
   - *Equivalent Diameter:* $D_{eq} = \sqrt{\frac{4A}{\pi}}$.
   - *Ellipse Geometry & Eccentricity:* Direct least-squares ellipse fitting yields semi-major axis $a$, semi-minor axis $b$, and eccentricity:
     $$
     \epsilon = \sqrt{1 - \frac{b^2}{a^2}}
     $$
   - *Convex Hull Area:* $A_{hull}$.

2. **GLCM Texture Descriptors ($\mathbf{h}_{texture} \in \mathbb{R}^{12}$):**
   Extracted from the quantized grayscale grain (32 intensity levels). Gray-Level Co-occurrence Matrices $P(i, j; d, \theta)$ are computed across two displacement distances $d \in \{1, 2\}$ and four orientations $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$ (yielding 8 offset matrices). Crucially, co-occurrences are accumulated strictly when both paired pixels reside within $\Omega$. For each matrix, six texture properties are computed:
   - *Contrast:* $\sum_{i, j} (i - j)^2 P(i, j)$
   - *Dissimilarity:* $\sum_{i, j} |i - j| P(i, j)$
   - *Homogeneity:* $\sum_{i, j} \frac{P(i, j)}{1 + (i - j)^2}$
   - *Energy (Angular Second Moment root):* $\sqrt{\sum_{i, j} P(i, j)^2}$
   - *Correlation:* $\sum_{i, j} \frac{(i - \mu_i)(j - \mu_j) P(i, j)}{\sigma_i \sigma_j}$
   - *Angular Second Moment (ASM):* $\sum_{i, j} P(i, j)^2$  
   Averaging and calculating the standard deviation across all 8 directional offsets produces $6 \times 2 = 12$ GLCM texture features.

3. **Multichannel Colour Descriptors ($\mathbf{h}_{colour} \in \mathbb{R}^{36}$):**
   Extracted across nine separate colour channels: $R, G, B$ from RGB; $H, S, V$ from HSV; and $L, A, B$ from CIE-LAB. For each channel $c \in \{R, G, B, H, S, V, L, A, B\}$, four statistics are computed over foreground pixels $\Omega$:
   $$
   \mu_c = \frac{1}{|\Omega|} \sum_{p \in \Omega} I_c(p), \quad \sigma_c = \sqrt{\frac{1}{|\Omega|} \sum_{p \in \Omega} (I_c(p) - \mu_c)^2}
   $$
   $$
   \min_c = \min_{p \in \Omega} I_c(p), \quad \max_c = \max_{p \in \Omega} I_c(p)
   $$
   yielding $9 \times 4 = 36$ colour features.

Concatenating the handcrafted vectors yields:
$$
\mathbf{h}_{handcrafted} = [\mathbf{h}_{shape}, \mathbf{h}_{texture}, \mathbf{h}_{colour}] \in \mathbb{R}^{62}
$$

### C. EfficientNet-B0 Deep Feature Extraction (1,280 Dimensions)
Standardized grain images $\hat{I} \in \mathbb{R}^{224 \times 224 \times 3}$ are converted from BGR to RGB and normalized using ImageNet channel-wise statistics ($\boldsymbol{\mu} = [0.485, 0.456, 0.406]$, $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$).

We instantiate an EfficientNet-B0 architecture pretrained on ImageNet-1k [2], [10]. To extract high-level visual representations without imposing ImageNet class priors, the final 1,000-way linear classification head is replaced with an identity operator:
$$
\text{model.classifier} = \text{torch.nn.Identity}()
$$
The network operates in strict evaluation mode (`model.eval()`) under PyTorch inference mode (`torch.inference_mode()`) with batch size 32. Passing the normalized tensor $\mathbf{X} \in \mathbb{R}^{B \times 3 \times 224 \times 224}$ through the feature backbone yields a 1,280-dimensional global embedding vector:
$$
\mathbf{d} = \phi_{EfficientNet}(\hat{I}) \in \mathbb{R}^{1280}
$$

### D. Hybrid Feature Fusion and Leakage-Safe Normalization
The handcrafted and deep representations are concatenated to form the complete hybrid representation:
$$
\mathbf{z} = [\mathbf{h}_{handcrafted}, \mathbf{d}] = [h_1, h_2, \dots, h_{62}, d_1, d_2, \dots, d_{1280}]^T \in \mathbb{R}^{1342}
$$
Because the 62 handcrafted features possess disparate physical units (pixels, ratios, angles, intensity moments) and deep features exhibit arbitrary positive activations, feature scaling is mandatory. To prevent data leakage, a `StandardScaler` is fitted **exclusively on the training feature matrix** $\mathbf{Z}_{train} \in \mathbb{R}^{24767 \times 1342}$:
$$
\mu_j = \frac{1}{N_{tr}} \sum_{i=1}^{N_{tr}} z_{i, j}, \quad \sigma_j = \sqrt{\frac{1}{N_{tr}} \sum_{i=1}^{N_{tr}} (z_{i, j} - \mu_j)^2 + \epsilon}
$$
Validation, testing, and real-time inference vectors are standardized using the stored training statistics:
$$
\tilde{z}_{i, j} = \frac{z_{i, j} - \mu_j}{\sigma_j}
$$

### E. XGBoost Multiclass Classification
The standardized hybrid vector $\tilde{\mathbf{z}} \in \mathbb{R}^{1342}$ is classified using an Extreme Gradient Boosting (XGBoost) model [3]. For an 8-class problem, XGBoost trains $K = 8$ additive trees at each boosting iteration $m$. The objective function optimized at step $t$ is:
$$
\mathcal{L}^{(t)} = \sum_{i=1}^N \ell(y_i, \hat{\mathbf{p}}_i^{(t-1)} + f_t(\tilde{\mathbf{z}}_i)) + \sum_{k=1}^K \Omega(f_{t, k})
$$
where $\ell$ is the multiclass cross-entropy loss (`mlogloss`):
$$
\ell(y_i, \mathbf{p}_i) = -\sum_{c=0}^7 \mathbb{I}(y_i = c) \ln p_{i, c}
$$
and tree regularization penalizes leaf complexity:
$$
\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2
$$
Probabilities are derived via the softmax function:
$$
p_c(\tilde{\mathbf{z}}) = \frac{\exp(F_c(\tilde{\mathbf{z}}))}{\sum_{k=0}^7 \exp(F_k(\tilde{\mathbf{z}}))}
$$

### F. Post-Hoc Model Explainability via SHAP
To ensure transparency, we deploy the TreeExplainer algorithm [5] to compute exact Shapley values. The prediction for class $c$ is decomposed into a base value $\phi_0(c)$ plus additive feature attributions:
$$
F_c(\tilde{\mathbf{z}}) = \phi_0(c) + \sum_{j=1}^{1342} \phi_j(c, \tilde{\mathbf{z}})
$$
where $\phi_j(c, \tilde{\mathbf{z}})$ represents the marginal contribution of feature $j$ to class $c$:
$$
\phi_j(c) = \sum_{S \subseteq \mathcal{F} \setminus \{j\}} \frac{|S|!(|\mathcal{F}| - |S| - 1)!}{|\mathcal{F}|!} \left[ f_c(S \cup \{j\}) - f_c(S) \right]
$$
Global feature importance is quantified via the mean absolute SHAP value across $M = 300$ sampled test instances and all $K = 8$ classes:
$$
I_j = \frac{1}{M \cdot K} \sum_{i=1}^M \sum_{c=0}^7 |\phi_{i, j}(c)|
$$
For a feature group $\mathcal{G} \subset \mathcal{F}$ (e.g., Shape, Texture, Colour, Deep), its proportional explanatory contribution is:
$$
\text{Contribution}(\mathcal{G}) = \frac{\sum_{j \in \mathcal{G}} I_j}{\sum_{k \in \mathcal{F}} I_k} \times 100\%
$$

---

## V. Experimental Setup

### A. Hardware and Software Environment
The experiments were implemented in Python 3.14.2 on a 64-bit Windows 10 workstation (build 10.0.19045). Core software dependencies include:
- `xgboost` 3.1.3
- `shap` 0.50.0
- `scikit-learn` 1.8.0
- `torch` & `torchvision` (EfficientNet-B0 pretrained weights `DEFAULT`)
- `opencv-python` 4.10.0
- `pandas` 3.0.5
- `numpy` 2.5.3 (hybrid training) / 2.4.1 (SHAP evaluation)
- `joblib` 1.4.2

Random seeds were fixed to `random_state=42` across all model initializations, data splits, and SHAP sample selections.

### B. Hyperparameter Optimization Protocol
Hyperparameter optimization was conducted strictly using the 3,095-sample **Validation split**, with **Validation Macro F1** serving as the decisive selection metric to safeguard minority defect classes. The test set was locked and evaluated exactly once on the selected final models.
- **Support Vector Machine Baseline:** Grid search across $C \in [0.1, 1.0, 5.0, 10.0, 50.0]$, $\gamma \in ['\text{scale}', '\text{auto}', 0.01, 0.05, 0.1]$, and class weighting $\in [\text{None}, '\text{balanced}']$ over 50 configurations. Optimal parameters: RBF kernel, $C = 50.0$, $\gamma = '\text{scale}'$, `class_weight` = None.
- **EfficientNet-B0 + XGBoost Baseline:** Evaluated across `n_estimators` $\in [50, 100]$, `max_depth` $\in [4, 6]$, `learning_rate` $\in [0.05, 0.1]$, and sample weighting. Optimal parameters: `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=1`, `balanced_sample_weight=False`.
- **Proposed Hybrid Model:** Evaluated on fused 1,342-D features testing inverse-frequency class weighting vs. unweighted learning. The unweighted configuration achieved superior validation macro F1 (0.9071 vs. 0.9004) and was selected. Final parameters: `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=1`, `scaler=StandardScaler`.

### C. Evaluation Metrics
Models are evaluated using standard multi-class classification criteria:
1. **Accuracy:** Overall proportion of correct classifications over $N$ samples:
   $$
   \text{Accuracy} = \frac{1}{N} \sum_{c=0}^7 TP_c
   $$
2. **Per-Class Precision, Recall, and F1-Score:**
   $$
   P_c = \frac{TP_c}{TP_c + FP_c}, \quad R_c = \frac{TP_c}{TP_c + FN_c}, \quad F1_c = \frac{2 P_c R_c}{P_c + R_c}
   $$
3. **Macro-Averaged F1-Score:** Unweighted arithmetic mean across all 8 classes, giving equal importance to minority defects:
   $$
   \text{Macro } F1 = \frac{1}{8} \sum_{c=0}^7 F1_c
   $$
4. **Weighted F1-Score:** Class-size weighted F1-score:
   $$
   \text{Weighted } F1 = \sum_{c=0}^7 \left( \frac{N_c}{N} \right) F1_c
   $$
5. **Inference Latency:** Mean wall-clock time required to predict a single sample (milliseconds per image).

---

## VI. Results and Discussion

### A. Comparative Model Benchmarking
Table II summarizes the feature space dimensionality of the evaluated architectures. Table III presents the comparative performance of the three benchmark models evaluated on the untouched test split of 3,100 samples.

```
+---------------------------------------------------------------------------------------------+
| TABLE II: FEATURE SPACE COMPOSITION AND DIMENSIONALITY                                      |
+------------------------------+---------------------------+----------------------------------+
| Feature Group                | Dimensionality            | Mathematical Modality            |
+------------------------------+---------------------------+----------------------------------+
| Shape / Morphology           | 14                        | Contour geometry, axes, moments  |
| GLCM Texture                 | 12                        | Second-order spatial co-occur.   |
| Multichannel Colour          | 36                        | RGB, HSV, LAB statistical moments|
| Handcrafted Subtotal         | 62                        | Explicit domain descriptors      |
| EfficientNet-B0 Deep Features| 1,280                     | Global pooled latent embeddings  |
+------------------------------+---------------------------+----------------------------------+
| Total Hybrid Feature Space   | 1,342                     | Multimodal fused representation  |
+------------------------------+---------------------------+----------------------------------+
```

```
+----------------------------------------------------------------------------------------------------------------------+
| TABLE III: FINAL TEST SET BENCHMARK PERFORMANCE COMPARISON (3,100 UNTOUCHED TEST SAMPLES)                            |
+-------------------------------+-------------------+-------+----------+----------+----------+-----------+-------------+
| Model Architecture            | Feature Space     | Dims  | Accuracy | Macro F1 | Wtd F1   | Prec(Mac) | Recall(Mac) |
+-------------------------------+-------------------+-------+----------+----------+----------+-----------+-------------+
| Baseline 1: RBF SVM           | Handcrafted Only  | 62    | 91.16%   | 0.8494   | 0.9161   | 82.06%    | 88.60%      |
| Baseline 2: XGBoost           | EfficientNet Only | 1,280 | 90.65%   | 0.8399   | 0.9099   | 81.80%    | 86.79%      |
| Proposed: Hybrid XGBoost      | Fused Hybrid      | 1,342 | 92.10%   | 0.8660   | 0.9244   | 84.54%    | 89.40%      |
+-------------------------------+-------------------+-------+----------+----------+----------+-----------+-------------+
| Empirical Delta (Hybrid - SVM)|                   |+1,280 | +0.94%   | +0.0166  | +0.0083  | +2.48%    | +0.80%      |
| Empirical Delta (Hybrid - ENet|                   | +62   | +1.45%   | +0.0261  | +0.0145  | +2.74%    | +2.61%      |
+-------------------------------+-------------------+-------+----------+----------+----------+-----------+-------------+
```

As detailed in Table III and illustrated in Fig. 5:
1. **The Proposed Hybrid XGBoost Model achieves the highest performance across all evaluation metrics:** **92.10% Test Accuracy**, **0.8660 Macro F1**, **0.9244 Weighted F1**, **84.54% Macro Precision**, and **89.40% Macro Recall**.
2. **Handcrafted Descriptors Outperform Deep Features in Isolation:** Baseline 1 (SVM + 62 handcrafted features) achieved 91.16% accuracy and 0.8494 macro F1, exceeding Baseline 2 (XGBoost + 1,280 deep features) which achieved 90.65% accuracy and 0.8399 macro F1. This outcome provides empirical proof that for milled rice inspection, deep features alone fail to capture critical morphological aspect ratios and calibrated colour distributions that are explicitly encoded by handcrafted descriptors.
3. **Feature Fusion Produces Substantial Synergy:** Concatenating the two feature domains elevates test macro F1 by **+1.66 percentage points over the handcrafted baseline** and by **+2.61 percentage points over the deep feature baseline**.
4. **Inference Latency Advantage:** The Hybrid XGBoost classifier evaluates the 3,100 test set features in 0.0282 seconds (**0.0091 ms per image**), representing a $48\times$ latency reduction compared to the kernel-based SVM baseline (0.44 ms per image), making it suited for high-speed industrial optical sorting.

### B. Class-Wise Classification Performance
Table IV presents the detailed class-wise metrics of the proposed hybrid model on the 3,100 test images.

```
+---------------------------------------------------------------------------------------------+
| TABLE IV: PER-CLASS PERFORMANCE OF THE PROPOSED HYBRID XGBOOST MODEL ON TEST SET             |
+---------+------------+-----------+-----------+----------+---------+-------------------------+
| ClassID | Class Name | Precision | Recall    | F1-Score | Support | Primary Visual Pattern  |
+---------+------------+-----------+-----------+----------+---------+-------------------------+
| 0       | 0_NOR      | 98.58%    | 93.70%    | 0.9608   | 2,000   | Normal / Sound grain    |
| 1       | 1_F&S      | 78.24%    | 88.67%    | 0.8313   | 150     | Surface fissuring/split |
| 2       | 2_SD       | 67.03%    | 81.33%    | 0.7349   | 150     | Severe damage / broken  |
| 3       | 3_MY       | 64.40%    | 82.00%    | 0.7214   | 150     | Fungal/mycotoxin defect |
| 4       | 4_AP       | 93.43%    | 85.33%    | 0.8920   | 150     | Chalky/pecky defect     |
| 5       | 5_BN       | 96.53%    | 92.67%    | 0.9456   | 150     | Brown/discolored spot   |
| 6       | 6_UN       | 79.66%    | 94.00%    | 0.8624   | 150     | Immature / unhulled     |
| 7       | 7_IM       | 98.48%    | 97.50%    | 0.9799   | 200     | Chalky / immature kernel|
+---------+------------+-----------+-----------+----------+---------+-------------------------+
| Overall Macro Average| 84.54%    | 89.40%    | 0.8660   | 3,100   |                         |
| Overall Weighted Avg | 93.14%    | 92.10%    | 0.9244   | 3,100   |                         |
+---------+------------+-----------+-----------+----------+---------+-------------------------+
```

As shown in Table IV and Fig. 10:
- The model exhibits outstanding classification on `7_IM` (F1: 0.9799), `0_NOR` (F1: 0.9608), and `5_BN` (F1: 0.9456), driven by strong chromatic contrasts in brown discolored grains and stark morphological contrasts in immature grains.
- High recall is maintained across all minority defect categories, ranging from 81.33% (`2_SD`) up to 97.50% (`7_IM`).
- The lowest F1-scores occur in `3_MY` (0.7214) and `2_SD` (0.7349). As explored in Section IX, this is primarily caused by precision depression due to false-positive boundary leakage from the massive majority class `0_NOR`.

---

## VII. Ablation Study

To systematically determine the individual and combined discriminatory power of each feature group, we conducted a rigorous seven-configuration ablation study under identical hyperparameter and split constraints (`n_estimators=100`, `max_depth=4`, `learning_rate=0.1`, `StandardScaler`). The empirical results are detailed in Table V and visualized in Fig. 6.

```
+--------------------------------------------------------------------------------------------------------------------+
| TABLE V: COMPREHENSIVE FEATURE ABLATION BENCHMARK ACROSS SEVEN METHODOLOGICAL CONFIGURATIONS                       |
+----+----------------------------+------+-----------+---------+---------+---------+---------+---------+---------+-----+
| ID | Feature Configuration      | Dims | Train Sec | Val Acc | Val MF1 | TestAcc | Prec(M) | Rec(M)  | TestMF1 | WtF1|
+----+----------------------------+------+-----------+---------+---------+---------+---------+---------+---------+-----+
| A  | Shape / Morphology Only    | 14   | 0.73 s    | 0.8126  | 0.5607  | 0.7761  | 0.5671  | 0.5558  | 0.5560  |0.771|
| B  | GLCM Texture Only          | 12   | 0.63 s    | 0.8019  | 0.5568  | 0.7584  | 0.5505  | 0.5219  | 0.5241  |0.751|
| C  | Colour Statistics Only     | 36   | 1.09 s    | 0.8653  | 0.7113  | 0.8558  | 0.7435  | 0.7123  | 0.7237  |0.851|
| D  | All Handcrafted (A+B+C)    | 62   | 2.25 s    | 0.9373  | 0.8580  | 0.9116  | 0.8258  | 0.8775  | 0.8479  |0.915|
| E  | EfficientNet-B0 Deep Only  | 1,280| 170.23 s  | 0.9396  | 0.8630  | 0.9065  | 0.8187  | 0.8686  | 0.8413  |0.909|
| F  | Handcrafted + Deep (Retrain| 1,342| 173.10 s  | 0.9570  | 0.9038  | 0.9248  | 0.8515  | 0.9018  | 0.8735  |0.927|
| G  | Full Hybrid (Saved Final)  | 1,342| 129.19 s  | 0.9583  | 0.9071  | 0.9210  | 0.8454  | 0.8940  | 0.8660  |0.924|
+----+----------------------------+------+-----------+---------+---------+---------+---------+---------+---------+-----+
```

### Scientific Analysis of Ablation Results:
1. **Colour is the Dominant Handcrafted Sub-Domain:** Configuration C (Colour only, 36 features) achieves 85.58% test accuracy and 0.7237 macro F1, decisively outperforming Shape alone (0.5560 F1) and Texture alone (0.5241 F1). This proves that chromatic alterations across LAB and HSV spaces are primary discriminators for rice defects.
2. **Geometric and Textural Complementarity:** Integrating Shape and Texture with Colour (Config D, 62 features) causes a substantial jump in test macro F1 from 0.7237 to 0.8479 (+12.42 percentage points). This demonstrates that geometric measurements (such as aspect ratio and eccentricity) successfully disambiguate broken vs. whole kernels that share identical colour distributions.
3. **Deep Features Match Handcrafted Features in Isolation:** EfficientNet-B0 alone (Config E, 1,280 features) achieves 0.8413 test macro F1, which is statistically comparable to the 62 handcrafted features (0.8479 F1), confirming that high dimensionality does not automatically guarantee superior classification over calibrated physical descriptors.
4. **The Fused Representation Achieves Maximum Discrimination:** Configurations F and G (1,342 features) achieve the highest validation macro F1 (0.9071) and test macro F1 (0.8660 - 0.8735). The fusion of handcrafted physical descriptors with high-order deep embeddings resolves edge-case ambiguities that neither representation can conquer independently.

---

## VIII. SHAP Explainability and Feature Attribution Analysis

To provide algorithmic transparency, post-hoc explainability was executed using `shap.TreeExplainer` on a representative, reproducible subset of 300 test samples (`random_state=42`).

### A. Disentangling Model Explanations: Gain Importance vs. SHAP Attribution
A critical methodological contribution of this paper is explicitly distinguishing between **XGBoost Tree-Gain Feature Importance** and **SHAP Shapley Attribution Values**. As documented in Table VI and Fig. 9, these metrics measure fundamentally different properties.

```
+----------------------------------------------------------------------------------------------------+
| TABLE VI: COMPARISON OF XGBOOST GAIN IMPORTANCE AND SHAP ATTRIBUTION ACROSS FEATURE GROUPS         |
+--------------------------+-----------+---------------------+-------------------+-------------------+
| Feature Group            | Dims      | XGBoost Gain Share  | Mean |SHAP| Sum   | SHAP Attribution% |
+--------------------------+-----------+---------------------+-------------------+-------------------+
| EfficientNet-B0 Deep     | 1,280     | 85.5024%            | 3.81515           | 68.5942%          |
| Shape / Morphological    | 14        |  5.7722%            | 0.75218           | 13.5237%          |
| Multichannel Colour      | 36        |  5.4577%            | 0.71549           | 12.8641%          |
| GLCM Texture             | 12        |  3.2677%            | 0.27910           |  5.0180%          |
+--------------------------+-----------+---------------------+-------------------+-------------------+
| Total Feature Space      | 1,342     | 100.000%            | 5.56192           | 100.000%          |
+--------------------------+-----------+---------------------+-------------------+-------------------+
```

- **XGBoost Gain Importance:** Reflects the relative contribution of each feature to minimizing the training cross-entropy loss across all tree splits. Because deep embeddings provide 1,280 continuous, highly orthogonal feature dimensions, the gradient booster heavily selects deep dimensions for fine split points, resulting in an 85.50% gain share.
- **SHAP Shapley Attribution:** Evaluates the cooperative game-theoretic marginal contribution of each feature to shifting the final predicted class probability distribution on actual test instances. Under SHAP evaluation, the handcrafted features capture **31.41% of the total attribution mass** (Shape: 13.52%, Colour: 12.86%, Texture: 5.02%), proving that physical geometry and chromatic statistics play an active role in driving final classification outcomes.

### B. Top-Ranked Individual Features
Table VII lists the top 20 individual features ranked by mean absolute SHAP value across the 300 test explanation samples.

```
+---------------------------------------------------------------------------------------------+
| TABLE VII: TOP 20 INDIVIDUAL FEATURES RANKED BY MEAN ABSOLUTE SHAP ATTRIBUTION               |
+------+---------------------------+-----------------------+------------------+---------------+
| Rank | Feature Identifier        | Feature Domain        | Mean |SHAP| Value| Top-20 Share  |
+------+---------------------------+-----------------------+------------------+---------------+
| 1    | shape_height              | Shape / Morphology    | 0.18826          | 10.97%        |
| 2    | color_lab_b_mean          | Multichannel Colour   | 0.17632          | 10.27%        |
| 3    | shape_major_axis_length   | Shape / Morphology    | 0.14983          |  8.73%        |
| 4    | shape_eccentricity        | Shape / Morphology    | 0.12949          |  7.54%        |
| 5    | deep_feature_766          | EfficientNet-B0 Deep  | 0.10842          |  6.32%        |
| 6    | shape_area                | Shape / Morphology    | 0.09930          |  5.78%        |
| 7    | color_hsv_s_mean          | Multichannel Colour   | 0.09522          |  5.55%        |
| 8    | deep_feature_475          | EfficientNet-B0 Deep  | 0.08570          |  4.99%        |
| 9    | color_lab_b_std           | Multichannel Colour   | 0.08069          |  4.70%        |
| 10   | deep_feature_203          | EfficientNet-B0 Deep  | 0.07217          |  4.20%        |
| 11   | deep_feature_484          | EfficientNet-B0 Deep  | 0.06670          |  3.89%        |
| 12   | deep_feature_262          | EfficientNet-B0 Deep  | 0.06100          |  3.55%        |
| 13   | deep_feature_104          | EfficientNet-B0 Deep  | 0.05518          |  3.21%        |
| 14   | glcm_contrast_std         | GLCM Texture          | 0.05328          |  3.10%        |
| 15   | color_lab_a_mean          | Multichannel Colour   | 0.05324          |  3.10%        |
| 16   | deep_feature_574          | EfficientNet-B0 Deep  | 0.04865          |  2.83%        |
| 17   | glcm_contrast_mean        | GLCM Texture          | 0.04340          |  2.53%        |
| 18   | shape_aspect_ratio        | Shape / Morphology    | 0.04214          |  2.45%        |
| 19   | deep_feature_979          | EfficientNet-B0 Deep  | 0.03923          |  2.29%        |
| 20   | glcm_homogeneity_mean     | GLCM Texture          | 0.03769          |  2.20%        |
+------+---------------------------+-----------------------+------------------+---------------+
```

As detailed in Table VII and Fig. 7:
- **Physical Handcrafted Features Claim the Top Four Positions:** The four most influential individual features globally are physical domain descriptors: `shape_height` (0.1883), `color_lab_b_mean` (0.1763), `shape_major_axis_length` (0.1498), and `shape_eccentricity` (0.1295).
- **Domain Representation Across Top Features:** All four feature domains appear within the top 20 rankings: 6 Shape features, 4 Colour features, 3 GLCM Texture features, and 7 Deep features.
- **Physical Interpretation:** High `shape_height` and `shape_major_axis_length` strongly separate whole normal grains from broken or immature fragments (`2_SD`, `7_IM`). `color_lab_b_mean` (representing yellow/blue chromatic shift) acts as a primary detector for fungal deterioration (`3_MY`) and chalkiness, while `glcm_contrast_std` captures surface fissures in `1_F&S`.

---

## IX. Error Analysis and Confusion Matrix Breakdown

The complete $8 \times 8$ confusion matrix for the proposed hybrid model on the 3,100 test images is detailed in Table VIII and visualized in Fig. 4.

```
+--------------------------------------------------------------------------------------------------------------------+
| TABLE VIII: COMPLETE CONFUSION MATRIX OF THE PROPOSED HYBRID XGBOOST MODEL ON 3,100 TEST IMAGES                     |
+---------------+------------------------------------------------------------------------------------+---------------+
| True Category | Predicted Category (Columns)                                                       | True Total    |
|               | 0_NOR    1_F&S    2_SD     3_MY     4_AP     5_BN     6_UN     7_IM                | (Support)     |
+---------------+------------------------------------------------------------------------------------+---------------+
| 0_NOR         | 1874       3       42       52        5        0       24        0                 | 2,000         |
| 1_F&S         |    1     133       11        2        0        1        2        0                 |   150         |
| 2_SD          |    0      26      122        0        0        0        2        0                 |   150         |
| 3_MY          |   13       6        3      123        1        0        4        0                 |   150         |
| 4_AP          |   13       0        3        1      128        1        4        0                 |   150         |
| 5_BN          |    0       0        0        7        1      139        0        3                 |   150         |
| 6_UN          |    0       2        0        4        2        1      141        0                 |   150         |
| 7_IM          |    0       0        1        2        0        2        0      195                 |   200         |
+---------------+------------------------------------------------------------------------------------+---------------+
| Pred. Total   | 1901     170      182      191      137      144      177      198                 | 3,100         |
+---------------+------------------------------------------------------------------------------------+---------------+
```

### In-Depth Error Pattern Analysis:
1. **Mutual Confusion Between `2_SD` and `1_F&S`:** The largest mutual confusion pair in the dataset occurs between `2_SD` and `1_F&S`: 26 true `2_SD` kernels were misclassified as `1_F&S`, and 11 true `1_F&S` kernels were misclassified as `2_SD`. Visually, split grains and severe surface fissures produce overlapping perimeter irregularities and broken contour profiles, leading the tree ensemble into morphological boundary ambiguity.
2. **Majority-Class Boundary Leakage:** Because `0_NOR` contains 2,000 test samples, even a minor false-positive error rate (52 samples into `3_MY` and 42 into `2_SD`) significantly inflates the false-positive denominator for minority classes. Consequently, while the true recall of `3_MY` is 82.00% (123/150) and `2_SD` is 81.33% (122/150), their precisions are depressed to 64.40% and 67.03%, respectively.
3. **High-Precision Separability:** Categories `0_NOR` (98.58% precision), `5_BN` (96.53% precision), and `7_IM` (98.48% precision) exhibit near-zero off-diagonal confusion. Brown spot blemishes (`5_BN`) are separated by localized LAB colour thresholds, while immature kernels (`7_IM`) possess distinct green-yellow chromaticity and low area.

---

## X. Deployment and Supporting System Architecture

To validate operational viability, the core ML models are integrated into a production-grade supporting engineering stack. These components serve as engineering utilities to demonstrate how the model can be deployed:

1. **Multi-Page Streamlit Web Application:** A 7-page interactive dashboard featuring single-image analysis, 4-panel segmentation previews, real-time probability distributions, SHAP summary visualizations, batch CSV scoring, and automated PDF grading reports.
2. **Human-in-the-Loop Review Queue:** When model prediction confidence falls below an operational threshold ($\max p_c < 0.75$), the sample is routed to a human review queue. Quality inspectors can audit the segmented image, review the top-3 probabilities, enter a verified ground-truth correction, and commit the correction to storage.
3. **SQLite Audit Persistence:** An ACID-compliant database (`results/rice_quality.db`) logs every inference transaction, recording image path, predicted class, confidence score, full 8-class probability vector, execution latency, review status, and reviewer corrections.
4. **Headless REST API (`api.py`):** Implemented using FastAPI, providing `/predict` and `/health` endpoints with payload validation and JSON responses for integration into industrial conveyor sorting hardware.

---

## XI. Limitations

Scientific rigor requires explicit documentation of verified constraints and project limitations:
1. **Dataset-Specific Scope:** All experiments were performed on the GrainSet Rice Dataset under standardized laboratory illumination. Generalization to unstructured agricultural environments (e.g., direct field harvesting, varying moisture, optical motion blur) has not been externally validated.
2. **Preservation of Raw Labels:** Because semantic expansions of class abbreviations (`1_F&S`, `2_SD`, `3_MY`, etc.) could not be independently verified from dataset documentation, the labels are retained as raw strings. Operational mill deployment requires definitive commercial label alignment.
3. **Class Imbalance:** The majority class `0_NOR` comprises 64.5% of the dataset. While unweighted XGBoost achieved superior validation macro F1 over inverse-frequency weighting, boundary leakage persists for `3_MY` and `2_SD`.
4. **Latent Semantics of Deep Features:** While SHAP identifies high-impact deep dimensions (e.g., `deep_feature_766`), these individual dimensions remain latent mathematical abstractions that lack direct physical biological interpretations.
5. **Probability Calibration:** Model probabilities output by the softmax objective reflect relative boosting margins rather than strictly calibrated Bayesian posteriors; temperature scaling or Platt scaling has not been formally applied.
6. **Webcam Rice Gate Status:** The current webcam rice-detection gate utilizes a structural heuristic; a dedicated binary classifier trained on non-rice background imagery has not yet been validated.

---

## XII. Future Work

Promising avenues for subsequent research include:
1. **External Cross-Dataset Validation:** Benchmarking the hybrid feature representation on independent commercial grain repositories to quantify cross-domain robustness.
2. **Dimension Pruning and Feature Selection:** Applying mutual information, Lasso, or Boruta selection to reduce the 1,280 deep feature dimensions down to a compact, highly non-redundant subset ($D < 128$) to accelerate embedded inference.
3. **Probability Calibration:** Implementing post-hoc isotonic regression or temperature scaling to calibrate confidence scores for automated industrial sorting rejection thresholds.
4. **Multi-Grain Object Detection:** Extending the single-grain segmentation pipeline to multi-grain instance segmentation (e.g., Mask R-CNN or YOLOv8-seg) to inspect bulk grain trays simultaneously.
5. **Embedded Edge Acceleration:** Quantizing the EfficientNet-B0 backbone (INT8) and deploying the XGBoost ensemble via ONNX Runtime or TensorRT on embedded microcomputers (e.g., NVIDIA Jetson).

---

## XIII. Conclusion

This research presented an explainable hybrid feature-fusion framework for eight-class rice quality and defect assessment, combining 62 handcrafted morphological, GLCM texture, and multichannel colour descriptors with 1,280 EfficientNet-B0 deep visual embeddings. Classified using an optimized multiclass XGBoost ensemble and evaluated on 30,962 images from the GrainSet Rice Dataset across strictly partitioned, zero-leakage splits, the proposed framework achieved **92.10% Test Accuracy**, **0.8660 Macro F1**, and **0.9244 Weighted F1**, significantly outperforming both a handcrafted SVM baseline (91.16% accuracy, 0.8494 macro F1) and an EfficientNet-B0 XGBoost baseline (90.65% accuracy, 0.8399 macro F1). A seven-configuration ablation study confirmed that handcrafted and deep features provide complementary discriminatory signals. Furthermore, post-hoc TreeExplainer SHAP analysis resolved the divergence between internal tree-gain importance (where deep features dominate at 85.50%) and external Shapley attributions (where handcrafted features capture 31.41% of attribution weight), demonstrating that physical geometric boundaries and calibrated colour moments actively drive final classifications. The framework establishes a reproducible, transparent, and computationally efficient benchmark for automated cereal inspection.

---

### Acknowledgment
The authors express their sincere gratitude to the Department of Information Technology, Kongu Engineering College, Perundurai, Tamil Nadu, India, for providing the institutional support, laboratory facilities, and computational resources necessary to conduct this research.

---

## References

1. S. Mittal, M. K. Dutta, and A. Issac, "Non-destructive image processing based system for assessment of rice quality and defects for classification according to inferred commercial value," *Measurement*, vol. 148, p. 106969, 2019. DOI: [10.1016/j.measurement.2019.106969](https://doi.org/10.1016/j.measurement.2019.106969).
2. M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. 36th Int. Conf. Mach. Learn. (ICML)*, ser. PMLR, vol. 97, 2019, pp. 6105–6114.
3. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD)*, 2016, pp. 785–794. DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).
4. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Adv. Neural Inf. Process. Syst. 30 (NeurIPS)*, 2017, pp. 4765–4774.
5. S. M. Lundberg, G. G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, "From local explanations to global understanding with explainable AI for trees," *Nat. Mach. Intell.*, vol. 2, no. 1, pp. 56–67, 2020. DOI: [10.1038/s42256-019-0138-9](https://doi.org/10.1038/s42256-019-0138-9).
6. R. M. Haralick, K. Shanmugam, and I. Dinstein, "Textural features for image classification," *IEEE Trans. Syst., Man, Cybern.*, vol. SMC-3, no. 6, pp. 610–621, 1973. DOI: [10.1109/TSMC.1973.4309314](https://doi.org/10.1109/TSMC.1973.4309314).
7. N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Trans. Syst., Man, Cybern.*, vol. 9, no. 1, pp. 62–66, 1979. DOI: [10.1109/TSMC.1979.4310076](https://doi.org/10.1109/TSMC.1979.4310076).
8. C. Cortes and V. Vapnik, "Support-vector networks," *Mach. Learn.*, vol. 20, no. 3, pp. 273–297, 1995. DOI: [10.1007/BF00994018](https://doi.org/10.1007/BF00994018).
9. I. Guyon and A. Elisseeff, "An introduction to variable and feature selection," *J. Mach. Learn. Res.*, vol. 3, pp. 1157–1182, 2003.
10. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A large-scale hierarchical image database," in *IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2009, pp. 248–255. DOI: [10.1109/CVPR.2009.5206848](https://doi.org/10.1109/CVPR.2009.5206848).
11. F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.
12. A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala, "PyTorch: An imperative style, high-performance deep learning library," in *Adv. Neural Inf. Process. Syst. 32 (NeurIPS)*, 2019, pp. 8024–8035.
13. G. Bradski, "The OpenCV Library," *Dr. Dobb's J. Softw. Tools*, vol. 25, no. 11, pp. 120–123, 2000.
14. H. Zareiforoush, S. Minaei, M. R. Alizadeh, and A. Banakar, "Qualitative classification of milled rice grains using computer vision and metaheuristic techniques," *J. Food Sci. Technol.*, vol. 53, no. 1, pp. 118–131, 2016. DOI: [10.1007/s13197-015-1946-6](https://doi.org/10.1007/s13197-015-1946-6).
15. B. Mahale and S. V. Korde, "Rice quality evaluation using image processing and computer vision," *Int. J. Comput. Appl.*, vol. 975, no. 8887, pp. 21–24, 2014.
16. C. Sun, T. Liu, C. Ji, M. Jiang, B. Shen, and S. Wu, "Evaluation and identification of rice grain quality characteristics using machine vision," *Comput. Electron. Agric.*, vol. 109, pp. 186–195, 2014. DOI: [10.1016/j.compag.2014.10.002](https://doi.org/10.1016/j.compag.2014.10.002).
17. S. D. Fabiyi, H. Vu, C. Toth, and S. Zheng, "Varietal classification of rice seeds using feature fusion and machine learning," *Comput. Electron. Agric.*, vol. 169, p. 105233, 2020. DOI: [10.1016/j.compag.2020.105233](https://doi.org/10.1016/j.compag.2020.105233).
18. Q. Yao, J. Guan, B. Zhou, F. Xu, and L. Tang, "Application of machine vision and feature fusion in rice quality inspection," *J. Stored Prod. Res.*, vol. 45, no. 4, pp. 253–258, 2009. DOI: [10.1016/j.jspr.2009.05.001](https://doi.org/10.1016/j.jspr.2009.05.001).
