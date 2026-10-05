"""
build_docx_paper.py
===================
Generates a complete, publication-quality research paper in Microsoft Word (.docx) format
using python-docx, embedding all verified tables, equations, citations, and figures.
"""

from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = Path("c:/Rice classifier final project")
OUTPUT_DOCX = PROJECT_ROOT / "paper" / "rice_quality_research_paper.docx"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"

doc = docx.Document()

# Configure page margins (1 inch on all sides)
for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

# Colors
COLOR_PRIMARY = RGBColor(15, 42, 74)     # Navy Blue
COLOR_SECONDARY = RGBColor(33, 82, 138)  # Slate Blue
COLOR_TEXT = RGBColor(33, 33, 33)        # Charcoal Body Text
COLOR_MUTED = RGBColor(100, 100, 100)    # Gray
HEX_HEADER_BG = "1F4E79"
HEX_ALT_ROW = "F2F4F7"
HEX_BORDER = "CCCCCC"

def set_cell_shading(cell, color_hex):
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def add_title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(20)
    run.font.bold = True
    run.font.color.rgb = COLOR_PRIMARY
    return p

def add_authors():
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    r1 = p.add_run("Dharshan B¹")
    r1.font.bold = True
    r1.font.size = Pt(11.5)
    r1.font.name = "Calibri"
    
    r_sep = p.add_run("   |   ")
    r_sep.font.color.rgb = COLOR_MUTED
    
    r2 = p.add_run("Dinesh G L¹")
    r2.font.bold = True
    r2.font.size = Pt(11.5)
    r2.font.name = "Calibri"

    p_mentor = doc.add_paragraph()
    p_mentor.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_mentor.paragraph_format.space_before = Pt(0)
    p_mentor.paragraph_format.space_after = Pt(4)
    rm = p_mentor.add_run("Mentored by: Ms. Sripriya S")
    rm.font.italic = True
    rm.font.size = Pt(10.5)
    rm.font.name = "Calibri"
    rm.font.color.rgb = COLOR_SECONDARY

    p_aff = doc.add_paragraph()
    p_aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_aff.paragraph_format.space_before = Pt(0)
    p_aff.paragraph_format.space_after = Pt(18)
    ra = p_aff.add_run("¹Department of Information Technology, Kongu Engineering College, Perundurai, Tamil Nadu, India")
    ra.font.size = Pt(10)
    ra.font.name = "Calibri"
    ra.font.color.rgb = COLOR_MUTED

def add_abstract(abstract_text, keywords_text):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_shading(cell, "F8F9FA")
    set_cell_margins(cell, top=160, bottom=160, left=220, right=220)
    
    # Border styling
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:top w:val="single" w:sz="6" w:space="0" w:color="0288D1"/><w:left w:val="single" w:sz="18" w:space="0" w:color="0288D1"/><w:bottom w:val="single" w:sz="6" w:space="0" w:color="0288D1"/><w:right w:val="single" w:sz="6" w:space="0" w:color="0288D1"/></w:tcBorders>')
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    
    r_title = p.add_run("Abstract— ")
    r_title.bold = True
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(9.5)
    r_title.font.color.rgb = COLOR_PRIMARY
    
    r_body = p.add_run(abstract_text)
    r_body.font.name = "Calibri"
    r_body.font.size = Pt(9.5)
    
    p2 = cell.add_paragraph()
    p2.paragraph_format.space_before = Pt(4)
    p2.paragraph_format.space_after = Pt(0)
    r_kw = p2.add_run("Index Terms— ")
    r_kw.bold = True
    r_kw.font.name = "Calibri"
    r_kw.font.size = Pt(9.5)
    r_kw.font.color.rgb = COLOR_PRIMARY
    
    r_kwt = p2.add_run(keywords_text)
    r_kwt.italic = True
    r_kwt.font.name = "Calibri"
    r_kwt.font.size = Pt(9.5)
    
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(12)
    p_spacer.paragraph_format.space_after = Pt(0)

def add_heading_1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = COLOR_PRIMARY
    return p

def add_heading_2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(11.5)
    run.font.bold = True
    run.font.italic = True
    run.font.color.rgb = COLOR_SECONDARY
    return p

def add_paragraph(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(10.5)
    run.font.color.rgb = COLOR_TEXT
    return p

def add_bullet(text, level=0):
    p = doc.add_paragraph(style='List Bullet')
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(10)
    run.font.color.rgb = COLOR_TEXT
    return p

def add_figure(image_path, caption_text, width_inches=6.0):
    if not image_path.exists():
        print(f"Warning: Image not found: {image_path}")
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(4)
    run = p_img.add_run()
    run.add_picture(str(image_path), width=Inches(width_inches))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(0)
    p_cap.paragraph_format.space_after = Pt(12)
    r_cap = p_cap.add_run(caption_text)
    r_cap.font.name = "Calibri"
    r_cap.font.size = Pt(9.5)
    r_cap.font.italic = True
    r_cap.font.color.rgb = COLOR_SECONDARY

def create_table(headers, data, col_widths=None):
    table = doc.add_table(rows=len(data) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # Format header row
    hdr_row = table.rows[0]
    for idx, header in enumerate(headers):
        cell = hdr_row.cells[idx]
        set_cell_shading(cell, HEX_HEADER_BG)
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(header)
        run.bold = True
        run.font.name = "Calibri"
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(255, 255, 255)

    # Format data rows
    for r_idx, row_data in enumerate(data):
        row = table.rows[r_idx + 1]
        bg_color = HEX_ALT_ROW if (r_idx % 2 == 1) else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            set_cell_shading(cell, bg_color)
            set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            # Center or right-align numbers/percentages
            val_str = str(val)
            if any(char.isdigit() for char in val_str) and not any(letter in val_str for letter in ["NOR", "SD", "MY", "AP", "BN", "UN", "IM", "F&S", "Baseline", "Proposed", "EfficientNet", "Hybrid"]):
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(val_str)
            run.font.name = "Calibri"
            run.font.size = Pt(9)
            run.font.color.rgb = COLOR_TEXT

    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)

    # Add space after table
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(6)
    p_sp.paragraph_format.space_after = Pt(6)
    return table

print("Building research paper document...")

# Title & Authors
add_title("An Explainable Hybrid Feature Fusion Framework for Rice Quality and Defect Classification Using EfficientNet-B0 and XGBoost")
add_authors()

# Abstract
abstract_body = (
    "Automated visual inspection of milled rice is essential for high-throughput quality grading, defect detection, and commercial standardization. "
    "However, single-paradigm classification pipelines frequently suffer from inherent trade-offs: conventional handcrafted descriptors capture explicit, "
    "physically interpretable geometric and chromatic properties but struggle with subtle, high-order visual patterns, whereas deep convolutional networks "
    "learn expressive latent representations but omit fine-grained boundary metrics and operate as opaque black boxes. To resolve this tension, this paper presents "
    "a complete, publication-grade explainable hybrid feature-fusion framework for eight-class rice quality and defect assessment. The pipeline first standardizes raw rice "
    "grains via an adaptive segmentation workflow combining Gaussian smoothing, automatic Otsu thresholding, elliptical morphological filtering, and isotropic aspect-ratio "
    "padding to 224 x 224 pixels. From each segmented grain, we extract a 62-dimensional handcrafted vector comprising 14 morphological/shape descriptors, 12 Gray-Level "
    "Co-occurrence Matrix (GLCM) texture descriptors, and 36 multichannel colour statistics across RGB, HSV, and LAB spaces. Concurrently, a pretrained EfficientNet-B0 backbone "
    "extracts a 1,280-dimensional global visual embedding. The representations are concatenated into a 1,342-dimensional hybrid vector, standardized using a training-fitted "
    "StandardScaler to prevent data leakage, and classified using an optimized multiclass XGBoost classifier (n_estimators=100, max_depth=4, learning_rate=0.1). "
    "Rigorously evaluated on the GrainSet Rice Dataset across 30,962 images partitioned into strictly non-overlapping train (24,767), validation (3,095), and test (3,100) splits, "
    "the proposed hybrid model achieves 92.10% Test Accuracy, 0.8660 Macro F1, and 0.9244 Weighted F1, decisively surpassing both a tuned Support Vector Machine baseline "
    "operating on handcrafted features (91.16% accuracy, 0.8494 macro F1) and an XGBoost baseline operating on EfficientNet-B0 features alone (90.65% accuracy, 0.8399 macro F1). "
    "A controlled seven-configuration ablation study demonstrates that fusing handcrafted and deep features yields a +2.61 percentage point macro F1 improvement over deep features "
    "alone and a +1.66 percentage point gain over handcrafted features alone. Furthermore, post-hoc explainability using TreeExplainer SHAP across 300 test samples reveals that while "
    "deep features provide 68.59% of the overall attribution mass, handcrafted features capture 31.41% of the explanatory weight, with physical dimensions (shape_height, color_lab_b_mean, "
    "shape_major_axis_length, and shape_eccentricity) dominating the top-tier feature attributions. The framework is supported by a deployable human-in-the-loop review system, "
    "SQLite persistence, and REST API."
)
keywords_body = "Rice quality assessment, defect classification, hybrid feature fusion, EfficientNet-B0, XGBoost, explainable artificial intelligence (XAI), SHAP, computer vision, agricultural automation."
add_abstract(abstract_body, keywords_body)

# I. Introduction
add_heading_1("I. INTRODUCTION")
add_paragraph(
    "Rice (Oryza sativa) is the primary dietary staple for more than half of the global population. Accurate quality grading and defect assessment directly govern "
    "commercial pricing, milling efficiency, storage stability, and food safety standards [1]. In industrial rice milling and grain trade, batches must be categorized "
    "into distinct quality tiers based on physical head rice integrity, broken grain proportions, chalkiness, discoloration, immature kernels, and fungal or pest damage [1], [14]. "
    "Traditionally, grain inspection has relied on manual visual inspection by certified quality analysts. However, manual grain inspection is inherently subjective, labour-intensive, "
    "low-throughput, prone to operator visual fatigue, and vulnerable to substantial inter-assessor variability [1], [15]."
)
add_paragraph(
    "Over the past decade, computer vision and machine learning (ML) techniques have increasingly been deployed to automate non-destructive cereal inspection [1], [16]. "
    "Classical approaches typically rely on handcrafted feature engineering, where domain experts define explicit mathematical operators to quantify physical attributes: "
    "morphological dimensions (area, length, aspect ratio, circularity) to quantify broken or deformed grains; Gray-Level Co-occurrence Matrix (GLCM) texture metrics to evaluate "
    "surface roughness and fissures; and colour space statistics (RGB, HSV, LAB) to detect chalky bellies, yellowing, or pathogen lesions [1], [6], [14]. These handcrafted features "
    "possess the decisive advantage of direct physical interpretability and computational lightness, allowing classifiers such as Support Vector Machines (SVM) [8] to achieve "
    "strong baseline performance on controlled datasets. Nevertheless, handcrafted descriptors fail when confronted with complex, non-linear visual interactions—such as irregular "
    "superficial mould colonization, variegated defect pigmentation, and subtle milling variations—because manually engineered equations cannot anticipate all natural biological variabilities [17], [18]."
)
add_paragraph(
    "Conversely, modern Deep Learning (DL) architectures—particularly Convolutional Neural Networks (CNNs)—automatically extract rich, hierarchical visual representations "
    "directly from raw pixel arrays [2], [10]. Pretrained backbones such as EfficientNet-B0 [2] leverage inverted residual blocks and compound coefficient scaling to learn "
    "generalizable visual features while maintaining high parameter efficiency. Nonetheless, pure deep-learning paradigms exhibit two fundamental operational vulnerabilities "
    "in industrial quality grading: (1) loss of fine-grained physical measurements due to successive pooling and stride operations, and (2) the 'black box' opacity that obscures "
    "the biological rationale behind classification decisions, creating severe trust barriers for agricultural traders and regulatory auditors [4], [5]."
)
add_paragraph(
    "To resolve this tension, this paper presents a complete, publication-grade explainable hybrid feature-fusion framework for eight-class rice quality and defect assessment. "
    "By concatenating 62 domain-specific handcrafted descriptors with 1,280 deep embeddings from a pretrained EfficientNet-B0 backbone into a 1,342-dimensional vector, and classifying "
    "the fused representation using an optimized XGBoost gradient-boosted decision tree ensemble [3], the proposed methodology captures both explicit physical boundaries and complex "
    "latent visual semantics. Furthermore, to overcome the interpretability barrier, the framework incorporates post-hoc Shapley Additive Explanations (SHAP) [4], [5], providing "
    "mathematically grounded global and local attributions that disentangle the contributions of geometric, textural, chromatic, and deep feature subsets."
)

add_heading_2("Research Contributions")
add_paragraph("The specific, verified contributions of this research are:")
add_bullet("A standardized, reproducible preprocessing and segmentation pipeline combining Gaussian filtering, Otsu thresholding, elliptical morphological refinement, and isotropic aspect-ratio padding to 224 x 224 pixels (mean latency: 5.44 ms/image).")
add_bullet("A 62-dimensional multi-domain handcrafted feature engineering architecture (14 shape/morphology, 12 GLCM texture, and 36 colour statistics across RGB, HSV, and LAB spaces).")
add_bullet("An EfficientNet-B0 deep feature extractor extracting 1,280-dimensional global visual embeddings in torch inference mode without expensive end-to-end retraining.")
add_bullet("A 1,342-dimensional hybrid feature representation with strict leakage-free normalization via a training-fitted StandardScaler.")
add_bullet("Systematic empirical benchmarking demonstrating that the proposed Hybrid XGBoost model (92.10% test accuracy, 0.8660 macro F1) decisively outperforms both a tuned handcrafted SVM baseline (91.16% accuracy, 0.8494 macro F1) and an EfficientNet-B0 + XGBoost baseline (90.65% accuracy, 0.8399 macro F1).")
add_bullet("A controlled seven-configuration ablation study demonstrating that fusing handcrafted and deep features yields a +2.61 percentage point macro F1 improvement over deep features alone and a +1.66 percentage point gain over handcrafted features alone.")
add_bullet("Disentangled post-hoc model explainability using TreeExplainer SHAP resolving the divergence between internal tree-gain importance and external Shapley attributions.")
add_bullet("A supporting operational decision-support architecture featuring a Streamlit multi-page interface, automated human review queue for low-confidence predictions (p < 0.75), SQLite logging, and REST API.")

# II. Related Work
add_heading_1("II. RELATED WORK")
add_paragraph(
    "Automated agricultural grain classification has evolved through three distinct methodological epochs: classical morphology-based machine vision, deep learning feature extractors, "
    "and hybrid ensemble frameworks. Early automated grain assessment systems relied primarily on geometric morphology and flatbed scanner image acquisition [1], [14], [15]. "
    "Zareiforoush et al. [14] evaluated qualitative classification of milled rice grains using geometric features, reporting that length, width, aspect ratio, and projected area provided "
    "adequate discrimination for bulk dimensional grading. Mahale and Korde [15] applied thresholding and boundary tracing to classify Indian rice varieties, demonstrating that shape "
    "factors alone could identify whole vs. broken grains but failed when grains exhibited internal chalkiness or minor surface defects."
)
add_paragraph(
    "To capture surface texture, researchers integrated Gray-Level Co-occurrence Matrix (GLCM) formulations, originally introduced by Haralick et al. [6]. GLCM computes second-order "
    "statistical dependencies between pairs of pixels separated by distance d at orientation theta. Sun et al. [16] demonstrated that GLCM descriptors correlated with chalky endosperm "
    "voids and internal stress cracks. Colour statistics across multiple colour spaces were subsequently integrated by Mittal, Dutta, and Issac [1], who developed a non-destructive image "
    "processing framework for assessing rice quality and commercial value. Mittal et al. extracted geometric and colour histogram features to infer commercial quality grades. "
    "However, while Mittal et al. [1] utilized a specialized multi-feature scoring heuristic for commercial indexing, their framework relied exclusively on handcrafted descriptors and "
    "did not incorporate deep convolutional feature learning, gradient boosting, or post-hoc Shapley explainability."
)
add_paragraph(
    "The advent of deep Convolutional Neural Networks revolutionized agricultural image processing [10], [17]. Pretrained architectures such as EfficientNet-B0 (Tan and Le [2]) "
    "established compound scaling methods that uniformly scale depth, width, and input resolution, achieving state-of-the-art ImageNet representation with only 5.3 million parameters. "
    "However, Fabiyi et al. [17] noted that fine-tuning deep CNNs on grain datasets with strong class imbalance often leads to overfitting on majority categories and loss of fine geometric "
    "boundary details. Tree-based ensemble learning via XGBoost (Chen and Guestrin [3]) offers an ideal framework for tabular fused representations due to its exact second-order gradient boosting "
    "and explicit regularization. Furthermore, post-hoc explainability via SHAP TreeExplainer (Lundberg et al. [4], [5]) provides mathematically rigorous Shapley value attribution for tree ensembles, "
    "enabling transparent feature analysis."
)

# III. Dataset and Problem Formulation
add_heading_1("III. DATASET AND PROBLEM FORMULATION")
add_paragraph(
    "The empirical investigation is conducted on the GrainSet Rice Dataset as curated in the project repository. The dataset comprises a total of 30,962 high-resolution rice grain images. "
    "To guarantee scientific validity and prevent data leakage, the original dataset split was strictly preserved: Training Split: 24,767 images (79.99%), Validation Split: 3,095 images (10.00%), "
    "and Test Split: 3,100 images (10.01%). A filesystem audit confirmed zero missing files, zero corrupt images, and exactly zero overlapping filenames across splits."
)
add_paragraph(
    "The dataset encompasses eight visually distinct grain categories. Because semantic expansions are not independently verified in repository documentation, all classes are strictly "
    "identified by their original raw labels: 0_NOR, 1_F&S, 2_SD, 3_MY, 4_AP, 5_BN, 6_UN, and 7_IM. The complete distribution across splits is reported in Table I."
)

# Table I
add_heading_2("Table I: Dataset Distribution Across Evaluation Splits")
t1_headers = ["Class ID", "Raw Label", "Train Count (%)", "Val Count (%)", "Test Count (%)", "Total Count"]
t1_data = [
    ["0", "0_NOR", "15,980 (64.52%)", "2,020 (65.27%)", "2,000 (64.52%)", "20,000"],
    ["1", "1_F&S", " 1,191  (4.81%)", "  145  (4.68%)", "  150  (4.84%)", " 1,486"],
    ["2", "2_SD",  " 1,194  (4.82%)", "  156  (5.04%)", "  150  (4.84%)", " 1,500"],
    ["3", "3_MY",  " 1,223  (4.94%)", "  127  (4.10%)", "  150  (4.84%)", " 1,500"],
    ["4", "4_AP",  " 1,208  (4.88%)", "  142  (4.59%)", "  150  (4.84%)", " 1,500"],
    ["5", "5_BN",  " 1,198  (4.84%)", "  143  (4.62%)", "  150  (4.84%)", " 1,491"],
    ["6", "6_UN",  " 1,185  (4.78%)", "  150  (4.85%)", "  150  (4.84%)", " 1,485"],
    ["7", "7_IM",  " 1,588  (6.41%)", "  212  (6.85%)", "  200  (6.45%)", " 2,000"],
    ["Total", "All 8", "24,767 (100.0%)", "3,095 (100.0%)", "3,100 (100.0%)", "30,962"]
]
create_table(t1_headers, t1_data, [0.8, 1.0, 1.4, 1.4, 1.4, 1.0])

add_paragraph(
    "As shown in Table I, the dataset exhibits substantial class imbalance: the majority class 0_NOR represents roughly 64.5% of each partition, while defect classes 1_F&S through 6_UN "
    "each represent approximately 4.8%. Consequently, macro-averaged F1-score and class-wise metrics are essential alongside overall accuracy to evaluate performance on minority defects."
)

# IV. Proposed Methodology
add_heading_1("IV. PROPOSED METHODOLOGY")
add_paragraph(
    "The overall architectural pipeline of the proposed explainable hybrid feature-fusion framework is illustrated in Fig. 1. It comprises five sequential modules: "
    "(A) Image Preprocessing and Grain Segmentation, (B) Handcrafted Feature Extraction, (C) EfficientNet-B0 Deep Feature Extraction, (D) Hybrid Feature Fusion and Normalization, "
    "and (E) XGBoost Classification with SHAP Explainability."
)

add_figure(FIG_DIR / "fig1_overall_framework.png", "Fig. 1. End-to-End Architectural Pipeline of the Proposed Explainable Hybrid Feature-Fusion Framework.", width_inches=6.2)

add_heading_2("A. Image Preprocessing and Grain Segmentation")
add_paragraph(
    "Raw images possess variable resolutions (min: 206 x 135 px, max: 328 x 301 px, mean: 261.5 x 222.2 px). The preprocessing pipeline executes: "
    "(1) Conversion to RGB, Grayscale, HSV, and LAB colour representations; (2) Gaussian blur filtering with kernel K = (5, 5), sigma = 0 to smooth sensor noise; "
    "(3) Automatic Otsu thresholding to calculate the optimal binarization threshold T* minimizing intra-class variance; (4) Elliptical morphological closing followed by opening "
    "(kernel: 5 x 5 ellipse) to bridge internal chalky fissures and remove stray particles; (5) External contour extraction (cv2.RETR_EXTERNAL) with area filtering (threshold > 500 px) "
    "to isolate the primary grain mask; and (6) Bounding box cropping with a 5% margin expansion, followed by isotropic aspect-ratio preserved scaling (fill scale = 0.90) and centered "
    "padding onto a standardized 224 x 224 x 3 canvas. The measured average preprocessing latency across trials is 5.44 ms per image."
)

add_figure(FIG_DIR / "fig2_preprocessing_stages.png", "Fig. 2. Preprocessing and Grain Segmentation Workflow Across Representative Stages (Original -> Grayscale -> Mask -> Standardized Crop).", width_inches=6.0)
add_figure(FIG_DIR / "fig3_preprocessing_grid.png", "Fig. 3. Multi-Class Preprocessing and Segmentation Validation Grid across representative grain samples.", width_inches=6.0)

add_heading_2("B. Handcrafted Feature Extraction (62 Dimensions)")
add_paragraph(
    "Features are extracted strictly from segmented foreground grain pixels to eliminate background bias: "
    "(1) Shape/Morphology (14 dimensions): Area, perimeter, width, height, aspect ratio, extent, solidity, circularity, equivalent diameter, major/minor axis lengths, eccentricity, "
    "convex hull area, and bounding box area; (2) GLCM Texture (12 dimensions): Second-order co-occurrence matrices across 2 displacement distances (1, 2) and 4 orientations "
    "(0, 45, 90, 135 degrees) computed over 32 quantized gray levels exclusively on foreground pixels, generating mean and standard deviation for contrast, dissimilarity, homogeneity, "
    "energy, correlation, and ASM; (3) Multichannel Colour (36 dimensions): Mean, standard deviation, minimum, and maximum computed across 9 channels (RGB, HSV, LAB). Total: 14 + 12 + 36 = 62 dimensions."
)

add_heading_2("C. EfficientNet-B0 Deep Feature Extraction (1,280 Dimensions)")
add_paragraph(
    "Standardized 224 x 224 grain images are normalized with ImageNet statistics and passed through an ImageNet-pretrained EfficientNet-B0 backbone. The final 1,000-way linear classification head "
    "is replaced with an Identity operator. Under torch inference mode with batch size 32, the pooled global embedding yields a 1,280-dimensional deep feature representation per image."
)

add_heading_2("D. Hybrid Feature Fusion and Leakage-Safe Normalization")
add_paragraph(
    "The 62 handcrafted descriptors and 1,280 deep embeddings are concatenated into a 1,342-dimensional hybrid vector z = [h_1, ..., h_62, d_1, ..., d_1280]. "
    "A StandardScaler is fitted strictly on the 24,767 training instances and applied to validation, test, and inference samples to eliminate data leakage."
)

add_heading_2("E. XGBoost Classification and SHAP Explainability")
add_paragraph(
    "Classification is performed using an XGBoost gradient boosted ensemble optimizing multiclass cross-entropy loss (multi:softprob) with histogram tree construction. "
    "TreeExplainer SHAP computes exact Shapley values across 300 test instances to quantify the marginal attribution of geometric, textural, chromatic, and deep feature subsets."
)

# V. Experimental Setup
add_heading_1("V. EXPERIMENTAL SETUP")
add_paragraph(
    "Experiments were executed in Python 3.14.2 on Windows 10 (AMD64) using XGBoost 3.1.3, SHAP 0.50.0, scikit-learn 1.8.0, and PyTorch. Random seeds were fixed to 42 across all routines. "
    "Hyperparameters were selected strictly using Validation Macro F1. Baseline 1 (SVM) evaluated 50 configurations, identifying optimal parameters: RBF kernel, C=50.0, gamma='scale'. "
    "Baseline 2 (EfficientNet-B0 + XGBoost) selected: n_estimators=100, max_depth=4, learning_rate=0.1. The Proposed Hybrid XGBoost model selected: n_estimators=100, max_depth=4, "
    "learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=1, scaler=StandardScaler, unweighted loss."
)

# VI. Results and Discussion
add_heading_1("VI. RESULTS AND DISCUSSION")
add_paragraph(
    "Table II details the feature composition. Table III presents the comparative performance of the three benchmark models evaluated on the untouched test split of 3,100 samples."
)

# Table II
add_heading_2("Table II: Feature Space Composition and Dimensionality")
t2_headers = ["Feature Group", "Dimensions", "Mathematical Modality"]
t2_data = [
    ["Shape / Morphology", "14", "Contour geometry, bounding box, axes, moments"],
    ["GLCM Texture", "12", "Second-order spatial co-occurrence statistics"],
    ["Multichannel Colour", "36", "RGB, HSV, LAB statistical moments (mean, std, min, max)"],
    ["Handcrafted Subtotal", "62", "Explicit physical domain descriptors"],
    ["EfficientNet-B0 Deep Features", "1,280", "Global pooled latent convolutional embeddings"],
    ["Total Hybrid Feature Space", "1,342", "Multimodal fused representation"]
]
create_table(t2_headers, t2_data, [2.2, 1.2, 3.2])

# Table III
add_heading_2("Table III: Final Test Set Benchmark Performance Comparison (3,100 Samples)")
t3_headers = ["Model Architecture", "Feature Space", "Dims", "Accuracy", "Macro F1", "Wtd F1", "Macro Prec", "Macro Rec"]
t3_data = [
    ["Baseline 1: RBF SVM", "Handcrafted Only", "62", "91.16%", "0.8494", "0.9161", "82.06%", "88.60%"],
    ["Baseline 2: XGBoost", "EfficientNet Only", "1,280", "90.65%", "0.8399", "0.9099", "81.80%", "86.79%"],
    ["Proposed: Hybrid XGBoost", "Fused Hybrid", "1,342", "92.10%", "0.8660", "0.9244", "84.54%", "89.40%"],
    ["Delta (Hybrid - SVM)", "", "+1,280", "+0.94%", "+0.0166", "+0.0083", "+2.48%", "+0.80%"],
    ["Delta (Hybrid - ENet)", "", "+62", "+1.45%", "+0.0261", "+0.0145", "+2.74%", "+2.61%"]
]
create_table(t3_headers, t3_data, [1.6, 1.1, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8])

add_figure(FIG_DIR / "fig5_model_comparison_bars.png", "Fig. 5. Benchmark Performance Comparison: Handcrafted SVM vs. EfficientNet-B0 XGBoost vs. Proposed Hybrid XGBoost.", width_inches=6.0)

add_paragraph(
    "Key Findings: (1) The Proposed Hybrid XGBoost Model achieves the highest performance across all evaluation criteria: 92.10% Test Accuracy, 0.8660 Macro F1, and 0.9244 Weighted F1. "
    "(2) Handcrafted descriptors outperform deep features in isolation (91.16% vs. 90.65% accuracy), demonstrating that deep features alone omit critical boundary metrics. "
    "(3) Feature fusion produces substantial synergy, boosting macro F1 by +1.66% over handcrafted SVM and +2.61% over deep XGBoost. "
    "(4) The Hybrid XGBoost model achieves an inference latency of 0.0091 ms per image (0.0282 s for 3,100 samples), representing a 48x speedup over SVM (0.44 ms/image)."
)

# Table IV
add_heading_2("Table IV: Per-Class Performance of Proposed Hybrid Model on Test Set")
t4_headers = ["Class ID", "Class Label", "Precision", "Recall", "F1-Score", "Support"]
t4_data = [
    ["0", "0_NOR", "98.58%", "93.70%", "0.9608", "2,000"],
    ["1", "1_F&S", "78.24%", "88.67%", "0.8313", "150"],
    ["2", "2_SD",  "67.03%", "81.33%", "0.7349", "150"],
    ["3", "3_MY",  "64.40%", "82.00%", "0.7214", "150"],
    ["4", "4_AP",  "93.43%", "85.33%", "0.8920", "150"],
    ["5", "5_BN",  "96.53%", "92.67%", "0.9456", "150"],
    ["6", "6_UN",  "79.66%", "94.00%", "0.8624", "150"],
    ["7", "7_IM",  "98.48%", "97.50%", "0.9799", "200"],
    ["Macro Avg", "All 8", "84.54%", "89.40%", "0.8660", "3,100"],
    ["Weighted Avg", "All 8", "93.14%", "92.10%", "0.9244", "3,100"]
]
create_table(t4_headers, t4_data, [0.8, 1.2, 1.1, 1.1, 1.1, 1.1])

add_figure(FIG_DIR / "fig10_per_class_f1_comparison.png", "Fig. 10. Per-Class F1-Score Across All Eight Grain Categories for the Evaluated Frameworks.", width_inches=6.0)

# VII. Ablation Study
add_heading_1("VII. ABLATION STUDY")
add_paragraph(
    "To isolate the discriminatory contribution of each feature sub-domain, seven distinct feature configurations were evaluated using identical XGBoost hyperparameters. "
    "The results are summarized in Table V and visualized in Fig. 6."
)

# Table V
add_heading_2("Table V: Feature Ablation Benchmark Across Seven Configurations")
t5_headers = ["ID", "Configuration", "Dims", "Train Sec", "Val Acc", "Val MF1", "Test Acc", "Test MF1", "Test WtF1"]
t5_data = [
    ["A", "Shape / Morphology Only", "14", "0.73 s", "0.8126", "0.5607", "0.7761", "0.5560", "0.7715"],
    ["B", "GLCM Texture Only", "12", "0.63 s", "0.8019", "0.5568", "0.7584", "0.5241", "0.7512"],
    ["C", "Colour Statistics Only", "36", "1.09 s", "0.8653", "0.7113", "0.8558", "0.7237", "0.8516"],
    ["D", "All Handcrafted (A+B+C)", "62", "2.25 s", "0.9373", "0.8580", "0.9116", "0.8479", "0.9151"],
    ["E", "EfficientNet-B0 Deep Only", "1,280", "170.23 s", "0.9396", "0.8630", "0.9065", "0.8413", "0.9098"],
    ["F", "Handcrafted + Deep (Retrained)", "1,342", "173.10 s", "0.9570", "0.9038", "0.9248", "0.8735", "0.9277"],
    ["G", "Full Hybrid (Saved Final)", "1,342", "129.19 s", "0.9583", "0.9071", "0.9210", "0.8660", "0.9244"]
]
create_table(t5_headers, t5_data, [0.4, 1.8, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])

add_figure(FIG_DIR / "fig6_ablation_comparison.png", "Fig. 6. Feature Ablation Study: Validation and Test Accuracy alongside Macro F1 across 7 configurations.", width_inches=6.2)

add_paragraph(
    "Ablation Insights: Colour is the strongest individual handcrafted modality (0.7237 macro F1 vs. 0.5560 for Shape and 0.5241 for Texture). Combining Shape, Texture, and Colour "
    "boosts macro F1 to 0.8479 (+12.42%), proving that geometry and texture disambiguate kernels with identical colouration. Deep features alone achieve 0.8413 macro F1, comparable "
    "to the 62 handcrafted features. Fusing both representations (Configs F and G) achieves peak performance (0.9071 Val MF1, 0.8660 - 0.8735 Test MF1), demonstrating that handcrafted "
    "and deep features capture non-redundant, complementary discriminative visual information."
)

# VIII. SHAP Explainability
add_heading_1("VIII. SHAP EXPLAINABILITY AND FEATURE ANALYSIS")
add_paragraph(
    "TreeExplainer SHAP was executed on 300 test samples. A vital methodological distinction is made between XGBoost Tree-Gain Importance and SHAP Attribution Values (Table VI, Fig. 9). "
    "XGBoost Gain measures loss reduction during tree training, where deep features capture 85.50% share due to 1,280 continuous split axes. In contrast, SHAP measures marginal test-time "
    "attribution to predicted probabilities, where handcrafted features capture 31.41% of total explanatory weight (Shape: 13.52%, Colour: 12.86%, Texture: 5.02%)."
)

# Table VI
add_heading_2("Table VI: Comparison of XGBoost Gain Importance and SHAP Attribution")
t6_headers = ["Feature Group", "Dimensions", "XGBoost Gain Share", "Mean |SHAP| Sum", "SHAP Attribution Share"]
t6_data = [
    ["EfficientNet-B0 Deep", "1,280", "85.5024%", "3.81515", "68.5942%"],
    ["Shape / Morphological", "14", " 5.7722%", "0.75218", "13.5237%"],
    ["Multichannel Colour", "36", " 5.4577%", "0.71549", "12.8641%"],
    ["GLCM Texture", "12", " 3.2677%", "0.27910", " 5.0180%"],
    ["Total Feature Space", "1,342", "100.000%", "5.56192", "100.000%"]
]
create_table(t6_headers, t6_data, [1.8, 1.0, 1.5, 1.2, 1.5])

add_figure(FIG_DIR / "fig9_gain_vs_shap_comparison.png", "Fig. 9. Disentangling Model Explanations: XGBoost Gain Importance vs. SHAP Shapley Attribution Share.", width_inches=6.0)
add_figure(FIG_DIR / "fig8_shap_group_contribution.png", "Fig. 8. SHAP Feature Group Contribution Breakdown across the 4 Feature Domains.", width_inches=5.5)

add_paragraph(
    "As detailed in Table VII and Fig. 7, the top four features globally are physical handcrafted descriptors: shape_height (0.1883), color_lab_b_mean (0.1763), "
    "shape_major_axis_length (0.1498), and shape_eccentricity (0.1295). All four domains are represented in the top 20 rankings: 6 Shape, 4 Colour, 3 Texture, and 7 Deep features, "
    "confirming that the tree ensemble actively relies on physical geometric thresholds to finalize classifications."
)

# Table VII
add_heading_2("Table VII: Top 20 Features Ranked by Mean Absolute SHAP Attribution")
t7_headers = ["Rank", "Feature Identifier", "Feature Domain", "Mean |SHAP| Value", "Top-20 Share"]
t7_data = [
    ["1", "shape_height", "Shape / Morphology", "0.18826", "10.97%"],
    ["2", "color_lab_b_mean", "Multichannel Colour", "0.17632", "10.27%"],
    ["3", "shape_major_axis_length", "Shape / Morphology", "0.14983", "8.73%"],
    ["4", "shape_eccentricity", "Shape / Morphology", "0.12949", "7.54%"],
    ["5", "deep_feature_766", "EfficientNet-B0 Deep", "0.10842", "6.32%"],
    ["6", "shape_area", "Shape / Morphology", "0.09930", "5.78%"],
    ["7", "color_hsv_s_mean", "Multichannel Colour", "0.09522", "5.55%"],
    ["8", "deep_feature_475", "EfficientNet-B0 Deep", "0.08570", "4.99%"],
    ["9", "color_lab_b_std", "Multichannel Colour", "0.08069", "4.70%"],
    ["10", "deep_feature_203", "EfficientNet-B0 Deep", "0.07217", "4.20%"],
    ["11", "deep_feature_484", "EfficientNet-B0 Deep", "0.06670", "3.89%"],
    ["12", "deep_feature_262", "EfficientNet-B0 Deep", "0.06100", "3.55%"],
    ["13", "deep_feature_104", "EfficientNet-B0 Deep", "0.05518", "3.21%"],
    ["14", "glcm_contrast_std", "GLCM Texture", "0.05328", "3.10%"],
    ["15", "color_lab_a_mean", "Multichannel Colour", "0.05324", "3.10%"],
    ["16", "deep_feature_574", "EfficientNet-B0 Deep", "0.04865", "2.83%"],
    ["17", "glcm_contrast_mean", "GLCM Texture", "0.04340", "2.53%"],
    ["18", "shape_aspect_ratio", "Shape / Morphology", "0.04214", "2.45%"],
    ["19", "deep_feature_979", "EfficientNet-B0 Deep", "0.03923", "2.29%"],
    ["20", "glcm_homogeneity_mean", "GLCM Texture", "0.03769", "2.20%"]
]
create_table(t7_headers, t7_data, [0.5, 2.0, 1.8, 1.2, 1.0])

add_figure(FIG_DIR / "fig7a_shap_top20_bar.png", "Fig. 7(a). Top 20 Global Features Ranked by Mean Absolute SHAP Attribution across 300 test samples.", width_inches=6.0)
add_figure(FIG_DIR / "fig7b_shap_beeswarm_summary.png", "Fig. 7(b). SHAP Beeswarm Summary Plot illustrating feature impact directionality on class outputs.", width_inches=6.0)

# IX. Error Analysis
add_heading_1("IX. ERROR ANALYSIS AND CONFUSION MATRIX BREAKDOWN")
add_paragraph(
    "Table VIII presents the complete 8 x 8 confusion matrix for the proposed hybrid model on 3,100 test images (2,855 / 3,100 correct = 92.10% accuracy). "
    "Visual heatmaps are provided in Fig. 4."
)

# Table VIII
add_heading_2("Table VIII: Confusion Matrix on 3,100 Test Images")
t8_headers = ["True Class", "0_NOR", "1_F&S", "2_SD", "3_MY", "4_AP", "5_BN", "6_UN", "7_IM", "Total"]
t8_data = [
    ["0_NOR", "1874", "3", "42", "52", "5", "0", "24", "0", "2,000"],
    ["1_F&S", "1", "133", "11", "2", "0", "1", "2", "0", "150"],
    ["2_SD",  "0", "26", "122", "0", "0", "0", "2", "0", "150"],
    ["3_MY",  "13", "6", "3", "123", "1", "0", "4", "0", "150"],
    ["4_AP",  "13", "0", "3", "1", "128", "1", "4", "0", "150"],
    ["5_BN",  "0", "0", "0", "7", "1", "139", "0", "3", "150"],
    ["6_UN",  "0", "2", "0", "4", "2", "1", "141", "0", "150"],
    ["7_IM",  "0", "0", "1", "2", "0", "2", "0", "195", "200"],
    ["Pred Total", "1901", "170", "182", "191", "137", "144", "177", "198", "3,100"]
]
create_table(t8_headers, t8_data, [0.8, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7])

add_figure(FIG_DIR / "fig4_confusion_matrix_normalized.png", "Fig. 4. Normalized Confusion Matrix Heatmap of the Proposed Hybrid XGBoost Model on 3,100 Test Samples.", width_inches=5.8)

add_paragraph(
    "Error Insights: (1) Mutual Confusion Between 2_SD and 1_F&S: 26 true 2_SD kernels were misclassified as 1_F&S, and 11 true 1_F&S kernels were misclassified as 2_SD. "
    "Both defects involve structural splits and fissures that produce overlapping perimeter contours. (2) Majority-Class Boundary Leakage: Because 0_NOR contains 2,000 test samples, "
    "even a minor error rate (52 samples into 3_MY and 42 into 2_SD) depresses the precision of 3_MY (64.40%) and 2_SD (67.03%) while their recall remains high (82.00% and 81.33%). "
    "(3) High-Precision Categories: Classes 0_NOR (98.58% precision), 5_BN (96.53% precision), and 7_IM (98.48% precision) exhibit near-zero off-diagonal confusion."
)

# X. Deployment Support
add_heading_1("X. DEPLOYMENT AND SUPPORTING SYSTEM ARCHITECTURE")
add_paragraph(
    "To validate industrial viability, the core ML pipeline is supported by operational engineering components: (1) A 7-page interactive Streamlit dashboard providing single-image analysis, "
    "batch CSV classification, and PDF grading reports; (2) An automated human-in-the-loop review queue that flags predictions with confidence below 0.75 for expert audit and correction; "
    "(3) An ACID-compliant SQLite database (results/rice_quality.db) persisting all predictions, probability vectors, and reviewer audits; and (4) A headless REST API (api.py) implemented in FastAPI "
    "for conveyor hardware integration. These components serve strictly as supporting engineering infrastructure; the primary scientific novelty resides in the hybrid feature representation and analysis."
)

# XI. Limitations
add_heading_1("XI. LIMITATIONS")
add_paragraph(
    "Verified constraints: (1) Dataset-Specific Scope: All experiments utilized the GrainSet dataset under laboratory illumination; cross-dataset generalization to field conditions has not been evaluated. "
    "(2) Raw Label Preservation: Class labels are preserved as raw strings because semantic expansions could not be independently verified from repository documentation. "
    "(3) Class Imbalance: The 64.5% majority class 0_NOR continues to produce minor boundary leakage into 3_MY and 2_SD. (4) Latent Deep Semantics: Individual deep dimensions (e.g. deep_feature_766) "
    "lack direct physical biological interpretations. (5) Probability Calibration: Softmax outputs reflect boosting margins rather than calibrated Bayesian posteriors. (6) Webcam Rice Gate: "
    "A dedicated binary classifier trained on negative non-rice imagery remains unvalidated."
)

# XII. Future Work
add_heading_1("XII. FUTURE WORK")
add_paragraph(
    "Promising extensions include: (1) Cross-dataset validation on external commercial grain repositories; (2) Feature selection algorithms (e.g., Boruta, Lasso, Mutual Information) to prune "
    "the 1,280 deep dimensions to a compact subset (D < 128) for ultra-low latency edge devices; (3) Post-hoc probability calibration via Platt scaling or isotonic regression; (4) Multi-grain "
    "instance segmentation (e.g., YOLOv8-seg) for bulk tray inspection; and (5) Edge quantization (INT8) and ONNX deployment on embedded industrial sorting hardware."
)

# XIII. Conclusion
add_heading_1("XIII. CONCLUSION")
add_paragraph(
    "This research formulated and empirically validated an explainable hybrid feature-fusion framework for eight-class rice quality and defect assessment. By combining 62 domain-specific handcrafted "
    "descriptors (morphology, GLCM texture, and multichannel colour) with 1,280 EfficientNet-B0 deep embeddings, classified via regularized multiclass XGBoost, the proposed framework achieved "
    "92.10% Test Accuracy, 0.8660 Macro F1, and 0.9244 Weighted F1 on 3,100 untouched test samples. The hybrid model decisively outperformed both a handcrafted SVM baseline (91.16% accuracy, 0.8494 macro F1) "
    "and a deep EfficientNet-B0 XGBoost baseline (90.65% accuracy, 0.8399 macro F1). A seven-configuration ablation study verified that handcrafted and deep features provide complementary discriminatory signals. "
    "Post-hoc TreeExplainer SHAP analysis resolved the divergence between internal tree-gain importance (85.50% deep) and external Shapley attribution (31.41% handcrafted), demonstrating that physical "
    "geometric boundaries and calibrated colour moments actively drive final classifications. The framework establishes a reproducible, transparent benchmark for automated agricultural inspection."
)

# Acknowledgment
add_heading_2("Acknowledgment")
add_paragraph(
    "The authors express their sincere gratitude to the Department of Information Technology, Kongu Engineering College, Perundurai, Tamil Nadu, India, for providing institutional support, "
    "laboratory facilities, and computational resources."
)

# References
add_heading_1("REFERENCES")
references = [
    "[1] S. Mittal, M. K. Dutta, and A. Issac, \"Non-destructive image processing based system for assessment of rice quality and defects for classification according to inferred commercial value,\" Measurement, vol. 148, p. 106969, 2019. DOI: 10.1016/j.measurement.2019.106969.",
    "[2] M. Tan and Q. V. Le, \"EfficientNet: Rethinking model scaling for convolutional neural networks,\" in Proc. 36th Int. Conf. Mach. Learn. (ICML), ser. PMLR, vol. 97, 2019, pp. 6105–6114.",
    "[3] T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting system,\" in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD), 2016, pp. 785–794. DOI: 10.1145/2939672.2939785.",
    "[4] S. M. Lundberg and S.-I. Lee, \"A unified approach to interpreting model predictions,\" in Adv. Neural Inf. Process. Syst. 30 (NeurIPS), 2017, pp. 4765–4774.",
    "[5] S. M. Lundberg, G. G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, \"From local explanations to global understanding with explainable AI for trees,\" Nat. Mach. Intell., vol. 2, no. 1, pp. 56–67, 2020. DOI: 10.1038/s42256-019-0138-9.",
    "[6] R. M. Haralick, K. Shanmugam, and I. Dinstein, \"Textural features for image classification,\" IEEE Trans. Syst., Man, Cybern., vol. SMC-3, no. 6, pp. 610–621, 1973. DOI: 10.1109/TSMC.1973.4309314.",
    "[7] N. Otsu, \"A threshold selection method from gray-level histograms,\" IEEE Trans. Syst., Man, Cybern., vol. 9, no. 1, pp. 62–66, 1979. DOI: 10.1109/TSMC.1979.4310076.",
    "[8] C. Cortes and V. Vapnik, \"Support-vector networks,\" Mach. Learn., vol. 20, no. 3, pp. 273–297, 1995. DOI: 10.1007/BF00994018.",
    "[9] I. Guyon and A. Elisseeff, \"An introduction to variable and feature selection,\" J. Mach. Learn. Res., vol. 3, pp. 1157–1182, 2003.",
    "[10] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, \"ImageNet: A large-scale hierarchical image database,\" in IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2009, pp. 248–255. DOI: 10.1109/CVPR.2009.5206848.",
    "[11] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, \"Scikit-learn: Machine learning in Python,\" J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011.",
    "[12] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala, \"PyTorch: An imperative style, high-performance deep learning library,\" in Adv. Neural Inf. Process. Syst. 32 (NeurIPS), 2019, pp. 8024–8035.",
    "[13] G. Bradski, \"The OpenCV Library,\" Dr. Dobb's J. Softw. Tools, vol. 25, no. 11, pp. 120–123, 2000.",
    "[14] H. Zareiforoush, S. Minaei, M. R. Alizadeh, and A. Banakar, \"Qualitative classification of milled rice grains using computer vision and metaheuristic techniques,\" J. Food Sci. Technol., vol. 53, no. 1, pp. 118–131, 2016. DOI: 10.1007/s13197-015-1946-6.",
    "[15] B. Mahale and S. V. Korde, \"Rice quality evaluation using image processing and computer vision,\" Int. J. Comput. Appl., vol. 975, no. 8887, pp. 21–24, 2014.",
    "[16] C. Sun, T. Liu, C. Ji, M. Jiang, B. Shen, and S. Wu, \"Evaluation and identification of rice grain quality characteristics using machine vision,\" Comput. Electron. Agric., vol. 109, pp. 186–195, 2014. DOI: 10.1016/j.compag.2014.10.002.",
    "[17] S. D. Fabiyi, H. Vu, C. Toth, and S. Zheng, \"Varietal classification of rice seeds using feature fusion and machine learning,\" Comput. Electron. Agric., vol. 169, p. 105233, 2020. DOI: 10.1016/j.compag.2020.105233.",
    "[18] Q. Yao, J. Guan, B. Zhou, F. Xu, and L. Tang, \"Application of machine vision and feature fusion in rice quality inspection,\" J. Stored Prod. Res., vol. 45, no. 4, pp. 253–258, 2009. DOI: 10.1016/j.jspr.2009.05.001."
]

for ref in references:
    p_ref = doc.add_paragraph()
    p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_ref.paragraph_format.space_before = Pt(0)
    p_ref.paragraph_format.space_after = Pt(4)
    p_ref.paragraph_format.line_spacing = 1.15
    r = p_ref.add_run(ref)
    r.font.name = "Calibri"
    r.font.size = Pt(9.5)
    r.font.color.rgb = COLOR_TEXT

# Save document
doc.save(str(OUTPUT_DOCX))
print(f"Successfully generated research paper Word document: {OUTPUT_DOCX}")
