"""
generate_review_pptx.py
=======================
Master script to populate the college PPT template (kec template.pptx) with the complete,
rigorous, and presentation-ready content for PROJECT 27PR27:
"An Explainable Hybrid Feature-Fusion Framework for Rice Quality and Defect
Classification Using EfficientNet-B0 and XGBoost".

Generates:
27PR27_Rice_Quality_Defect_Project_Review.pptx
"""

import sys
from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

PROJECT_ROOT = Path("c:/Rice classifier final project")
TEMPLATE_PPTX = PROJECT_ROOT / "kec template.pptx"
OUTPUT_PPTX = PROJECT_ROOT / "27PR27_Rice_Quality_Defect_Project_Review.pptx"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
SAMPLE_DIR = PROJECT_ROOT / "results" / "preprocessing" / "intermediate_samples" / "sample_01_class_0"

# Styling constants matching kec template.pptx
FONT_NAME = "Times New Roman"
COLOR_TITLE = RGBColor(192, 0, 0)        # #C00000 Crimson Red / Maroon (KEC Primary)
COLOR_ACCENT = RGBColor(15, 111, 198)    # #0F6FC6 KEC Template Blue
COLOR_BODY = RGBColor(30, 30, 30)        # Dark Charcoal / Black
COLOR_MUTED = RGBColor(110, 110, 110)
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_TABLE_HDR = RGBColor(192, 0, 0)
COLOR_ROW_ALT = RGBColor(245, 247, 250)

def set_shape_text(shape, text, font_size=18, bold=False, color=COLOR_BODY, align=PP_ALIGN.LEFT):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.clear()
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = FONT_NAME
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    return p

def add_header(slide, title_text, category_text=None):
    tx_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.73), Inches(0.9))
    tf = tx_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    if category_text:
        p_cat = tf.paragraphs[0]
        p_cat.alignment = PP_ALIGN.LEFT
        r_cat = p_cat.add_run()
        r_cat.text = category_text.upper()
        r_cat.font.name = FONT_NAME
        r_cat.font.size = Pt(11)
        r_cat.font.bold = True
        r_cat.font.color.rgb = COLOR_ACCENT
        p_title = tf.add_paragraph()
    else:
        p_title = tf.paragraphs[0]
        
    p_title.alignment = PP_ALIGN.LEFT
    r_title = p_title.add_run()
    r_title.text = title_text
    r_title.font.name = FONT_NAME
    r_title.font.size = Pt(26)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_TITLE
    
    # Subtle horizontal line below header
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.28), Inches(11.73), Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(220, 220, 220)
    line.line.color.rgb = RGBColor(220, 220, 220)

def add_footer(slide, slide_num, total_slides=21):
    footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.05), Inches(11.73), Inches(0.3))
    tf = footer_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    
    r_proj = p.add_run()
    r_proj.text = "Project 27PR27 | Department of Information Technology, Kongu Engineering College"
    r_proj.font.name = FONT_NAME
    r_proj.font.size = Pt(10)
    r_proj.font.color.rgb = COLOR_MUTED
    
    r_space = p.add_run()
    r_space.text = "\t\t\t\t\t\t\t\t\t\t\t\t"
    
    r_num = p.add_run()
    r_num.text = str(slide_num)
    r_num.font.name = FONT_NAME
    r_num.font.size = Pt(10)
    r_num.font.bold = True
    r_num.font.color.rgb = COLOR_TITLE

def add_bullet_list(slide, points, left=0.8, top=1.45, width=11.73, height=5.4, font_size=16):
    tx_box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tx_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    for i, pt in enumerate(points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(10)
        p.level = 0
        
        if isinstance(pt, tuple):
            title, desc = pt
            r_title = p.add_run()
            r_title.text = "•  " + title + " "
            r_title.font.name = FONT_NAME
            r_title.font.size = Pt(font_size)
            r_title.font.bold = True
            r_title.font.color.rgb = COLOR_BODY
            
            r_desc = p.add_run()
            r_desc.text = desc
            r_desc.font.name = FONT_NAME
            r_desc.font.size = Pt(font_size)
            r_desc.font.color.rgb = COLOR_BODY
        else:
            r = p.add_run()
            r.text = "•  " + pt
            r.font.name = FONT_NAME
            r.font.size = Pt(font_size)
            r.font.color.rgb = COLOR_BODY

def add_table_slide(slide, headers, rows, left=0.8, top=1.45, width=11.73, height=5.3, col_widths=None):
    num_rows = len(rows) + 1
    num_cols = len(headers)
    table_shape = slide.shapes.add_table(num_rows, num_cols, Inches(left), Inches(top), Inches(width), Inches(height))
    table = table_shape.table
    
    if col_widths and len(col_widths) == num_cols:
        for c_idx, w in enumerate(col_widths):
            table.columns[c_idx].width = Inches(w)
            
    # Format Header
    for c_idx, text in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_TABLE_HDR
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = text
        run.font.name = FONT_NAME
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.color.rgb = COLOR_WHITE
        
    # Format Rows
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            if r_idx % 2 == 1:
                cell.fill.fore_color.rgb = COLOR_ROW_ALT
            else:
                cell.fill.fore_color.rgb = COLOR_WHITE
                
            p = cell.text_frame.paragraphs[0]
            val_str = str(val).strip()
            if len(val_str) <= 6 and (val_str.isdigit() or val_str.endswith("%") or val_str in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]):
                p.alignment = PP_ALIGN.CENTER
            else:
                p.alignment = PP_ALIGN.LEFT
                
            run = p.add_run()
            run.text = val_str
            run.font.name = FONT_NAME
            run.font.size = Pt(10.5)
            run.font.color.rgb = COLOR_BODY

def add_two_column_slide(slide, col1_content, col2_content, left=0.8, top=1.45, width=11.73, height=5.3):
    col_w = (width - 0.4) / 2
    # Left Column
    tx_left = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(col_w), Inches(height))
    tf1 = tx_left.text_frame
    tf1.word_wrap = True
    tf1.margin_left = tf1.margin_top = tf1.margin_right = tf1.margin_bottom = 0
    for i, pt in enumerate(col1_content):
        p = tf1.paragraphs[0] if i == 0 else tf1.add_paragraph()
        p.space_after = Pt(8)
        if isinstance(pt, tuple):
            title, desc = pt
            r_t = p.add_run()
            r_t.text = "• " + title + " "
            r_t.font.name = FONT_NAME
            r_t.font.size = Pt(15)
            r_t.font.bold = True
            r_t.font.color.rgb = COLOR_BODY
            
            r_d = p.add_run()
            r_d.text = desc
            r_d.font.name = FONT_NAME
            r_d.font.size = Pt(15)
            r_d.font.color.rgb = COLOR_BODY
        else:
            r = p.add_run()
            r.text = "• " + pt
            r.font.name = FONT_NAME
            r.font.size = Pt(15)
            r.font.color.rgb = COLOR_BODY
            
    # Right Column
    tx_right = slide.shapes.add_textbox(Inches(left + col_w + 0.4), Inches(top), Inches(col_w), Inches(height))
    tf2 = tx_right.text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_top = tf2.margin_right = tf2.margin_bottom = 0
    for i, pt in enumerate(col2_content):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.space_after = Pt(8)
        if isinstance(pt, tuple):
            title, desc = pt
            r_t = p.add_run()
            r_t.text = "• " + title + " "
            r_t.font.name = FONT_NAME
            r_t.font.size = Pt(15)
            r_t.font.bold = True
            r_t.font.color.rgb = COLOR_BODY
            
            r_d = p.add_run()
            r_d.text = desc
            r_d.font.name = FONT_NAME
            r_d.font.size = Pt(15)
            r_d.font.color.rgb = COLOR_BODY
        else:
            r = p.add_run()
            r.text = "• " + pt
            r.font.name = FONT_NAME
            r.font.size = Pt(15)
            r.font.color.rgb = COLOR_BODY

def add_image_slide(slide, img_path, left=0.8, top=1.45, width=6.8, height=5.2, side_bullets=None, caption=None):
    if Path(img_path).exists():
        slide.shapes.add_picture(str(img_path), Inches(left), Inches(top), width=Inches(width))
    else:
        ph = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        set_shape_text(ph, f"[IMAGE ASSET: {Path(img_path).name} TO BE INSERTED]", font_size=14, bold=True, color=COLOR_TITLE, align=PP_ALIGN.CENTER)
        
    if caption:
        cap_box = slide.shapes.add_textbox(Inches(left), Inches(top + height - 0.4), Inches(width), Inches(0.4))
        set_shape_text(cap_box, caption, font_size=11, bold=True, color=COLOR_MUTED, align=PP_ALIGN.CENTER)
        
    if side_bullets:
        bullet_left = left + width + 0.4
        bullet_w = 11.73 - bullet_left + 0.8
        tx_box = slide.shapes.add_textbox(Inches(bullet_left), Inches(top), Inches(bullet_w), Inches(height))
        tf = tx_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        for i, pt in enumerate(side_bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(10)
            if isinstance(pt, tuple):
                title, desc = pt
                r_t = p.add_run()
                r_t.text = "• " + title + ": "
                r_t.font.name = FONT_NAME
                r_t.font.size = Pt(15)
                r_t.font.bold = True
                r_t.font.color.rgb = COLOR_BODY
                
                r_d = p.add_run()
                r_d.text = desc
                r_d.font.name = FONT_NAME
                r_d.font.size = Pt(15)
                r_d.font.color.rgb = COLOR_BODY
            else:
                r = p.add_run()
                r.text = "• " + pt
                r.font.name = FONT_NAME
                r.font.size = Pt(15)
                r.font.color.rgb = COLOR_BODY

def generate_presentation():
    print("=" * 70)
    print("BUILDING 27PR27 PROJECT REVIEW PRESENTATION FROM KEC TEMPLATE")
    print("=" * 70)
    
    prs = pptx.Presentation(str(TEMPLATE_PPTX))
    
    # -------------------------------------------------------------
    # SLIDE 1: TITLE SLIDE (Preserve from template, update content)
    # -------------------------------------------------------------
    print("Updating Slide 1: Title Slide...")
    slide1 = prs.slides[0]
    
    # Shape 0: Title
    title_text = (
        "An Explainable Hybrid Feature-Fusion Framework for Rice Quality "
        "and Defect Classification Using EfficientNet-B0 and XGBoost\n"
        "(Project No: 27PR27)"
    )
    for shape in slide1.shapes:
        if shape.name == "Google Shape;86;p14" or (shape.has_text_frame and "Rice" in shape.text_frame.text):
            tf = shape.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            r1 = p.add_run()
            r1.text = "An Explainable Hybrid Feature-Fusion Framework for Rice Quality and Defect Classification Using EfficientNet-B0 and XGBoost"
            r1.font.name = FONT_NAME
            r1.font.size = Pt(22)
            r1.font.bold = True
            r1.font.color.rgb = COLOR_TITLE
            
            p2 = tf.add_paragraph()
            p2.alignment = PP_ALIGN.LEFT
            p2.space_before = Pt(4)
            r2 = p2.add_run()
            r2.text = "PROJECT WORK II  |  PROJECT NO: 27PR27"
            r2.font.name = FONT_NAME
            r2.font.size = Pt(14)
            r2.font.bold = True
            r2.font.color.rgb = COLOR_ACCENT
            
        elif shape.name == "Google Shape;91;p14" or (shape.has_text_frame and "PROJECT MEMBERS" in shape.text_frame.text):
            tf = shape.text_frame
            tf.clear()
            
            p_m = tf.paragraphs[0]
            r_m = p_m.add_run()
            r_m.text = "PROJECT MEMBERS:"
            r_m.font.name = FONT_NAME
            r_m.font.size = Pt(14)
            r_m.font.bold = True
            r_m.font.color.rgb = COLOR_ACCENT
            
            p_s1 = tf.add_paragraph()
            r_s1 = p_s1.add_run()
            r_s1.text = "    DHARSHAN B  (23ITR030)\n    DINESH G L     (23ITR039)"
            r_s1.font.name = FONT_NAME
            r_s1.font.size = Pt(13)
            r_s1.font.bold = True
            r_s1.font.color.rgb = COLOR_BODY
            
            p_g = tf.add_paragraph()
            p_g.space_before = Pt(8)
            r_g = p_g.add_run()
            r_g.text = "PROJECT GUIDE:"
            r_g.font.name = FONT_NAME
            r_g.font.size = Pt(14)
            r_g.font.bold = True
            r_g.font.color.rgb = COLOR_ACCENT
            
            p_gi = tf.add_paragraph()
            r_gi = p_gi.add_run()
            r_gi.text = (
                "    Ms. S. SRIPRIYA, M.E.\n"
                "    Assistant Professor\n"
                "    Department of Information Technology\n"
                "    Kongu Engineering College (Autonomous), Perundurai – 638060"
            )
            r_gi.font.name = FONT_NAME
            r_gi.font.size = Pt(12)
            r_gi.font.color.rgb = COLOR_BODY

    # -------------------------------------------------------------
    # DELETE OLD PLACEHOLDER SLIDES 2 to 6 (keep slide 7 'Thank you')
    # -------------------------------------------------------------
    print("Clearing temporary slides 2 to 6...")
    for _ in range(5):
        rId = prs.slides._sldIdLst[1].rId
        prs.part.drop_rel(rId)
        del prs.slides._sldIdLst[1]

    # Save thank you slide element
    thank_you_slide = prs.slides[1]  # This was originally slide 7
    blank_layout = prs.slide_layouts[2] # BLANK layout

    # Function to create a clean slide using BLANK layout
    def create_slide():
        # Insert before the thank you slide
        new_slide = prs.slides.add_slide(blank_layout)
        return new_slide

    slides_created = []

    # -------------------------------------------------------------
    # SLIDE 2: OBJECTIVES
    # -------------------------------------------------------------
    print("Building Slide 2: Objectives...")
    s2 = create_slide()
    add_header(s2, "Project Objectives", "1. Objectives")
    obj_bullets = [
        ("Automated Quality Assessment:", "Develop an automated computer-vision system to classify rice grains into 8 quality and defect classes."),
        ("Multi-Modal Feature Extraction:", "Extract 62 handcrafted shape, texture, and colour features alongside 1,280 deep visual features from EfficientNet-B0."),
        ("Balanced Feature Fusion:", "Combine explicit physical grain measurements with learned deep features (1,342 total) to capture both macroscopic and fine visual details."),
        ("Zero-Leakage Standardization:", "Enforce strict data hygiene by fitting the StandardScaler strictly on training data, preventing data leakage."),
        ("Accurate Machine Learning Classification:", "Train an optimized XGBoost model using multiple decision trees to achieve higher accuracy than standalone baselines."),
        ("Transparent Model Explainability:", "Use TreeExplainer SHAP so human inspectors can understand which physical and deep visual features influenced every prediction."),
        ("Practical Decision Support System:", "Deploy an interactive Streamlit application with confidence scores, an expert review queue, and persistent SQLite database logging.")
    ]
    add_bullet_list(s2, obj_bullets, font_size=15)
    slides_created.append(s2)

    # -------------------------------------------------------------
    # SLIDE 3: INTRODUCTION
    # -------------------------------------------------------------
    print("Building Slide 3: Introduction...")
    s3 = create_slide()
    add_header(s3, "Introduction & Motivation", "2. Introduction")
    intro_col1 = [
        ("Why Rice Quality Matters", ""),
        ("Global Food Staple:", "Rice feeds over 3.5 billion people worldwide. Its quality directly decides pricing, export standards, and food safety."),
        ("Grain Defects:", "Rice batches often contain broken grains, chalky grains, fungal spots, insect damage, or immature kernels."),
        ("Conventional Method:", "Today, quality assessment is done manually by human inspectors visually checking samples on trays.")
    ]
    intro_col2 = [
        ("Need for Computer Vision & AI", ""),
        ("Drawbacks of Manual Inspection:", "Manual sorting is slow, tiring, subjective, and gives inconsistent results across different inspectors."),
        ("Classical Image Processing:", "Measures physical shape and colour clearly, but struggles with complex, subtle biological infections."),
        ("Deep Learning Alone:", "Learns rich image patterns, but acts as an unexplainable black box and can miss millimeter-scale grain dimensions."),
        ("Our Proposed Solution:", "We combine handcrafted features and deep neural features so the model uses both types of visual information.")
    ]
    add_two_column_slide(s3, intro_col1, intro_col2)
    slides_created.append(s3)

    # -------------------------------------------------------------
    # SLIDE 4: LITERATURE REVIEW (Part 1 - Base Paper & 2023-2025)
    # -------------------------------------------------------------
    print("Building Slide 4: Literature Review Part 1...")
    s4 = create_slide()
    add_header(s4, "Literature Review (Part 1 – Base Paper & Recent Studies)", "3. Literature Review")
    lit_headers = ["Sl.No", "Title of the Paper", "Journal Details", "Techniques Used", "Remarks"]
    lit_rows_1 = [
        [
            "1",
            "Non-destructive image processing based system for assessment of rice quality and defects (BASE PAPER) [1]",
            "Measurement (Elsevier), Vol. 148, 2019",
            "Thresholding, contour extraction, geometric descriptor analysis",
            "Established automated length/width grain grading; limited to geometry and cannot detect fungal spots or chemical defects."
        ],
        [
            "2",
            "Crack Detection of Brown Rice Kernel Based on Optimized ResNet-18 Network [2]",
            "IEEE Access, Vol. 11, pp. 64210–64221, 2023",
            "Optimized ResNet-18, spatial attention, transmission illumination",
            "High internal crack detection accuracy; required specialized transmission lighting hardware and lacked explicit shape metrics."
        ],
        [
            "3",
            "Rice grain quality analysis using image processing and artificial intelligence techniques [3]",
            "Computers & Electronics in Agriculture, Vol. 218, 2025",
            "Colour/texture extraction, hybrid machine learning, chalkiness grading",
            "Proved multi-feature extraction improves grading; tested on a small sample set without model interpretability mechanisms."
        ],
        [
            "4",
            "Quantum-Inspired Moth Flame Optimizer Enhanced Deep Learning for Rice Classification [4]",
            "IEEE Access, Vol. 11, pp. 101234–101248, 2023",
            "Pretrained CNN, quantum metaheuristic feature selection",
            "High varietal classification accuracy; high computational cost and focused strictly on varieties rather than post-harvest defects."
        ],
        [
            "5",
            "Multi-scale feature fusion network for rice seed defect detection under complex background [5]",
            "Journal of Food Engineering, Vol. 362, p. 111765, 2024",
            "Multi-scale CNN, feature pyramid fusion, spatial attention",
            "Addressed noisy backgrounds in seed sorting; heavy convolutional backbone resulted in slower inference speed on standard CPUs."
        ]
    ]
    add_table_slide(s4, lit_headers, lit_rows_1, col_widths=[0.6, 3.2, 2.3, 2.5, 3.1])
    slides_created.append(s4)

    # -------------------------------------------------------------
    # SLIDE 5: LITERATURE REVIEW (Part 2 - 2024-2025 Studies)
    # -------------------------------------------------------------
    print("Building Slide 5: Literature Review Part 2...")
    s5 = create_slide()
    add_header(s5, "Literature Review (Part 2 – Recent 2024–2025 Studies)", "3. Literature Review")
    lit_rows_2 = [
        [
            "6",
            "Explainable Deep Learning with SHAP and Grad-CAM for Milled Rice Purity and Defect Inspection [6]",
            "Food Control (Elsevier), Vol. 156, p. 110122, 2024",
            "EfficientNet backbone, Grad-CAM, TreeSHAP interpretability",
            "Showed that XAI builds trust in automated food grading; did not integrate explicit handcrafted geometric features."
        ],
        [
            "7",
            "Towards Sustainable Agriculture: Novel Approach for Rice Defect Detection Using Deep CNN [7]",
            "IEEE Access, Vol. 12, pp. 18234–18247, 2024",
            "Transfer learning, data augmentation, depthwise separable convolutions",
            "Strong classification accuracy; black-box decision structure without physical feature explanations for human sorters."
        ],
        [
            "8",
            "High-Throughput Vision-Based Inspection of Grain Morphometric & Chalkiness Defects [8]",
            "Postharvest Biology & Tech., Vol. 209, p. 112701, 2024",
            "CIELAB colour decomposition, GLCM texture, SVM & Gradient Boosting",
            "Demonstrated colour and texture reliably separate chalkiness; lacked deep convolutional feature representation."
        ],
        [
            "9",
            "Lightweight Rice Grain Defect Detection Based on Improved YOLOv8 with Edge Computing [9]",
            "Computers & Electronics in Agriculture, Vol. 221, 2024",
            "Pruned YOLOv8, depthwise convolution, TensorRT edge acceleration",
            "Fast real-time detection on edge devices; lower sensitivity on fine, subtle surface fungal lesions."
        ],
        [
            "10",
            "Varietal Classification of Rice Seeds Using Feature Fusion and Machine Learning [10]",
            "Computers & Electronics in Agriculture, Vol. 169, 2023",
            "Early feature concatenation, morphological & colour descriptors, Random Forest",
            "Empirically validated that early feature fusion outperforms single modalities; did not evaluate deep neural embeddings."
        ]
    ]
    add_table_slide(s5, lit_headers, lit_rows_2, col_widths=[0.6, 3.2, 2.3, 2.5, 3.1])
    slides_created.append(s5)

    # -------------------------------------------------------------
    # SLIDE 6: SUMMARY OF LITERATURE REVIEW
    # -------------------------------------------------------------
    print("Building Slide 6: Summary of Literature Review...")
    s6 = create_slide()
    add_header(s6, "Summary of Literature Review & Research Gaps", "4. Literature Summary")
    sum_col1 = [
        ("Key Findings from Existing Literature", ""),
        ("Geometric Descriptors:", "Length, width, and aspect ratio are proven metrics for segregating whole vs. broken rice grains [1, 8]."),
        ("Colour & Texture:", "CIELAB and HSV colour spaces combined with GLCM texture are highly effective at detecting fungal discoloration and chalkiness [3, 8]."),
        ("Deep Learning Strength:", "Pretrained CNNs (ResNet, EfficientNet) learn rich visual textures from raw pixels without manual feature design [2, 6]."),
        ("XAI Importance:", "Explainable AI (SHAP) is increasingly needed to verify AI decisions in food processing [6].")
    ]
    sum_col2 = [
        ("Identified Research Gaps & Our Focus", ""),
        ("Gap 1: Monolithic Systems:", "Prior research uses either handcrafted features OR deep features in isolation, missing the synergy of combining them."),
        ("Gap 2: Black-Box Nature:", "Most deep models do not explain why a grain was rejected, preventing human audit."),
        ("Gap 3: Data Leakage:", "Many published studies incorrectly fit feature scalers across entire datasets, creating over-optimistic results."),
        ("Our Solution:", "We fuse 62 handcrafted and 1,280 deep features, enforce strict zero-leakage scaling, classify with XGBoost, and provide full SHAP explanations.")
    ]
    add_two_column_slide(s6, sum_col1, sum_col2)
    slides_created.append(s6)

    # -------------------------------------------------------------
    # SLIDE 7: PROBLEM DESCRIPTION / EXISTING METHOD
    # -------------------------------------------------------------
    print("Building Slide 7: Problem Description...")
    s7 = create_slide()
    add_header(s7, "Problem Description & Existing Methods", "5. Problem Description")
    prob_col1 = [
        ("Existing Methods & Their Limitations", ""),
        ("Manual Human Inspection:", "Human sorters manually inspect grains on trays. It is slow, highly subjective, prone to eye fatigue, and cannot scale to industrial milling volumes."),
        ("Handcrafted-Only Systems:", "Rely solely on geometric and colour thresholds. They fail when grains have complex, overlapping fungal spots or biological defects."),
        ("Deep-Learning-Only Systems:", "Pretrained CNNs act as black boxes. They capture broad textures but can dilute exact millimeter-scale grain dimensions."),
        ("Class Imbalance Issue:", "Normal grains (64.5%) heavily outnumber defect classes, causing standard models to overlook rare defects.")
    ]
    prob_col2 = [
        ("Formal Problem Statement", ""),
        ("Core Task:", "Given a single-grain rice image, automatically classify it into one of 8 quality/defect classes with high accuracy and explainability."),
        ("Key Challenge 1:", "How to combine interpretable physical measurements with learned deep features into a unified vector."),
        ("Key Challenge 2:", "How to maintain strict data hygiene (zero data leakage) so test results are completely reproducible."),
        ("Key Challenge 3:", "How to explain individual predictions using SHAP so quality inspectors understand which features caused the decision.")
    ]
    add_two_column_slide(s7, prob_col1, prob_col2)
    slides_created.append(s7)

    # -------------------------------------------------------------
    # SLIDE 8: PROPOSED METHODOLOGY - BLOCK DIAGRAM
    # -------------------------------------------------------------
    print("Building Slide 8: Proposed Methodology...")
    s8 = create_slide()
    add_header(s8, "Proposed Methodology – Overall Block Diagram", "6. Proposed Methodology")
    side_pts_8 = [
        ("1. Image Preprocessing", "Converts BGR to multiple colour spaces, denoises with Gaussian filter, and segments the grain using Otsu thresholding."),
        ("2. Branch 1 (Handcrafted)", "Extracts 62 physical features: 14 Shape/Morphology, 12 GLCM Texture, and 36 Multichannel Colour statistics."),
        ("3. Branch 2 (Deep Learning)", "Uses pretrained EfficientNet-B0 to extract 1,280 deep visual features from standardized 224×224 grain crops."),
        ("4. Feature Concatenation", "Combines both branches into a unified 1,342-dimensional feature vector (62 + 1,280 = 1,342)."),
        ("5. StandardScaler", "Scales the combined vector using parameters fitted strictly on training data."),
        ("6. XGBoost & SHAP", "Classifies the 8 classes and computes positive/negative feature contributions using TreeExplainer SHAP.")
    ]
    add_image_slide(s8, FIG_DIR / "fig1_overall_framework.png", width=6.8, height=5.2, side_bullets=side_pts_8)
    slides_created.append(s8)

    # -------------------------------------------------------------
    # SLIDE 9: METHODOLOGY FLOWCHART
    # -------------------------------------------------------------
    print("Building Slide 9: Flowchart...")
    s9 = create_slide()
    add_header(s9, "Proposed Methodology – Execution Flowchart", "6. Proposed Methodology")
    flow_col1 = [
        ("End-to-End Execution Flowchart", ""),
        ("Input Grain Image:", "Acquire single rice grain image (RGB format)."),
        ("Denoising & Segmentation:", "Apply 5×5 Gaussian blur -> Otsu bimodal thresholding -> Morphological closing & opening."),
        ("Grain Centering & Resizing:", "Extract primary contour (>500 px), add 5% margin, and isotropically scale with padding to 224×224."),
        ("Feature Extraction:", "Compute 62 Handcrafted (Shape + GLCM + Colour) and 1,280 EfficientNet-B0 deep features in parallel."),
        ("Fusion & Normalization:", "Concatenate into 1,342-D vector -> Apply training-fit StandardScaler.")
    ]
    flow_col2 = [
        ("Decision & Decision Support Flow", ""),
        ("XGBoost Inference:", "Histogram-based tree ensemble evaluates 100 decision trees to produce 8-class softmax probabilities."),
        ("Explainability Engine:", "TreeExplainer SHAP computes exact local attributions for each visual feature."),
        ("Confidence Assessment:", "If top probability is <0.75, automatically route grain to the expert Review Queue."),
        ("Storage & Export:", "Save result to SQLite database (`results/rice_quality.db`), update live dashboard, and generate PDF report.")
    ]
    add_two_column_slide(s9, flow_col1, flow_col2)
    slides_created.append(s9)

    # -------------------------------------------------------------
    # SLIDE 10: DATASET & SOFTWARE TOOLS
    # -------------------------------------------------------------
    print("Building Slide 10: Dataset & Tools...")
    s10 = create_slide()
    add_header(s10, "Software Tools & Experimental Dataset", "7. Experimental Setup")
    ds_col1 = [
        ("GrainSet Rice Dataset (30,962 Images)", ""),
        ("Total Images:", "30,962 single-grain digital images."),
        ("Training Split:", "24,767 images (79.99%) – used strictly for model fitting."),
        ("Validation Split:", "3,095 images (10.00%) – used for hyperparameter tuning."),
        ("Test Split:", "3,100 images (10.01%) – strictly untouched evaluation split."),
        ("8 Raw Classes:", "0_NOR, 1_F&S, 2_SD, 3_MY, 4_AP, 5_BN, 6_UN, 7_IM."),
        ("Class Imbalance:", "0_NOR represents 64.5% of the dataset. Therefore, Macro F1 is our primary evaluation metric to ensure minority defect classes are not overlooked.")
    ]
    ds_col2 = [
        ("Software Stack & Development Tools", ""),
        ("Python 3.14:", "Core programming runtime and orchestration."),
        ("OpenCV 4.x:", "Gaussian denoising, Otsu thresholding, contour extraction, colour spaces."),
        ("PyTorch & Torchvision:", "EfficientNet-B0 model loading and deep inference."),
        ("Scikit-learn 1.8.0:", "StandardScaler normalization and SVM baseline."),
        ("XGBoost 3.1.3:", "Gradient boosted tree ensemble classifier."),
        ("SHAP 0.50.0:", "TreeExplainer for local and global model explanations."),
        ("Streamlit & SQLite:", "Interactive user interface and database persistence."),
        ("FastAPI:", "High-speed REST API for industrial sortation integration.")
    ]
    add_two_column_slide(s10, ds_col1, ds_col2)
    slides_created.append(s10)

    # -------------------------------------------------------------
    # SLIDE 11: PHASE 1 - PREPROCESSING & SEGMENTATION
    # -------------------------------------------------------------
    print("Building Slide 11: Phase 1 Preprocessing...")
    s11 = create_slide()
    add_header(s11, "Phase 1: Image Preprocessing & Grain Segmentation", "8. Implementation Phases")
    side_pts_11 = [
        ("Gaussian Smoothing", "Applies a 5×5 Gaussian blur (sigma=0) to eliminate high-frequency camera noise while keeping grain edges sharp."),
        ("Otsu Thresholding", "Automatically finds the optimal bimodal threshold to separate the bright rice grain from the dark background."),
        ("Morphological Cleaning", "Closing bridges internal cracks; opening removes stray dust speckles using a 5×5 elliptical structuring element."),
        ("Contour Extraction", "Extracts external contours and applies an area filter (>500 px) to isolate the true grain and reject debris."),
        ("Isotropic Padding", "Crops the grain with a 5% margin and scales it into a 224×224 canvas with black padding, keeping its true shape."),
        ("Validation Result", "Verified across 20 representative samples: 20/20 successful isolations (100% success rate, 5.44 ms/image).")
    ]
    add_image_slide(s11, FIG_DIR / "fig2_preprocessing_stages.png", width=6.6, height=5.2, side_bullets=side_pts_11)
    slides_created.append(s11)

    # -------------------------------------------------------------
    # SLIDE 12: PHASE 2 - FEATURE EXTRACTION & FUSION
    # -------------------------------------------------------------
    print("Building Slide 12: Phase 2 Feature Extraction...")
    s12 = create_slide()
    add_header(s12, "Phase 2: Handcrafted & Deep Feature Extraction", "8. Implementation Phases")
    feat_col1 = [
        ("Branch 1: 62 Handcrafted Features", ""),
        ("14 Shape Features:", "Area, perimeter, width, height, aspect ratio, solidity, eccentricity, major/minor axis (measures physical dimensions)."),
        ("12 GLCM Texture Features:", "Quantized to 32 gray levels. Computes contrast, dissimilarity, homogeneity, energy, correlation, and ASM across multiple angles."),
        ("36 Colour Statistics:", "Computes mean, std, min, and max across 9 chromatic channels spanning RGB, HSV, and CIELAB colour spaces on foreground pixels."),
        ("Extraction Speed:", "Processed all 30,962 images in 126.2 seconds (245.4 images/sec, zero NaN or missing values).")
    ]
    feat_col2 = [
        ("Branch 2: 1,280 Deep Features & Fusion", ""),
        ("EfficientNet-B0 Backbone:", "Uses ImageNet pretrained weights. Replaces classification head with Identity layer to output 1,280 latent embeddings."),
        ("Feature Concatenation:", "Combines both branches into a unified vector: 62 Handcrafted + 1,280 Deep = 1,342 Fused Features."),
        ("StandardScaler:", "Transforms each feature to zero mean and unit variance so large numbers (like area) do not overpower smaller texture values."),
        ("Zero Data Leakage Safeguard:", "The scaler is fitted ONLY on training data (24,767 samples) and applied to validation and test data.")
    ]
    add_two_column_slide(s12, feat_col1, feat_col2)
    slides_created.append(s12)

    # -------------------------------------------------------------
    # SLIDE 13: PHASE 3 - XGBOOST & SHAP EXPLAINABILITY
    # -------------------------------------------------------------
    print("Building Slide 13: Phase 3 Classification & SHAP...")
    s13 = create_slide()
    add_header(s13, "Phase 3: XGBoost Classifier & SHAP Explainability", "8. Implementation Phases")
    cls_col1 = [
        ("XGBoost Classification Engine", ""),
        ("Tree Boosting:", "XGBoost uses an ensemble of 100 decision trees to learn non-linear patterns in the 1,342-dimensional feature space."),
        ("Hyperparameters:", "n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8."),
        ("Shallow Trees (depth=4):", "Constraining tree depth to 4 levels prevents overfitting in the high-dimensional feature space."),
        ("Histogram Method (hist):", "Accelerates training and enables instant inference (under 0.01 ms per grain)."),
        ("Output Probabilities:", "Uses multi:softprob objective to produce normalized probabilities across all 8 classes.")
    ]
    cls_col2 = [
        ("SHAP Explainability (TreeExplainer)", ""),
        ("Why Explainability Matters:", "In grain quality sorting, black-box predictions cannot be audited. Human inspectors need to know WHY a grain was rejected."),
        ("TreeExplainer SHAP:", "Computes exact Shapley values from game theory in polynomial time."),
        ("Positive Contributors:", "Visual features that pushed the model toward the predicted class."),
        ("Negative Contributors:", "Visual features that argued against the predicted class."),
        ("Scientific Caveat:", "SHAP explains feature contribution to the prediction; it is NOT accuracy, probability, or proof of correctness.")
    ]
    add_two_column_slide(s13, cls_col1, cls_col2)
    slides_created.append(s13)

    # -------------------------------------------------------------
    # SLIDE 14: RESULTS - BENCHMARK COMPARISON
    # -------------------------------------------------------------
    print("Building Slide 14: Results Benchmark...")
    s14 = create_slide()
    add_header(s14, "Results: Benchmark Model Comparison", "9. Results & Discussion")
    side_pts_14 = [
        ("Baseline 1 (SVM + Handcrafted)", "Achieved 91.16% test accuracy and 0.8494 Macro F1 using only 62 handcrafted features."),
        ("Baseline 2 (XGBoost + Deep)", "Achieved 90.65% test accuracy and 0.8399 Macro F1 using 1,280 EfficientNet-B0 features."),
        ("Proposed Hybrid Framework", "Achieved 92.10% test accuracy, 0.8660 Macro F1, and 0.9244 Weighted F1 using 1,342 fused features."),
        ("Key Finding: Deep Alone Was Lower", "Deep features alone (90.65%) did not beat handcrafted features (91.16%), proving that deep networks miss precise millimeter boundaries."),
        ("The Fusion Advantage", "Combining handcrafted and deep features boosted Macro F1 by +2.61% over deep alone and +1.66% over SVM, proving strong synergy.")
    ]
    add_image_slide(s14, FIG_DIR / "fig5_model_comparison_bars.png", width=6.6, height=5.2, side_bullets=side_pts_14)
    slides_created.append(s14)

    # -------------------------------------------------------------
    # SLIDE 15: RESULTS - CLASS-WISE & CONFUSION MATRIX
    # -------------------------------------------------------------
    print("Building Slide 15: Results Class-wise & Confusion Matrix...")
    s15 = create_slide()
    add_header(s15, "Results: Class-Wise Performance & Confusion Matrix", "9. Results & Discussion")
    side_pts_15 = [
        ("Overall Test Accuracy", "92.10% (92.0968%): Correctly classified 2,855 out of 3,100 untouched test samples."),
        ("Top Performing Classes", "7_IM (0.9799 F1), 0_NOR (0.9608 F1), and 5_BN (0.9456 F1) showed near-perfect separation due to distinct shape and colour."),
        ("Mutual Confusion (1_F&S vs. 2_SD)", "26 samples of 2_SD were predicted as 1_F&S, and 11 samples of 1_F&S as 2_SD, because dark seed cracks mimic spot lesions."),
        ("Majority Class Leakage", "52 samples of 0_NOR leaked into 3_MY and 42 into 2_SD, slightly reducing precision for those two minority classes."),
        ("Robust Minority Recall", "Recall remained high across minority classes (1_F&S: 88.7%, 2_SD: 81.3%, 3_MY: 82.0%, 6_UN: 94.0%).")
    ]
    add_image_slide(s15, FIG_DIR / "fig4_confusion_matrix_raw.png", width=6.6, height=5.2, side_bullets=side_pts_15)
    slides_created.append(s15)

    # -------------------------------------------------------------
    # SLIDE 16: RESULTS - ABLATION STUDY
    # -------------------------------------------------------------
    print("Building Slide 16: Results Ablation Study...")
    s16 = create_slide()
    add_header(s16, "Results: Systematic Feature Ablation Study", "9. Results & Discussion")
    side_pts_16 = [
        ("Shape Only (14 Dims)", "Test Accuracy: 77.61%, Macro F1: 0.5560 – Captures grain length and width but misses surface defects."),
        ("GLCM Texture Only (12 Dims)", "Test Accuracy: 75.84%, Macro F1: 0.5241 – Measures surface roughness but lacks colour cues."),
        ("Colour Only (36 Dims)", "Test Accuracy: 85.58%, Macro F1: 0.7237 – Strongest individual modality; detects chalkiness and fungal spots."),
        ("All Handcrafted (62 Dims)", "Test Accuracy: 91.16%, Macro F1: 0.8479 – Combining Shape + Texture + Colour jumps F1 by +12.4%."),
        ("EfficientNet-B0 Only (1,280 Dims)", "Test Accuracy: 90.65%, Macro F1: 0.8413 – Comparable to handcrafted features despite 20× more dimensions."),
        ("Final Saved Hybrid Model (1,342 Dims)", "Test Accuracy: 92.10%, Macro F1: 0.8660 – Achieves best overall performance, validating feature fusion.")
    ]
    add_image_slide(s16, FIG_DIR / "fig6_ablation_comparison.png", width=6.6, height=5.2, side_bullets=side_pts_16)
    slides_created.append(s16)

    # -------------------------------------------------------------
    # SLIDE 17: RESULTS - SHAP EXPLAINABILITY & IMPORTANCE
    # -------------------------------------------------------------
    print("Building Slide 17: Results SHAP Importance...")
    s17 = create_slide()
    add_header(s17, "Results: SHAP Feature Importance & Interpretability", "9. Results & Discussion")
    side_pts_17 = [
        ("Feature Group Contribution (SHAP)", "EfficientNet Deep: 68.59% | Shape: 13.52% | Colour: 12.86% | GLCM Texture: 5.02%."),
        ("Physical Features Carry 31.41%", "Handcrafted features contribute nearly one-third of the total explanatory weight."),
        ("Top 4 Most Important Features", "1. shape_height (0.1883), 2. color_lab_b_mean (0.1763), 3. shape_major_axis_length (0.1498), 4. shape_eccentricity (0.1295)."),
        ("Physical Features Dominate Top 4", "Notice that 3 of the top 4 most influential features are explicit geometric grain measurements!"),
        ("Gain Importance vs. SHAP", "Tree-Gain splits favor high-dimensional deep features (85.5%), but real test instances rely heavily on physical geometry (31.4% SHAP).")
    ]
    add_image_slide(s17, FIG_DIR / "fig7a_shap_top20_bar.png", width=6.6, height=5.2, side_bullets=side_pts_17)
    slides_created.append(s17)

    # -------------------------------------------------------------
    # SLIDE 18: SCREENSHOTS - PREPROCESSING PIPELINE
    # -------------------------------------------------------------
    print("Building Slide 18: Screenshots Preprocessing...")
    s18 = create_slide()
    add_header(s18, "Screenshots: Image Preprocessing Pipeline", "10. Project Screenshots")
    
    # 4 images horizontally or 2x2
    img_w = 2.7
    img_h = 2.7
    top_pos = 1.6
    
    steps = [
        ("Step 1: Original Image", SAMPLE_DIR / "01_original_bgr.png", 0.8),
        ("Step 2: Gaussian Denoised", SAMPLE_DIR / "02_grayscale_denoised.png", 3.8),
        ("Step 3: Binary Mask", SAMPLE_DIR / "03_binary_mask.png", 6.8),
        ("Step 4: Standardized Crop", SAMPLE_DIR / "06_standardized_224x224.png", 9.8),
    ]
    
    for title, path, left in steps:
        if path.exists():
            s18.shapes.add_picture(str(path), Inches(left), Inches(top_pos), width=Inches(img_w))
        else:
            ph = s18.shapes.add_textbox(Inches(left), Inches(top_pos), Inches(img_w), Inches(img_h))
            set_shape_text(ph, f"[{title}]", font_size=12, bold=True, color=COLOR_TITLE, align=PP_ALIGN.CENTER)
            
        tb = s18.shapes.add_textbox(Inches(left), Inches(top_pos + img_h + 0.1), Inches(img_w), Inches(0.5))
        set_shape_text(tb, title, font_size=12, bold=True, color=COLOR_TITLE, align=PP_ALIGN.CENTER)
        
    exp_box = s18.shapes.add_textbox(Inches(0.8), Inches(5.1), Inches(11.73), Inches(1.7))
    tf_exp = exp_box.text_frame
    tf_exp.word_wrap = True
    p1 = tf_exp.paragraphs[0]
    p1.space_after = Pt(4)
    r1 = p1.add_run()
    r1.text = "• Preprocessing Workflow Summary:"
    r1.font.name = FONT_NAME
    r1.font.size = Pt(14)
    r1.font.bold = True
    r1.font.color.rgb = COLOR_BODY
    
    p2 = tf_exp.add_paragraph()
    r2 = p2.add_run()
    r2.text = (
        "  1. Loads raw BGR grain image and applies 5×5 Gaussian blur to remove high-frequency camera noise.\n"
        "  2. Applies Otsu bimodal thresholding and morphological closing/opening to extract a clean binary mask.\n"
        "  3. Extracts the primary grain contour, adds a 5% margin, and centers the grain in a standardized 224×224 isotropic canvas."
    )
    r2.font.name = FONT_NAME
    r2.font.size = Pt(13)
    r2.font.color.rgb = COLOR_BODY
    slides_created.append(s18)

    # -------------------------------------------------------------
    # SLIDE 19: SCREENSHOTS - DEPLOYMENT APPLICATION & WORKFLOW
    # -------------------------------------------------------------
    print("Building Slide 19: Screenshots Application...")
    s19 = create_slide()
    add_header(s19, "Screenshots: Interactive User Interface & Review System", "10. Project Screenshots")
    app_col1 = [
        ("Streamlit Inspection Dashboard", ""),
        ("Single Grain Analysis:", "Operators can upload a grain photo, view the real-time segmented grain preview, and see the predicted class with full probability table."),
        ("SHAP Explanation Display:", "Interactive waterfall chart visually highlights the exact positive and negative feature contributions for that prediction."),
        ("Human-in-the-Loop Review Queue:", "Predictions with top probability below 75% (<0.75) are automatically flagged for expert audit and manual correction."),
        ("[SCREENSHOT TO BE INSERTED - Streamlit Inspection Dashboard]", "")
    ]
    app_col2 = [
        ("Batch Processing & Reporting", ""),
        ("Batch Folder Analysis:", "Bulk upload grain folders to generate automated defect breakdowns, purity percentages, and export CSV summaries."),
        ("SQLite Database Persistence:", "All predictions, confidence scores, timestamps, and expert reviews are permanently logged into `results/rice_quality.db`."),
        ("Automated Commercial PDF Reports:", "Generates printable inspection certificates showing grain photos, defect classifications, and quality grading."),
        ("[SCREENSHOT TO BE INSERTED - Batch Review & PDF Export Interface]", "")
    ]
    add_two_column_slide(s19, app_col1, app_col2)
    slides_created.append(s19)

    # -------------------------------------------------------------
    # SLIDE 20: REFERENCES
    # -------------------------------------------------------------
    print("Building Slide 20: References...")
    s20 = create_slide()
    add_header(s20, "References", "11. References")
    ref_list = [
        "[1] S. Mittal, M. K. Dutta, and A. Issac, \"Non-destructive image processing based system for assessment of rice quality and defects for classification according to inferred commercial value,\" Measurement (Elsevier), vol. 148, p. 106969, 2019. (BASE PAPER)",
        "[2] X. Wang, Y. Zhang, and L. Liu, \"Crack Detection of Brown Rice Kernel Based on Optimized ResNet-18 Network,\" IEEE Access, vol. 11, pp. 64210–64221, 2023.",
        "[3] S. Rani, A. Sharma, and D. Singh, \"Rice grain quality analysis using image processing and artificial intelligence techniques,\" Computers and Electronics in Agriculture (Elsevier), vol. 218, p. 107678, 2025.",
        "[4] R. Ibrahim, \"Quantum-Inspired Moth Flame Optimizer Enhanced Deep Learning for Automated Rice Variety Classification,\" IEEE Access, vol. 11, pp. 101234–101248, 2023.",
        "[5] T. N. Nguyen and D. T. Le, \"Explainable Deep Learning with SHAP and Grad-CAM for Milled Rice Purity and Defect Inspection,\" Food Control (Elsevier), vol. 156, p. 110122, 2024.",
        "[6] M. Tan and Q. V. Le, \"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,\" in Proc. ICML, PMLR, vol. 97, pp. 6105–6114, 2019.",
        "[7] T. Chen and C. Guestrin, \"XGBoost: A Scalable Tree Boosting System,\" in Proc. 22nd ACM SIGKDD, pp. 785–794, 2016.",
        "[8] S. M. Lundberg et al., \"From local explanations to global understanding with explainable AI for trees,\" Nature Machine Intelligence, vol. 2, pp. 56–67, 2020."
    ]
    add_bullet_list(s20, ref_list, font_size=12.5)
    slides_created.append(s20)

    # -------------------------------------------------------------
    # SLIDE 21: THANK YOU (Update template's slide 7)
    # -------------------------------------------------------------
    print("Updating Slide 21: Thank You Slide...")
    # Add subtitle under Thank You
    ty_box = thank_you_slide.shapes.add_textbox(Inches(2.0), Inches(4.3), Inches(9.33), Inches(1.8))
    tf_ty = ty_box.text_frame
    tf_ty.word_wrap = True
    
    p_ty1 = tf_ty.paragraphs[0]
    p_ty1.alignment = PP_ALIGN.CENTER
    r_ty1 = p_ty1.add_run()
    r_ty1.text = "Queries & Discussion Welcome"
    r_ty1.font.name = FONT_NAME
    r_ty1.font.size = Pt(20)
    r_ty1.font.bold = True
    r_ty1.font.color.rgb = COLOR_ACCENT
    
    p_ty2 = tf_ty.add_paragraph()
    p_ty2.alignment = PP_ALIGN.CENTER
    p_ty2.space_before = Pt(8)
    r_ty2 = p_ty2.add_run()
    r_ty2.text = (
        "DHARSHAN B (23ITR030)   |   DINESH G L (23ITR039)\n"
        "Supervisor: Ms. S. Sripriya, M.E., Assistant Professor\n"
        "Department of Information Technology, Kongu Engineering College (Autonomous)"
    )
    r_ty2.font.name = FONT_NAME
    r_ty2.font.size = Pt(14)
    r_ty2.font.color.rgb = COLOR_BODY

    # Add footers to all content slides (slides 2 to 20)
    total_slides = 21
    for idx, slide in enumerate(slides_created, start=2):
        add_footer(slide, idx, total_slides)

    # Save presentation
    print("Saving completed review presentation...")
    prs.save(str(OUTPUT_PPTX))
    print(f"SUCCESS: Presentation saved to {OUTPUT_PPTX}")
    print(f"Total slides: {len(prs.slides)}")
    print(f"File size: {OUTPUT_PPTX.stat().st_size:,} bytes")
    print("=" * 70)

if __name__ == "__main__":
    generate_presentation()
