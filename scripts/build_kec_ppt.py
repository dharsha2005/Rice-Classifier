import os
from pathlib import Path
from lxml import etree
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Inches

BASE = Path("c:/Rice classifier final project")
TEMPLATE = BASE / "kec template.pptx"
FIGS = BASE / "paper" / "figures"
OUT = BASE / "27PR27_Rice_Classifier_Presentation.pptx"
SW = Inches(13.333)
SH = Inches(7.5)
FONT = "Times New Roman"
BT = Inches(0.72)
BL = Inches(0.25)
BW = Inches(12.85)

def rh(r, g, b):
    return f"{r:02X}{g:02X}{b:02X}"

def atb(slide, left, top, width, height, paras, anchor="t", autofit=False):
    NP = "http://schemas.openxmlformats.org/presentationml/2006/main"
    NA = "http://schemas.openxmlformats.org/drawingml/2006/main"
    sp = etree.SubElement(slide.shapes._spTree, f"{{{NP}}}sp")
    nv = etree.SubElement(sp, f"{{{NP}}}nvSpPr")
    uid = max((int(sh._element.get("id",0)) for sh in slide.shapes if sh._element.get("id")), default=100)+1
    cnp = etree.SubElement(nv, f"{{{NP}}}cNvPr")
    cnp.set("id", str(uid)); cnp.set("name", f"TB{uid}")
    etree.SubElement(nv, f"{{{NP}}}cNvSpPr").set("txBox","1")
    etree.SubElement(nv, f"{{{NP}}}nvPr")
    pp = etree.SubElement(sp, f"{{{NP}}}spPr")
    xf = etree.SubElement(pp, f"{{{NA}}}xfrm")
    of = etree.SubElement(xf, f"{{{NA}}}off"); of.set("x",str(int(left))); of.set("y",str(int(top)))
    ex = etree.SubElement(xf, f"{{{NA}}}ext"); ex.set("cx",str(int(width))); ex.set("cy",str(int(height)))
    pg = etree.SubElement(pp, f"{{{NA}}}prstGeom"); pg.set("prst","rect"); etree.SubElement(pg,f"{{{NA}}}avLst")
    etree.SubElement(pp, f"{{{NA}}}noFill")
    ln = etree.SubElement(pp, f"{{{NA}}}ln"); etree.SubElement(ln,f"{{{NA}}}noFill")
    tx = etree.SubElement(sp, f"{{{NP}}}txBody")
    bp = etree.SubElement(tx, f"{{{NA}}}bodyPr")
    bp.set("wrap","square"); bp.set("lIns","90000"); bp.set("tIns","46800")
    bp.set("rIns","90000"); bp.set("bIns","46800"); bp.set("anchor",anchor); bp.set("anchorCtr","0")
    if autofit: etree.SubElement(bp,f"{{{NA}}}normAutofit")
    else: etree.SubElement(bp,f"{{{NA}}}noAutofit")
    etree.SubElement(tx, f"{{{NA}}}lstStyle")
    for (text,sz,bold,color,align,sb,sa,lspc) in paras:
        p = etree.SubElement(tx, f"{{{NA}}}p")
        ppr = etree.SubElement(p, f"{{{NA}}}pPr")
        ppr.set("marL","0"); ppr.set("marR","0"); ppr.set("indent","0"); ppr.set("algn",align)
        ls = etree.SubElement(ppr,f"{{{NA}}}lnSpc"); etree.SubElement(ls,f"{{{NA}}}spcPct").set("val",str(lspc*1000))
        sb2 = etree.SubElement(ppr,f"{{{NA}}}spcBef"); etree.SubElement(sb2,f"{{{NA}}}spcPts").set("val",str(sb))
        sa2 = etree.SubElement(ppr,f"{{{NA}}}spcAft"); etree.SubElement(sa2,f"{{{NA}}}spcPts").set("val",str(sa))
        r = etree.SubElement(p, f"{{{NA}}}r")
        rp = etree.SubElement(r, f"{{{NA}}}rPr")
        rp.set("lang","en-US"); rp.set("sz",str(sz)); rp.set("b","1" if bold else "0")
        rp.set("i","0"); rp.set("u","none"); rp.set("strike","noStrike")
        if color:
            sf = etree.SubElement(rp, f"{{{NA}}}solidFill"); etree.SubElement(sf,f"{{{NA}}}srgbClr").set("val",rh(*color))
        lt = etree.SubElement(rp,f"{{{NA}}}latin"); lt.set("typeface",FONT)
        cs = etree.SubElement(rp,f"{{{NA}}}cs"); cs.set("typeface",FONT)
        t = etree.SubElement(r,f"{{{NA}}}t"); t.text = text
    return sp

def arect(slide, left, top, width, height, fill):
    NP = "http://schemas.openxmlformats.org/presentationml/2006/main"
    NA = "http://schemas.openxmlformats.org/drawingml/2006/main"
    sp = etree.SubElement(slide.shapes._spTree, f"{{{NP}}}sp")
    nv = etree.SubElement(sp, f"{{{NP}}}nvSpPr")
    uid = max((int(sh._element.get("id",0)) for sh in slide.shapes if sh._element.get("id")), default=200)+1
    cnp = etree.SubElement(nv, f"{{{NP}}}cNvPr"); cnp.set("id",str(uid)); cnp.set("name",f"R{uid}")
    etree.SubElement(nv, f"{{{NP}}}cNvSpPr"); etree.SubElement(nv, f"{{{NP}}}nvPr")
    pp = etree.SubElement(sp, f"{{{NP}}}spPr")
    xf = etree.SubElement(pp, f"{{{NA}}}xfrm")
    of = etree.SubElement(xf, f"{{{NA}}}off"); of.set("x",str(int(left))); of.set("y",str(int(top)))
    ex = etree.SubElement(xf, f"{{{NA}}}ext"); ex.set("cx",str(int(width))); ex.set("cy",str(int(height)))
    pg = etree.SubElement(pp, f"{{{NA}}}prstGeom"); pg.set("prst","rect"); etree.SubElement(pg,f"{{{NA}}}avLst")
    sf = etree.SubElement(pp, f"{{{NA}}}solidFill"); etree.SubElement(sf,f"{{{NA}}}srgbClr").set("val",rh(*fill))
    ln = etree.SubElement(pp, f"{{{NA}}}ln"); etree.SubElement(ln,f"{{{NA}}}noFill")
    tb = etree.SubElement(sp, f"{{{NP}}}txBody")
    etree.SubElement(tb,f"{{{NA}}}bodyPr"); etree.SubElement(tb,f"{{{NA}}}lstStyle"); etree.SubElement(tb,f"{{{NA}}}p")
    return sp

def aimg(slide, path, left, top, width, height):
    return slide.shapes.add_picture(str(path), left, top, width, height)

def bslide(prs):
    return prs.slides.add_slide(prs.slide_layouts[2])

def hdr(slide, title, num):
    bh = Inches(0.62)
    arect(slide, 0, 0, SW, bh, (0x06,0x32,0x5A))
    atb(slide, Inches(0.15), Inches(0.07), Inches(12.0), bh-Inches(0.06),
        [(title, 2400, True, (0xFF,0xFF,0xFF), "l", 0, 0, 100)], anchor="ctr")
    atb(slide, Inches(12.5), Inches(7.1), Inches(0.7), Inches(0.32),
        [(str(num), 1400, False, (0x88,0x98,0xC3), "ctr", 0, 0, 100)])
    atb(slide, Inches(3.5), Inches(7.1), Inches(6.5), Inches(0.32),
        [("Dept. of IT | Kongu Engineering College", 1000, False, (0x88,0x98,0xC3), "ctr", 0, 0, 100)])

def lbl(slide, text, left, top, width=None, sz=1900):
    w = width or Inches(12.0)
    atb(slide, left, top, w, Inches(0.4), [(text, sz, True, (0x0B,0x5C,0x95), "l", 0, 0, 100)])

def blt(slide, items, left, top, width, height, sz=1700, color=None):
    WHITE = (0xFF,0xFF,0xFF)
    c = color or WHITE
    paras = [(f"  \u2022  {item}", sz, False, c, "l", 60, 0, 115) for item in items]
    atb(slide, left, top, width, height, paras, anchor="t", autofit=True)

def cslide(prs, title, num, fn):
    s = bslide(prs)
    hdr(s, title, num)
    fn(s)
    return s

# ===== SLIDE BODIES =====
def s_outline(s):
    items = ["Introduction & Motivation","Problem Statement","Objectives",
             "Literature Survey","System Architecture & Methodology",
             "Dataset & Preprocessing","Feature Extraction & Fusion",
             "Model Training & Explainability (SHAP)",
             "Results & Performance Comparison","Ablation Study",
             "Conclusion & Future Work","References"]
    paras = [(f"  {i+1}.  {item}", 1900, i==0,
              (0x0B,0x5C,0x95) if i==0 else (0xFF,0xFF,0xFF),
              "l", 120, 0, 115) for i,item in enumerate(items)]
    atb(s, BL, BT, BW, Inches(6.6), paras, anchor="t")

def s_intro(s):
    lbl(s, "Background & Motivation", BL, BT)
    blt(s, ["Rice is a staple food for 3.5 billion people worldwide",
            "Global annual production exceeds 520 million metric tons",
            "Post-harvest quality defects cause 20-30% economic losses",
            "Manual inspection is slow, subjective and error-prone",
            "Automation via computer vision + ML is the industry need"],
        BL, BT+Inches(0.45), Inches(6.3), Inches(2.6))
    lbl(s, "Why Deep Learning + Explainability?", BL, BT+Inches(3.2))
    blt(s, ["EfficientNet-B0: SOTA CNN with compound scaling",
            "XGBoost: robust gradient boosting for tabular feature fusion",
            "SHAP: model-agnostic explainability for regulatory trust",
            "Hybrid framework bridges accuracy and interpretability"],
        BL, BT+Inches(3.65), Inches(9.0), Inches(2.5))
    fig = FIGS / "fig2_preprocessing_stages.png"
    if fig.exists(): aimg(s, fig, Inches(7.2), BT+Inches(0.35), Inches(5.8), Inches(3.1))

def s_problem(s):
    lbl(s, "Problem Statement", BL, BT)
    atb(s, BL, BT+Inches(0.5), BW, Inches(1.6),
        [("Current rice quality inspection relies on manual visual grading which is labour-intensive, "
          "inconsistent, and unscalable. Existing automated approaches either lack multi-class defect "
          "granularity or operate as black-box models unfit for food-safety audits.",
          1750, False, (0xFF,0xFF,0xFF), "l", 0, 0, 140)], anchor="t")
    lbl(s, "Key Challenges", BL, BT+Inches(2.2))
    blt(s, ["High intra-class visual similarity and inter-class colour overlap",
            "Limited labelled data with imbalanced class distribution",
            "Need for real-time, explainable classification at production scale",
            "Bridging handcrafted domain features with deep CNN embeddings",
            "Meeting food regulatory requirements for transparent AI decisions"],
        BL, BT+Inches(2.65), BW, Inches(3.8))

def s_objectives(s):
    lbl(s, "Project Objectives", BL, BT)
    blt(s, ["Develop end-to-end pipeline for 8-class rice grain quality classification",
            "Implement EfficientNet-B0 for deep feature extraction (transfer learning)",
            "Engineer 32 handcrafted features: 6 colour, 9 texture LBP, 10 shape, 7 spectral",
            "Fuse deep 1280-d with handcrafted 32-d => 1312-d hybrid feature vector",
            "Train XGBoost classifier on fused feature set for final classification",
            "Apply SHAP for feature-level interpretability and explainability",
            "Deploy Streamlit dashboard for live inference and batch analysis",
            "Achieve >99% test accuracy surpassing all baseline models"],
        BL, BT+Inches(0.45), BW, Inches(6.0))

def s_literature(s):
    lbl(s, "Literature Survey - Key Works", BL, BT)
    refs = [
        ("[1] Murthy et al. (2023) - CNN-based rice quality grading; 94.2% accuracy",
         "Gap: No hybrid feature fusion or SHAP analysis"),
        ("[2] Zhao & Li (2022) - EfficientNet transfer learning for grain classification; 96.8%",
         "Gap: Single deep-feature stream; no handcrafted features"),
        ("[3] Ahmad et al. (2021) - XGBoost for agricultural defect detection; 92%",
         "Gap: Relied on handcrafted features only, no deep embeddings"),
        ("[4] Lundberg & Lee (2017) - SHAP Unified Framework for ML explainability",
         "Enabler: Adopted as explainability layer in our framework"),
        ("[5] Tan & Le (2019) - EfficientNet: Compound model scaling; ImageNet SOTA",
         "Enabler: Backbone for feature extraction"),
    ]
    y = BT + Inches(0.45)
    for (ref, gap) in refs:
        atb(s, BL, y, BW, Inches(0.38),
            [(ref, 1550, True, (0x0B,0x5C,0x95), "l", 0, 0, 100)], anchor="t")
        atb(s, BL+Inches(0.2), y+Inches(0.36), BW-Inches(0.2), Inches(0.30),
            [(f"    -> {gap}", 1450, False, (0xC0,0x00,0x00), "l", 0, 0, 100)], anchor="t")
        y += Inches(0.84)

def s_arch(s):
    lbl(s, "System Architecture - Hybrid Feature-Fusion Framework", BL, BT)
    fig = FIGS / "fig1_overall_framework.png"
    if fig.exists(): aimg(s, fig, Inches(0.2), BT+Inches(0.4), Inches(12.9), Inches(5.9))

def s_dataset(s):
    lbl(s, "Dataset & Preprocessing", BL, BT)
    atb(s, BL, BT+Inches(0.45), Inches(5.8), Inches(2.8),
        [("Dataset Statistics", 1800, True, (0x0B,0x5C,0x95), "l", 0, 0, 100),
         ("  Source: Grainset KEC Rice Dataset (real farm imagery)", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Total Images: 22,250 rice grain images", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Classes: 8 - Normal, Fusarium & Smut, Stem-borer Damage,", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("           Mycotoxin, Aflatoxin, Brown Nigrescence, Unripened, Immature", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Split: 70% Train / 15% Val / 15% Test", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Augmentation: Flip, Rotation +/-30deg, Colour Jitter, Zoom", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ], anchor="t")
    atb(s, Inches(6.3), BT+Inches(0.45), Inches(6.8), Inches(2.8),
        [("Preprocessing Pipeline (6 Stages)", 1800, True, (0x0B,0x5C,0x95), "l", 0, 0, 100),
         ("  1. BGR -> Grayscale conversion", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  2. Gaussian denoising (sigma=1.0)", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  3. Otsu binary thresholding for segmentation", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  4. Masked grain extraction (background removal)", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  5. Bounding-box crop to grain region", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  6. Resize & normalise to 224x224 px (ImageNet stats)", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ], anchor="t")
    fig = FIGS / "fig3_preprocessing_grid.png"
    if fig.exists(): aimg(s, fig, Inches(0.2), BT+Inches(3.4), Inches(12.9), Inches(3.0))

def s_features(s):
    lbl(s, "Feature Extraction & Hybrid Fusion", BL, BT)
    atb(s, BL, BT+Inches(0.45), Inches(6.0), Inches(3.0),
        [("Deep Features (EfficientNet-B0)", 1750, True, (0x0B,0x5C,0x95), "l", 0, 0, 100),
         ("  Pretrained on ImageNet (1000 classes)", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Fine-tuned: last 20 layers unfrozen", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Global Average Pooling => 1280-d vector", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Dropout 0.3 for regularisation", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ], anchor="t")
    atb(s, Inches(6.4), BT+Inches(0.45), Inches(6.5), Inches(3.0),
        [("Handcrafted Features (32-d)", 1750, True, (0x0B,0x5C,0x95), "l", 0, 0, 100),
         ("  Colour: 6-bin HSV histogram (6 features)", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Texture: LBP histogram (9 features)", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Shape: Solidity, eccentricity, extent (10 features)", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Spectral: FFT energy, dominant freq (7 features)", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ], anchor="t")
    lbl(s, "Feature Fusion => XGBoost Classification", BL, BT+Inches(3.55))
    atb(s, BL, BT+Inches(4.0), BW, Inches(2.6),
        [("  Concatenation: [1280-d deep] + [32-d handcrafted] = 1312-d fused vector", 1650, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  XGBoost: n_estimators=500, max_depth=6, lr=0.05, subsample=0.8", 1650, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  5-fold stratified cross-validation for hyperparameter tuning", 1650, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Label encoding for 8-class output", 1650, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ], anchor="t")

def s_training(s):
    lbl(s, "Model Training & SHAP Explainability", BL, BT)
    atb(s, BL, BT+Inches(0.45), Inches(6.3), Inches(2.8),
        [("EfficientNet-B0 Training (Phase 1)", 1750, True, (0x0B,0x5C,0x95), "l", 0, 0, 100),
         ("  Optimizer: Adam, lr=1e-4 => CosineAnnealing", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Batch size: 32, Epochs: 50", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Loss: Categorical Cross-Entropy", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Early stopping (patience=10)", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Best val-acc: 98.7%", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ], anchor="t")
    atb(s, Inches(6.5), BT+Inches(0.45), Inches(6.5), Inches(2.8),
        [("XGBoost Training (Phase 2)", 1750, True, (0x0B,0x5C,0x95), "l", 0, 0, 100),
         ("  Input: 1312-d fused feature vectors", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Bayesian hyperparameter optimisation", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  SMOTE for class-imbalance handling", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Train accuracy: 99.97%", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Test accuracy:  99.41%", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ], anchor="t")
    lbl(s, "SHAP Explainability", BL, BT+Inches(3.35))
    fig = FIGS / "fig7a_shap_top20_bar.png"
    if fig.exists(): aimg(s, fig, Inches(0.2), BT+Inches(3.8), Inches(6.2), Inches(2.65))
    fig2 = FIGS / "fig8_shap_group_contribution.png"
    if fig2.exists(): aimg(s, fig2, Inches(6.7), BT+Inches(3.8), Inches(6.2), Inches(2.65))

def s_results(s):
    lbl(s, "Results & Performance Metrics", BL, BT)
    headers = ["Model","Val Acc","Test Acc","Macro F1","Prec","Recall"]
    rows = [
        ["SVM (Baseline)","84.2%","83.7%","0.835","0.842","0.831"],
        ["EfficientNet-B0 Only","97.1%","96.8%","0.968","0.971","0.966"],
        ["EfficientNet + Handcrafted","98.3%","98.0%","0.980","0.983","0.979"],
        ["Hybrid (Ours - Final)","99.5%","99.41%","0.994","0.994","0.994"],
    ]
    col_w = [Inches(3.5),Inches(1.5),Inches(1.5),Inches(1.5),Inches(1.4),Inches(1.4)]
    xs = [Inches(0.25)]
    for cw in col_w[:-1]: xs.append(xs[-1]+cw)
    rh2 = Inches(0.42)
    y0 = BT+Inches(0.48)
    for ci,(hdr2,x,cw) in enumerate(zip(headers,xs,col_w)):
        arect(s, x, y0, cw-Inches(0.02), rh2, (0x06,0x32,0x5A))
        atb(s, x+Inches(0.04), y0+Inches(0.06), cw-Inches(0.06), rh2-Inches(0.08),
            [(hdr2,1500,True,(0xFF,0xFF,0xFF),"ctr",0,0,100)], anchor="ctr")
    for ri,row in enumerate(rows):
        ry = y0+rh2*(ri+1)+Inches(0.02)
        hl = (ri==len(rows)-1)
        bg = (0xC6,0xEF,0xCE) if hl else ((0xE8,0xF0,0xF8) if ri%2==0 else (0xF5,0xF5,0xF5))
        tc = (0x00,0x70,0x00) if hl else (0x00,0x00,0x00)
        for ci,(cell,x,cw) in enumerate(zip(row,xs,col_w)):
            arect(s, x, ry, cw-Inches(0.02), rh2, bg)
            atb(s, x+Inches(0.04), ry+Inches(0.06), cw-Inches(0.06), rh2-Inches(0.08),
                [(cell,1500,hl,tc,"ctr",0,0,100)], anchor="ctr")
    fig = FIGS / "fig5_model_comparison_bars.png"
    if fig.exists(): aimg(s, fig, Inches(0.2), BT+Inches(2.75), Inches(6.3), Inches(3.7))
    fig2 = FIGS / "fig4_confusion_matrix_normalized.png"
    if fig2.exists(): aimg(s, fig2, Inches(6.7), BT+Inches(2.75), Inches(6.2), Inches(3.7))

def s_classwise(s):
    lbl(s, "Per-Class Performance Analysis", BL, BT)
    fig = FIGS / "fig10a_per_class_metrics_hybrid.png"
    if fig.exists(): aimg(s, fig, Inches(0.2), BT+Inches(0.4), Inches(6.3), Inches(5.9))
    fig2 = FIGS / "fig10_per_class_f1_comparison.png"
    if fig2.exists(): aimg(s, fig2, Inches(6.7), BT+Inches(0.4), Inches(6.2), Inches(5.9))

def s_ablation(s):
    lbl(s, "Ablation Study", BL, BT)
    atb(s, BL, BT+Inches(0.45), Inches(5.8), Inches(2.5),
        [("Study Design", 1750, True, (0x0B,0x5C,0x95), "l", 0, 0, 100),
         ("  Five ablation configurations tested:", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  A1: EfficientNet + All 4 feature groups (Full model)", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  A2: Remove Colour features -> -0.8% accuracy", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  A3: Remove Texture (LBP) -> -1.1% accuracy", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  A4: Remove Shape features -> -0.5% accuracy", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  A5: Remove ALL handcrafted -> -1.9% accuracy", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ], anchor="t")
    atb(s, Inches(6.2), BT+Inches(0.45), Inches(6.8), Inches(2.5),
        [("Key Findings", 1750, True, (0x0B,0x5C,0x95), "l", 0, 0, 100),
         ("  Texture (LBP) is the most discriminative handcrafted group", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  All 4 groups contribute non-redundantly to performance", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Removing ALL handcrafted drops accuracy by 1.9%", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Deep-only model still outperforms all baselines", 1600, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ], anchor="t")
    fig = FIGS / "fig6_ablation_comparison.png"
    if fig.exists(): aimg(s, fig, Inches(0.2), BT+Inches(3.05), Inches(12.9), Inches(3.6))

def s_shap(s):
    lbl(s, "SHAP Explainability - Feature Importance Deep Dive", BL, BT)
    atb(s, BL, BT+Inches(0.45), BW, Inches(0.85),
        [("  SHAP TreeExplainer applied on final XGBoost classifier", 1650, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  200-sample background set for efficient Shapley value computation", 1650, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ("  Top-20 features ranked by mean |SHAP| across all classes", 1650, False, (0xFF,0xFF,0xFF), "l", 60, 0, 115),
         ], anchor="t")
    fig1 = FIGS / "fig7b_shap_beeswarm_summary.png"
    if fig1.exists(): aimg(s, fig1, Inches(0.2), BT+Inches(1.5), Inches(6.5), Inches(5.0))
    fig2 = FIGS / "fig9_gain_vs_shap_comparison.png"
    if fig2.exists(): aimg(s, fig2, Inches(6.9), BT+Inches(1.5), Inches(6.2), Inches(5.0))

def s_conclusion(s):
    lbl(s, "Conclusion & Future Work", BL, BT)
    atb(s, BL, BT+Inches(0.45), BW, Inches(0.35),
        [("Conclusion", 1800, True, (0x0B,0x5C,0x95), "l", 0, 0, 100)], anchor="t")
    blt(s, ["Achieved 99.41% test accuracy on 8-class rice defect classification",
            "Hybrid feature fusion (deep + handcrafted) outperforms all single-stream baselines",
            "SHAP confirms texture (LBP) and spectral features drive model decisions",
            "Ablation study validates each feature group unique contribution",
            "Streamlit dashboard enables real-time single-image and batch inference",
            "Model is interpretable, auditable, and ready for food-safety applications"],
        BL, BT+Inches(0.9), BW, Inches(2.8))
    atb(s, BL, BT+Inches(3.85), BW, Inches(0.35),
        [("Future Work", 1800, True, (0x0B,0x5C,0x95), "l", 0, 0, 100)], anchor="t")
    blt(s, ["Extend to multi-grain species and region-specific disease profiles",
            "Integrate attention maps (Grad-CAM) for spatial explainability",
            "Deploy on edge hardware (Raspberry Pi / Jetson Nano) for field-level use",
            "Explore Vision Transformer (ViT) backbone as EfficientNet replacement",
            "Add contamination severity scoring (mild / moderate / severe grades)"],
        BL, BT+Inches(4.3), BW, Inches(2.2))

def s_references(s):
    lbl(s, "References", BL, BT)
    refs = [
        "[1] Murthy, C. V. R., et al. (2023). Deep learning for rice quality grading. Comput. Electron. Agric., 195, 106850.",
        "[2] Zhao, P. & Li, X. (2022). EfficientNet-based transfer learning for grain defect detection. IEEE Access, 10.",
        "[3] Ahmad, I., et al. (2021). XGBoost for post-harvest agricultural defect classification. Expert Syst. Appl., 178.",
        "[4] Lundberg, S. M. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.",
        "[5] Tan, M. & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML 2019.",
        "[6] Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.",
        "[7] He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.",
        "[8] Ojala, T., et al. (2002). Multiresolution gray-scale texture classification with LBP. IEEE TPAMI, 24(7).",
    ]
    y = BT + Inches(0.45)
    for ref in refs:
        atb(s, BL, y, BW, Inches(0.65), [(ref, 1450, False, (0xFF,0xFF,0xFF), "l", 0, 0, 120)], anchor="t")
        y += Inches(0.70)

# ===== MAIN =====
def del_slide(prs, index):
    sls = prs.slides._sldIdLst
    rid = sls[index].get(qn("r:id"))
    sls.remove(sls[index])
    del prs.part.related_parts[rid]

def build_title(prs):
    src = prs.slides[0]
    for shape in src.shapes:
        if shape.has_text_frame:
            tf = shape.text_frame
            full = tf.text
            if "Rice Mill" in full or "IoT" in full:
                for para in tf.paragraphs:
                    for run in para.runs: run.text = ""
                p0 = tf.paragraphs[0]
                new_title = ("An Explainable Hybrid Feature-Fusion Framework for "
                             "Rice Quality and Defect Classification Using EfficientNet-B0 and XGBoost")
                if p0.runs: p0.runs[0].text = new_title
                else:
                    from pptx.util import Pt
                    r = p0.add_run(); r.text = new_title
            elif "PROJECT MEMBERS" in full:
                for para in tf.paragraphs:
                    for run in para.runs: run.text = ""
                pp = tf.paragraphs
                if len(pp)>=1 and pp[0].runs: pp[0].runs[0].text = "PROJECT MEMBERS"
                if len(pp)>=3 and pp[2].runs: pp[2].runs[0].text = "      DHARSHAN B (23ITR030)"
                if len(pp)>=4 and pp[3].runs: pp[3].runs[0].text = "      DINESH G L (23ITR039)"
                if len(pp)>=5:
                    for run in pp[4].runs: run.text = ""
                if len(pp)>=6 and pp[5].runs: pp[5].runs[0].text = "PROJECT GUIDE"
                if len(pp)>=7:
                    for run in pp[6].runs: run.text = "      Ms. S. Sripriya"
                if len(pp)>=8:
                    for run in pp[7].runs: run.text = "      Assistant Professor / Dept. of IT"
                if len(pp)>=9:
                    for run in pp[8].runs: run.text = "      Kongu Engineering College"
    atb(src, Inches(12.5), Inches(7.1), Inches(0.7), Inches(0.32),
        [("1", 1400, False, (0x88,0x98,0xC3), "ctr", 0, 0, 100)])

def main():
    prs = Presentation(str(TEMPLATE))
    print(f"Template loaded: {len(prs.slides)} slides")
    build_title(prs)
    spec = [
        (2,"Outline",s_outline),(3,"Introduction & Motivation",s_intro),
        (4,"Problem Statement",s_problem),(5,"Objectives",s_objectives),
        (6,"Literature Survey",s_literature),(7,"System Architecture",s_arch),
        (8,"Dataset & Preprocessing",s_dataset),(9,"Feature Extraction & Fusion",s_features),
        (10,"Model Training & Explainability",s_training),(11,"Results & Performance Metrics",s_results),
        (12,"Per-Class Performance Analysis",s_classwise),(13,"Ablation Study",s_ablation),
        (14,"SHAP Explainability",s_shap),(15,"Conclusion & Future Work",s_conclusion),
        (16,"References",s_references),
    ]
    for (num,title,fn) in spec:
        cslide(prs, title, num, fn)
        print(f"  Slide {num}: {title}")
    for idx in range(6, 0, -1):
        try: del_slide(prs, idx); print(f"  Deleted old slide {idx}")
        except Exception as e: print(f"  Skip {idx}: {e}")
    ty = prs.slides.add_slide(prs.slide_layouts[1])
    for shape in list(ty.shapes): shape._element.getparent().remove(shape._element)
    atb(ty, Inches(3.5), Inches(2.8), Inches(6.5), Inches(1.0),
        [("Thank You!", 4800, True, (0xC0,0x00,0x00), "ctr", 0, 0, 100)], anchor="ctr")
    atb(ty, Inches(2.0), Inches(4.0), Inches(9.5), Inches(0.7),
        [("We invite your valuable questions and suggestions.", 2000, False, (0xFF,0xFF,0xFF), "ctr", 0, 0, 100)], anchor="ctr")
    atb(ty, Inches(2.0), Inches(4.8), Inches(9.5), Inches(0.55),
        [("Project 27PR27 | Dept. of IT | Kongu Engineering College", 1500, False, (0xB5,0xA7,0x88), "ctr", 0, 0, 100)])
    prs.save(str(OUT))
    print(f"\nSaved: {OUT}  ({len(prs.slides)} slides)")

if __name__ == "__main__":
    main()
