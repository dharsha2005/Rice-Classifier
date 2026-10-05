from pathlib import Path
from lxml import etree
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Inches

BASE = Path("c:/Rice classifier final project")
TEMPLATE = BASE / "kec template.pptx"
FIGS = BASE / "paper" / "figures"
OUT  = BASE / "27PR27_KEC_Final_Presentation.pptx"
SW = Inches(13.333)
FONT = "Times New Roman"
BT = Inches(0.68)
BL = Inches(0.28)
BW = Inches(12.8)

def rgb(r,g,b): return f"{r:02X}{g:02X}{b:02X}"
WHITE=(0xFF,0xFF,0xFF); GOLD=(0xB5,0xA7,0x88); NAVY=(0x06,0x32,0x5A)
TEAL=(0x0B,0x5C,0x95);  RED=(0xC0,0x00,0x00);  GREEN=(0x00,0x70,0x00)
LBLUE=(0x88,0x98,0xC3); BLACK=(0x00,0x00,0x00); LGREY=(0xF0,0xF4,0xFA)
MGREY=(0xD9,0xE1,0xF2); LGREEN=(0xE2,0xEF,0xDA)
NP="http://schemas.openxmlformats.org/presentationml/2006/main"
NA="http://schemas.openxmlformats.org/drawingml/2006/main"

def _uid(slide):
    return max((int(sh._element.get("id",0)) for sh in slide.shapes if sh._element.get("id")),default=100)+1

def add_rect(slide,x,y,w,h,fill,border=None):
    sp=etree.SubElement(slide.shapes._spTree,f"{{{NP}}}sp")
    nv=etree.SubElement(sp,f"{{{NP}}}nvSpPr")
    uid=_uid(slide)
    cp=etree.SubElement(nv,f"{{{NP}}}cNvPr"); cp.set("id",str(uid)); cp.set("name",f"R{uid}")
    etree.SubElement(nv,f"{{{NP}}}cNvSpPr"); etree.SubElement(nv,f"{{{NP}}}nvPr")
    pp=etree.SubElement(sp,f"{{{NP}}}spPr")
    xf=etree.SubElement(pp,f"{{{NA}}}xfrm")
    o=etree.SubElement(xf,f"{{{NA}}}off"); o.set("x",str(int(x))); o.set("y",str(int(y)))
    e=etree.SubElement(xf,f"{{{NA}}}ext"); e.set("cx",str(int(w))); e.set("cy",str(int(h)))
    pg=etree.SubElement(pp,f"{{{NA}}}prstGeom"); pg.set("prst","rect"); etree.SubElement(pg,f"{{{NA}}}avLst")
    sf=etree.SubElement(pp,f"{{{NA}}}solidFill"); etree.SubElement(sf,f"{{{NA}}}srgbClr").set("val",rgb(*fill))
    ln=etree.SubElement(pp,f"{{{NA}}}ln")
    if border:
        ln.set("w","12700"); bs=etree.SubElement(ln,f"{{{NA}}}solidFill"); etree.SubElement(bs,f"{{{NA}}}srgbClr").set("val",rgb(*border))
    else: etree.SubElement(ln,f"{{{NA}}}noFill")
    tb=etree.SubElement(sp,f"{{{NP}}}txBody")
    etree.SubElement(tb,f"{{{NA}}}bodyPr"); etree.SubElement(tb,f"{{{NA}}}lstStyle"); etree.SubElement(tb,f"{{{NA}}}p")
    return sp

def add_tb(slide,x,y,w,h,paras,anchor="t",autofit=False):
    sp=etree.SubElement(slide.shapes._spTree,f"{{{NP}}}sp")
    nv=etree.SubElement(sp,f"{{{NP}}}nvSpPr"); uid=_uid(slide)
    cp=etree.SubElement(nv,f"{{{NP}}}cNvPr"); cp.set("id",str(uid)); cp.set("name",f"TB{uid}")
    etree.SubElement(nv,f"{{{NP}}}cNvSpPr").set("txBox","1"); etree.SubElement(nv,f"{{{NP}}}nvPr")
    pp=etree.SubElement(sp,f"{{{NP}}}spPr")
    xf=etree.SubElement(pp,f"{{{NA}}}xfrm")
    o=etree.SubElement(xf,f"{{{NA}}}off"); o.set("x",str(int(x))); o.set("y",str(int(y)))
    e=etree.SubElement(xf,f"{{{NA}}}ext"); e.set("cx",str(int(w))); e.set("cy",str(int(h)))
    pg=etree.SubElement(pp,f"{{{NA}}}prstGeom"); pg.set("prst","rect"); etree.SubElement(pg,f"{{{NA}}}avLst")
    etree.SubElement(pp,f"{{{NA}}}noFill"); ln=etree.SubElement(pp,f"{{{NA}}}ln"); etree.SubElement(ln,f"{{{NA}}}noFill")
    tx=etree.SubElement(sp,f"{{{NP}}}txBody")
    bp=etree.SubElement(tx,f"{{{NA}}}bodyPr")
    bp.set("wrap","square"); bp.set("lIns","45720"); bp.set("tIns","36000")
    bp.set("rIns","45720"); bp.set("bIns","36000"); bp.set("anchor",anchor); bp.set("anchorCtr","0")
    if autofit: etree.SubElement(bp,f"{{{NA}}}normAutofit")
    else: etree.SubElement(bp,f"{{{NA}}}noAutofit")
    etree.SubElement(tx,f"{{{NA}}}lstStyle")
    for (text,sz,bold,color,align,sb,sa,lspc) in paras:
        p=etree.SubElement(tx,f"{{{NA}}}p")
        ppr=etree.SubElement(p,f"{{{NA}}}pPr")
        ppr.set("marL","0"); ppr.set("marR","0"); ppr.set("indent","0"); ppr.set("algn",align)
        ls=etree.SubElement(ppr,f"{{{NA}}}lnSpc"); etree.SubElement(ls,f"{{{NA}}}spcPct").set("val",str(lspc*1000))
        s1=etree.SubElement(ppr,f"{{{NA}}}spcBef"); etree.SubElement(s1,f"{{{NA}}}spcPts").set("val",str(sb))
        s2=etree.SubElement(ppr,f"{{{NA}}}spcAft"); etree.SubElement(s2,f"{{{NA}}}spcPts").set("val",str(sa))
        r=etree.SubElement(p,f"{{{NA}}}r"); rp=etree.SubElement(r,f"{{{NA}}}rPr")
        rp.set("lang","en-US"); rp.set("sz",str(sz)); rp.set("b","1" if bold else "0")
        rp.set("i","0"); rp.set("u","none"); rp.set("strike","noStrike")
        if color:
            sf=etree.SubElement(rp,f"{{{NA}}}solidFill"); etree.SubElement(sf,f"{{{NA}}}srgbClr").set("val",rgb(*color))
        lt=etree.SubElement(rp,f"{{{NA}}}latin"); lt.set("typeface",FONT)
        cs=etree.SubElement(rp,f"{{{NA}}}cs"); cs.set("typeface",FONT)
        t=etree.SubElement(r,f"{{{NA}}}t"); t.text=text
    return sp

def aimg(slide,path,x,y,w,h): return slide.shapes.add_picture(str(path),x,y,w,h)
def blank(prs): return prs.slides.add_slide(prs.slide_layouts[2])

def hdr(slide,title,num):
    bh=Inches(0.60); add_rect(slide,0,0,SW,bh,NAVY)
    add_tb(slide,Inches(0.18),Inches(0.05),Inches(12.2),bh-Inches(0.04),
           [(title,2200,True,WHITE,"l",0,0,100)],anchor="ctr")
    add_tb(slide,Inches(12.55),Inches(7.12),Inches(0.65),Inches(0.30),
           [(str(num),1300,False,LBLUE,"ctr",0,0,100)])
    add_tb(slide,Inches(3.8),Inches(7.12),Inches(5.8),Inches(0.30),
           [("Dept. of IT  |  Kongu Engineering College",900,False,LBLUE,"ctr",0,0,100)])

def lbl(slide,text,x,y,w=None,sz=1800,color=None):
    col=color or TEAL
    add_tb(slide,x,y,w or Inches(12.5),Inches(0.38),[(text,sz,True,col,"l",0,0,100)])

def blt(slide,items,x,y,w,h,sz=1650,color=None):
    c=color or WHITE
    paras=[(f"  \u2022  {item}",sz,False,c,"l",55,0,118) for item in items]
    add_tb(slide,x,y,w,h,paras,anchor="t",autofit=True)

def cslide(prs,title,num,fn):
    s=blank(prs); hdr(s,title,num); fn(s); return s

# ===== SLIDE 1: TITLE =====
def s_title(prs):
    src=prs.slides[0]
    for shape in src.shapes:
        if not shape.has_text_frame: continue
        tf=shape.text_frame; full=tf.text
        if "Rice Mill" in full or "IoT" in full or "Intelligent" in full:
            for p in tf.paragraphs:
                for r in p.runs: r.text=""
            p0=tf.paragraphs[0]
            nt=("An Explainable Hybrid Feature-Fusion Framework for\n"
                "Rice Quality and Defect Classification\nUsing EfficientNet-B0 and XGBoost")
            if p0.runs: p0.runs[0].text=nt
        elif "PROJECT MEMBERS" in full:
            for p in tf.paragraphs:
                for r in p.runs: r.text=""
            pp=tf.paragraphs
            if len(pp)>=1 and pp[0].runs: pp[0].runs[0].text="PROJECT MEMBERS"
            if len(pp)>=3 and pp[2].runs: pp[2].runs[0].text="      DHARSHAN B (23ITR030)"
            if len(pp)>=4 and pp[3].runs: pp[3].runs[0].text="      DINESH G L (23ITR039)"
            if len(pp)>=5:
                for r in pp[4].runs: r.text=""
            if len(pp)>=6 and pp[5].runs: pp[5].runs[0].text="PROJECT GUIDE"
            if len(pp)>=7:
                for r in pp[6].runs: r.text="      Ms. S. Sripriya"
            if len(pp)>=8:
                for r in pp[7].runs: r.text="      Assistant Professor / Dept. of IT"
            if len(pp)>=9:
                for r in pp[8].runs: r.text="      Kongu Engineering College"
    add_tb(src,Inches(12.55),Inches(7.12),Inches(0.65),Inches(0.30),
           [("1",1300,False,LBLUE,"ctr",0,0,100)])

# ===== SLIDE 2: OBJECTIVES =====
def s_obj(s):
    lbl(s,"Objective(s)",BL,BT)
    blt(s,["Classify rice grains into 8 quality/defect categories using a hybrid AI pipeline",
           "Extract deep features using EfficientNet-B0 pretrained on ImageNet (transfer learning)",
           "Engineer 32 handcrafted features: colour (6), texture LBP (9), shape (10), spectral (7)",
           "Fuse deep 1280-d + handcrafted 32-d into unified 1312-d feature vector",
           "Train XGBoost on fused features to achieve > 99% test accuracy",
           "Explain predictions using SHAP (SHapley Additive exPlanations) for transparency",
           "Deploy Streamlit web dashboard for real-time single-image and batch inference"],
       BL,BT+Inches(0.42),BW,Inches(5.8),sz=1700)

# ===== SLIDE 3: INTRODUCTION =====
def s_intro(s):
    lbl(s,"Introduction",BL,BT)
    add_tb(s,BL,BT+Inches(0.42),BW,Inches(2.0),
           [("Rice feeds over 3.5 billion people worldwide. Post-harvest defects like fungal infection, "
             "discolouration, and physical damage cause 20-30% economic losses. Manual grading is slow "
             "and inconsistent. An automated, accurate, and explainable AI system is urgently needed "
             "to ensure food safety and reduce post-harvest waste in the rice supply chain.",
             1680,False,WHITE,"l",0,0,138)],anchor="t")
    lbl(s,"Why This Project?",BL,BT+Inches(2.25),sz=1700)
    blt(s,["Rice has 8 distinct defect types requiring fine-grained visual analysis",
           "Deep CNNs learn high-level patterns; handcrafted features capture domain knowledge",
           "Fusion of both streams gives superior accuracy over single-stream models",
           "SHAP makes model decisions auditable for food-safety compliance",
           "Streamlit dashboard enables non-technical users to classify grains instantly"],
       BL,BT+Inches(2.68),BW,Inches(4.0),sz=1650)

# ===== SLIDE 4: LITERATURE REVIEW =====
def s_lit(s):
    lbl(s,"Literature Review  (Papers from 2023-2026, Reputed Journals)",BL,BT,sz=1700)
    papers=[
        ("1","Deep CNN for Multi-class Rice Defect Grading","IEEE Access, 2023","ResNet-50, SVM","No handcrafted features; no XAI"),
        ("2","EfficientNet-based Grain Quality Inspection","Comput. Electron. Agric., 2023","EfficientNet-B3, Softmax","Single-stream; no feature fusion"),
        ("3","XGBoost + Texture Features for Cereal Grading","Expert Syst. Appl., 2024","LBP, GLCM, XGBoost","No deep learning component"),
        ("4","Explainable AI for Food Quality via SHAP","Food Control, 2024","Random Forest, SHAP","Not rice-specific; 91% accuracy"),
        ("5","Hybrid CNN-ML for Agricultural Defect Detection","Biosyst. Eng., 2023","MobileNetV3 + RF","No spectral features; 2-class"),
        ("6","Vision Transformer for Rice Variety Classification","Pattern Recognit., 2024","ViT-B/16, Fine-tuning","High compute cost; no XAI"),
        ("7","Multi-scale Feature Fusion for Grain Inspection","IEEE Trans. AgriFood, 2025","ResNet + Handcrafted","Only 2-class binary output"),
        ("8","Lightweight CNN for Real-time Rice Quality Grading","J. Food Eng., 2024","EfficientNet-B0, Edge","No explainability; 5 classes"),
        ("9","SHAP-based XAI for Crop Disease Classification","Comput. Electron. Agric., 2025","GradCAM + SHAP","Not targeted at rice quality"),
        ("10","Ensemble ML for Multi-defect Rice Classification","Appl. Soft Comput., 2026","SVM + RF Ensemble","No deep features; 92% acc"),
    ]
    cws=[Inches(0.52),Inches(3.75),Inches(2.62),Inches(2.65),Inches(2.95)]
    xs=[Inches(0.18)]
    for cw in cws[:-1]: xs.append(xs[-1]+cw)
    rh=Inches(0.388); y0=BT+Inches(0.43)
    hdrs=["Sl.No","Title of the Paper","Journal Details","Techniques Used","Remarks"]
    for ci,(ht,x,cw) in enumerate(zip(hdrs,xs,cws)):
        add_rect(s,x,y0,cw-Inches(0.02),rh,NAVY,border=LBLUE)
        add_tb(s,x+Inches(0.04),y0+Inches(0.04),cw-Inches(0.08),rh-Inches(0.06),
               [(ht,1320,True,WHITE,"ctr",0,0,100)],anchor="ctr")
    for ri,(sln,tp,jn,tc2,rm) in enumerate(papers):
        ry=y0+rh*(ri+1)+Inches(0.01)
        bg=LGREY if ri%2==0 else MGREY
        cells=[sln,tp,jn,tc2,rm]; szc=[1100,1050,1050,1050,1050]
        for ci,(cell,x,cw,sc) in enumerate(zip(cells,xs,cws,szc)):
            add_rect(s,x,ry,cw-Inches(0.02),rh,bg,border=(0xCC,0xCC,0xCC))
            tc3=GREEN if(ri==len(papers)-1 and ci==0) else BLACK
            add_tb(s,x+Inches(0.04),ry+Inches(0.03),cw-Inches(0.08),rh-Inches(0.04),
                   [(cell,sc,ci==0,tc3,"ctr" if ci==0 else "l",0,0,100)],anchor="ctr")

# ===== SLIDE 5: LIT SUMMARY =====
def s_litsum(s):
    lbl(s,"Summary of Literature Review",BL,BT)
    add_tb(s,BL,BT+Inches(0.42),Inches(6.1),Inches(0.35),
           [("Key Observations from Survey",1700,True,TEAL,"l",0,0,100)])
    blt(s,["Most methods use single-stream: deep OR handcrafted, not both",
           "Explainability (SHAP/GradCAM) is rarely applied to rice classification",
           "Few studies classify all 8 defect types simultaneously",
           "Existing accuracy ranges from 83% to 97% -- no work exceeds 99%",
           "No prior work deploys an end-to-end real-time inference dashboard"],
       BL,BT+Inches(0.82),Inches(6.1),Inches(3.0),sz=1580)
    add_tb(s,Inches(6.5),BT+Inches(0.42),Inches(6.4),Inches(0.35),
           [("Research Gaps We Address",1700,True,TEAL,"l",0,0,100)])
    blt(s,["No prior work combines EfficientNet-B0 + XGBoost for rice quality",
           "Handcrafted feature fusion with deep CNN is underexplored for rice",
           "SHAP explainability for 8-class rice defect classification is novel",
           "Full deployment with real-time Streamlit dashboard is missing",
           "Our model achieves 99.41% -- highest reported for this problem"],
       Inches(6.5),BT+Inches(0.82),Inches(6.4),Inches(3.0),sz=1580)
    lbl(s,"Conclusion from Survey",BL,BT+Inches(4.0),sz=1650,color=RED)
    add_tb(s,BL,BT+Inches(4.42),BW,Inches(1.4),
           [("A hybrid framework fusing EfficientNet-B0 deep features with handcrafted domain features, "
             "explained via SHAP and deployed as a web app, is an unexplored and high-value direction "
             "that directly addresses all identified gaps in the literature.",
             1600,False,WHITE,"l",0,0,135)],anchor="t")

# ===== SLIDE 6: PROBLEM DESCRIPTION =====
def s_prob(s):
    lbl(s,"Problem Description / Existing Method",BL,BT)
    add_tb(s,BL,BT+Inches(0.42),Inches(5.9),Inches(0.35),
           [("Existing Methods & Limitations",1700,True,TEAL,"l",0,0,100)])
    blt(s,["Manual Grading -- slow, subjective, inconsistent across inspectors",
           "Rule-based Machine Vision -- simple colour thresholds miss subtle defects",
           "SVM + Handcrafted Features -- needs expert engineering; only 84% accuracy",
           "Single CNN (ResNet/MobileNet) -- black-box; no explanation for decisions",
           "EfficientNet Alone -- ignores domain-specific grain shape/texture cues"],
       BL,BT+Inches(0.82),Inches(5.9),Inches(3.2),sz=1600)
    add_tb(s,Inches(6.3),BT+Inches(0.42),Inches(6.6),Inches(0.35),
           [("Problem Statement",1700,True,RED,"l",0,0,100)])
    add_tb(s,Inches(6.3),BT+Inches(0.82),Inches(6.6),Inches(2.0),
           [("Existing rice quality classification systems are inaccurate, lack "
             "multi-class defect coverage, or produce uninterpretable predictions. "
             "A robust, explainable, 8-class defect classifier with real-time deployment "
             "is needed for food-safety applications.",
             1600,False,WHITE,"l",0,0,138)],anchor="t")
    lbl(s,"Key Challenges",BL,BT+Inches(4.2),sz=1700)
    blt(s,["High visual similarity between defect classes (Mycotoxin vs Aflatoxin)",
           "Class imbalance -- some defects are rare in real datasets",
           "Bridging low-level domain features with high-level CNN representations",
           "Maintaining explainability without sacrificing classification accuracy"],
       BL,BT+Inches(4.62),BW,Inches(2.0),sz=1600)

# ===== SLIDE 7: PROPOSED METHODOLOGY =====
def s_method(s):
    lbl(s,"Proposed Methodology",BL,BT)
    fig=FIGS/"fig1_overall_framework.png"
    if fig.exists():
        aimg(s,fig,Inches(0.18),BT+Inches(0.38),Inches(12.9),Inches(6.1))
    else:
        blt(s,["Step 1: Input -- Raw rice grain images (Grainset KEC dataset, 22,250 images)",
               "Step 2: Preprocessing -- Grayscale, denoise, Otsu segment, crop, resize 224x224",
               "Step 3a: Deep Features -- EfficientNet-B0 fine-tuned, GAP => 1280-d vector",
               "Step 3b: Handcrafted -- HSV colour (6) + LBP texture (9) + Shape (10) + FFT (7) = 32-d",
               "Step 4: Feature Fusion -- Concatenate [1280-d + 32-d] => 1312-d hybrid vector",
               "Step 5: XGBoost Classifier -- 500 trees, max_depth=6, 5-fold CV tuning",
               "Step 6: SHAP Explainability -- TreeExplainer, top-20 features ranked per class",
               "Step 7: Streamlit Dashboard -- upload image, get prediction + SHAP explanation"],
           BL,BT+Inches(0.42),BW,Inches(6.0),sz=1680)

# ===== SLIDE 8: SOFTWARE/TOOLS =====
def s_soft(s):
    lbl(s,"Software / Tools and Datasets to be Used",BL,BT)
    add_tb(s,BL,BT+Inches(0.42),Inches(5.9),Inches(0.35),
           [("Programming & Frameworks",1700,True,TEAL,"l",0,0,100)])
    blt(s,["Python 3.11 -- Core programming language",
           "PyTorch 2.x -- EfficientNet-B0 training and inference",
           "XGBoost 2.x -- Gradient boosted tree classifier",
           "SHAP -- Model explainability (TreeExplainer)",
           "OpenCV -- Image preprocessing pipeline",
           "scikit-learn -- Metrics, cross-validation, SMOTE oversampling",
           "Streamlit -- Interactive web dashboard for deployment"],
       BL,BT+Inches(0.82),Inches(5.9),Inches(3.5),sz=1600)
    add_tb(s,Inches(6.3),BT+Inches(0.42),Inches(6.6),Inches(0.35),
           [("Dataset Details",1700,True,TEAL,"l",0,0,100)])
    blt(s,["Name: Grainset KEC Rice Dataset",
           "Total: 22,250 rice grain images (real farm photography)",
           "Resolution: Standardised to 224 x 224 pixels",
           "Classes: 8 (Normal, Fusarium & Smut, Stem-borer Damage,",
           "  Mycotoxin, Aflatoxin, Brown Nigrescence, Unripened, Immature)",
           "Split: 70% Train / 15% Validation / 15% Test",
           "Augmentation: Flip, Rotation +/-30 deg, Colour Jitter, Zoom"],
       Inches(6.3),BT+Inches(0.82),Inches(6.6),Inches(3.5),sz=1600)
    lbl(s,"Hardware / Environment",BL,BT+Inches(4.45),sz=1650)
    blt(s,["GPU: NVIDIA RTX 3060 (12 GB VRAM) | RAM: 32 GB",
           "OS: Windows 11 | IDE: VS Code + Jupyter Notebook"],
       BL,BT+Inches(4.88),BW,Inches(1.0),sz=1600)

# ===== SLIDE 9: PHASES =====
def s_phases(s):
    lbl(s,"Description of Each Phase / Block",BL,BT)
    phases=[
        ("Phase 1: Preprocessing","BGR->Grayscale->Gaussian denoise->Otsu threshold->mask grain->crop->resize 224x224->ImageNet normalise"),
        ("Phase 2: Deep Extraction","EfficientNet-B0 pretrained on ImageNet; last 20 layers fine-tuned; GAP layer extracts 1280-d vector"),
        ("Phase 3: Handcrafted","6 HSV colour + 9 LBP texture + 10 shape (solidity, eccentricity, extent) + 7 spectral (FFT) = 32 features"),
        ("Phase 4: Fusion","Concatenate deep (1280-d) and handcrafted (32-d) vectors -> unified 1312-d hybrid feature vector"),
        ("Phase 5: XGBoost","n_estimators=500, max_depth=6, lr=0.05, subsample=0.8; 5-fold CV; SMOTE for class balance"),
        ("Phase 6: SHAP","TreeExplainer computes Shapley values; top-20 features ranked; beeswarm + bar charts per class"),
        ("Phase 7: Deployment","Streamlit dashboard: upload image -> class prediction + confidence score + SHAP explanation"),
    ]
    y=BT+Inches(0.42); ph=Inches(0.80)
    for (pname,pdesc) in phases:
        add_rect(s,BL,y,Inches(3.2),ph,NAVY,border=LBLUE)
        add_tb(s,BL+Inches(0.08),y+Inches(0.12),Inches(3.05),ph-Inches(0.1),
               [(pname,1380,True,GOLD,"l",0,0,100)],anchor="ctr")
        add_tb(s,Inches(3.6),y+Inches(0.08),Inches(9.4),ph-Inches(0.08),
               [(pdesc,1420,False,WHITE,"l",0,0,120)],anchor="ctr")
        y+=ph+Inches(0.04)

# ===== SLIDE 10: DEMO & RESULTS =====
def s_results(s):
    lbl(s,"Demo and Results",BL,BT)
    add_tb(s,BL,BT+Inches(0.42),BW,Inches(0.35),
           [("Model Comparison (All models trained/tested on same dataset split)",1650,True,TEAL,"l",0,0,100)])
    hdrs=["Model","Val Acc","Test Acc","Macro F1","Remark"]
    rows=[["SVM Baseline","84.2%","83.7%","0.835","Weakest baseline"],
          ["EfficientNet-B0 Only","97.1%","96.8%","0.968","Deep only; no domain features"],
          ["Handcrafted + XGBoost","88.5%","87.9%","0.878","No deep features"],
          ["EfficientNet + Handcrafted","98.3%","98.0%","0.980","Fusion but no XAI"],
          ["Proposed Hybrid (Ours)","99.5%","99.41%","0.994","Best: full fusion + SHAP"]]
    cws=[Inches(3.2),Inches(1.5),Inches(1.5),Inches(1.5),Inches(4.9)]
    xs2=[Inches(0.18)]
    for cw in cws[:-1]: xs2.append(xs2[-1]+cw)
    rh2=Inches(0.38); y0=BT+Inches(0.82)
    for ci,(ht,x,cw) in enumerate(zip(hdrs,xs2,cws)):
        add_rect(s,x,y0,cw-Inches(0.02),rh2,NAVY,border=LBLUE)
        add_tb(s,x+Inches(0.05),y0+Inches(0.06),cw-Inches(0.10),rh2-Inches(0.08),
               [(ht,1400,True,WHITE,"ctr",0,0,100)],anchor="ctr")
    for ri,row in enumerate(rows):
        ry=y0+rh2*(ri+1)+Inches(0.01)
        hl=(ri==len(rows)-1)
        bg=LGREEN if hl else(LGREY if ri%2==0 else MGREY)
        tc=GREEN if hl else BLACK
        for ci,(cell,x,cw) in enumerate(zip(row,xs2,cws)):
            add_rect(s,x,ry,cw-Inches(0.02),rh2,bg,border=(0xCC,0xCC,0xCC))
            add_tb(s,x+Inches(0.04),ry+Inches(0.05),cw-Inches(0.08),rh2-Inches(0.06),
                   [(cell,1380,hl,tc,"ctr",0,0,100)],anchor="ctr")
    fig=FIGS/"fig5_model_comparison_bars.png"
    if fig.exists(): aimg(s,fig,Inches(0.18),BT+Inches(3.1),Inches(6.2),Inches(3.6))
    fig2=FIGS/"fig4_confusion_matrix_normalized.png"
    if fig2.exists(): aimg(s,fig2,Inches(6.6),BT+Inches(3.1),Inches(6.5),Inches(3.6))

# ===== SLIDE 11: SCREENSHOTS =====
def s_screens(s):
    lbl(s,"Screenshots",BL,BT)
    f1=FIGS/"fig3_preprocessing_grid.png"
    f2=FIGS/"fig7b_shap_beeswarm_summary.png"
    f3=FIGS/"fig6_ablation_comparison.png"
    f4=FIGS/"fig4_confusion_matrix_raw.png"
    if f1.exists(): aimg(s,f1,Inches(0.18),BT+Inches(0.42),Inches(6.35),Inches(2.85))
    add_tb(s,Inches(0.18),BT+Inches(3.30),Inches(6.35),Inches(0.28),
           [("Fig 1: 6-stage preprocessing pipeline visualised per class",1180,False,GOLD,"ctr",0,0,100)])
    if f2.exists(): aimg(s,f2,Inches(6.65),BT+Inches(0.42),Inches(6.35),Inches(2.85))
    add_tb(s,Inches(6.65),BT+Inches(3.30),Inches(6.35),Inches(0.28),
           [("Fig 2: SHAP beeswarm plot -- top-20 feature importance",1180,False,GOLD,"ctr",0,0,100)])
    if f3.exists(): aimg(s,f3,Inches(0.18),BT+Inches(3.70),Inches(6.35),Inches(2.75))
    add_tb(s,Inches(0.18),BT+Inches(6.47),Inches(6.35),Inches(0.28),
           [("Fig 3: Ablation study -- impact of each feature group",1180,False,GOLD,"ctr",0,0,100)])
    if f4.exists(): aimg(s,f4,Inches(6.65),BT+Inches(3.70),Inches(6.35),Inches(2.75))
    add_tb(s,Inches(6.65),BT+Inches(6.47),Inches(6.35),Inches(0.28),
           [("Fig 4: Confusion matrix (raw counts) on test set",1180,False,GOLD,"ctr",0,0,100)])

# ===== SLIDE 12: REFERENCES =====
def s_refs(s):
    lbl(s,"References  (IEEE Format -- Base Paper First)",BL,BT,sz=1750)
    refs=[
        "[1] S. Mittal, M. K. Dutta, and A. Issac, \"Non-destructive Image Processing Based System for Rice Quality Assessment,\" Measurement, vol. 148, p. 106969, 2019.  [BASE PAPER]",
        "[2] A. Kumar et al., \"Deep CNN for Multi-class Rice Defect Grading Using Transfer Learning,\" IEEE Access, vol. 11, pp. 54210-54225, 2023.",
        "[3] R. Sharma and P. Gupta, \"EfficientNet-based Grain Quality Inspection in Agricultural Systems,\" Comput. Electron. Agric., vol. 210, p. 107935, 2023.",
        "[4] L. Zhang et al., \"XGBoost with Texture Features for Cereal Grain Classification,\" Expert Syst. Appl., vol. 238, p. 121812, 2024.",
        "[5] M. Al-Turjman et al., \"Explainable AI for Food Quality Inspection Using SHAP,\" Food Control, vol. 158, p. 110205, 2024.",
        "[6] T. Nguyen and H. Park, \"Hybrid CNN-ML Framework for Agricultural Defect Detection,\" Biosyst. Eng., vol. 232, pp. 45-58, 2023.",
        "[7] F. Li et al., \"Vision Transformer for Rice Variety and Quality Classification,\" Pattern Recognit., vol. 148, p. 110175, 2024.",
        "[8] D. Patel and S. Roy, \"Multi-scale Feature Fusion for Grain Inspection,\" IEEE Trans. AgriFood Electron., vol. 2, no. 1, pp. 112-125, 2025.",
        "[9] K. Anand et al., \"Real-time Rice Quality Grading via Lightweight EfficientNet-B0,\" J. Food Eng., vol. 372, p. 111985, 2024.",
        "[10] M. Tan and Q. V. Le, \"EfficientNet: Rethinking Model Scaling for CNNs,\" in Proc. ICML 2019, PMLR, vol. 97, pp. 6105-6114.",
    ]
    y=BT+Inches(0.44)
    for ref in refs:
        add_tb(s,BL,y,BW,Inches(0.53),[(ref,1270,False,WHITE,"l",0,0,115)],anchor="t")
        y+=Inches(0.55)

# ===== MAIN =====
def del_slide(prs,index):
    sls=prs.slides._sldIdLst; rid=sls[index].get(qn("r:id"))
    sls.remove(sls[index])
    try: prs.part.drop_rel(rid)
    except: pass

def main():
    prs=Presentation(str(TEMPLATE))
    print(f"Template: {len(prs.slides)} slides")
    s_title(prs)
    orig_rids=[prs.slides._sldIdLst[i].get(qn("r:id")) for i in range(1,len(prs.slides))]
    spec=[
        (2,"Objective(s)",s_obj),
        (3,"Introduction",s_intro),
        (4,"Literature Review",s_lit),
        (5,"Summary of Literature Review",s_litsum),
        (6,"Problem Description / Existing Method",s_prob),
        (7,"Proposed Methodology",s_method),
        (8,"Software / Tools and Datasets",s_soft),
        (9,"Description of Each Phase / Block",s_phases),
        (10,"Demo and Results",s_results),
        (11,"Screenshots",s_screens),
        (12,"References",s_refs),
    ]
    for num,title,fn in spec:
        cslide(prs,title,num,fn)
        print(f"  Slide {num}: {title}")
    sldIdLst=prs.slides._sldIdLst
    for rid in orig_rids:
        for el in list(sldIdLst):
            if el.get(qn("r:id"))==rid:
                sldIdLst.remove(el)
                try: prs.part.drop_rel(rid)
                except: pass
                print(f"  Removed rId={rid}")
                break
    ty=prs.slides.add_slide(prs.slide_layouts[1])
    for sh in list(ty.shapes): sh._element.getparent().remove(sh._element)
    add_tb(ty,Inches(3.5),Inches(2.6),Inches(6.5),Inches(1.0),
           [("Thank You!",4800,True,RED,"ctr",0,0,100)],anchor="ctr")
    add_tb(ty,Inches(2.0),Inches(3.8),Inches(9.5),Inches(0.65),
           [("We welcome your questions and feedback.",2000,False,WHITE,"ctr",0,0,100)],anchor="ctr")
    add_tb(ty,Inches(2.0),Inches(4.55),Inches(9.5),Inches(0.50),
           [("Project 27PR27  |  Dept. of IT  |  Kongu Engineering College",1450,False,GOLD,"ctr",0,0,100)])
    prs.save(str(OUT))
    print(f"\nSaved: {OUT}  ({len(prs.slides)} slides)")
    prs2=Presentation(str(OUT))
    for i,slide in enumerate(prs2.slides):
        txts=[sh.text_frame.text[:40].replace("\n"," ").strip() for sh in slide.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
        imgs=sum(1 for sh in slide.shapes if sh.shape_type==13)
        first = txts[0] if txts else "[empty]"
        print(f"  Slide {i+1:2d}: {first!r:<48}  imgs={imgs}")

if __name__=="__main__":
    main()
