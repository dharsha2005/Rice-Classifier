"""
generate_complete_project_report.py
===================================
Generates the complete B.Tech final-year academic project report:
"27PR27_Rice_Quality_Defect_Classification_Project_Report.docx"
for Kongu Engineering College (Autonomous), Department of Information Technology.

Strictly follows the structure, organization, formatting, and presentation
of the reference report (7TH SEM.pdf), tailored 100% to Project 27PR27:
"An Explainable Hybrid Feature-Fusion Framework for Rice Quality and Defect
Classification Using EfficientNet-B0 and XGBoost".
"""

import os
import sys
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = Path("c:/Rice classifier final project")
OUTPUT_DOCX = PROJECT_ROOT / "27PR27_Rice_Quality_Defect_Classification_Project_Report.docx"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
SAMPLE_DIR = PROJECT_ROOT / "results" / "preprocessing" / "intermediate_samples" / "sample_01_class_0"

def create_report():
    doc = docx.Document()
    
    # Page setup - A4, standard binding margins (Left: 1.25", Right: 1.0", Top: 1.0", Bottom: 1.0")
    for sec in doc.sections:
        sec.page_width = Inches(8.27)   # A4 width
        sec.page_height = Inches(11.69) # A4 height
        sec.top_margin = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin = Inches(1.25)  # Binding margin
        sec.right_margin = Inches(1.0)
        
    print("Document initialized with A4 paper and binding margins.")
    return doc

if __name__ == "__main__":
    doc = create_report()
    doc.save(OUTPUT_DOCX)
    print(f"Saved initial template to {OUTPUT_DOCX}")
