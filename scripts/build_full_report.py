"""
build_full_report.py
====================
Master build script to generate the complete B.Tech final-year academic project report:
"27PR27_Rice_Quality_Defect_Classification_Project_Report.docx"
for Kongu Engineering College (Autonomous), Department of Information Technology.
"""

import sys
from pathlib import Path
import docx
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Ensure repo root is on sys.path
PROJECT_ROOT = Path("c:/Rice classifier final project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.report_builder.styles import (
    set_page_setup, add_page_number_to_footer,
    FONT_NAME, COLOR_BLACK
)
from scripts.report_builder.front_matter import (
    add_cover_page, add_bonafide_certificate, add_declaration,
    add_abstract_section, add_acknowledgement_section,
    add_table_of_contents, add_list_of_abbreviations,
    add_list_of_figures, add_list_of_tables
)
from scripts.report_builder.chapters_1_to_4 import (
    build_chapter_1, build_chapter_2, build_chapter_3, build_chapter_4
)
from scripts.report_builder.chapter_5 import build_chapter_5
from scripts.report_builder.chapter_6 import build_chapter_6
from scripts.report_builder.chapters_7_8_back import (
    build_chapter_7, build_chapter_8, build_references, build_genai_disclosure
)

OUTPUT_FILE = PROJECT_ROOT / "27PR27_Rice_Quality_Defect_Classification_Project_Report.docx"

def generate_report():
    print("=" * 70)
    print("GENERATING COMPLETE B.TECH PROJECT REPORT: 27PR27")
    print("=" * 70)
    
    doc = docx.Document()
    
    # -------------------------------------------------------------
    # SECTION 1: COVER PAGE (No headers/footers)
    # -------------------------------------------------------------
    sec_cover = doc.sections[0]
    set_page_setup(sec_cover)
    print("Adding Cover Page...")
    add_cover_page(doc)
    
    # -------------------------------------------------------------
    # SECTION 2: FRONT MATTER (Roman numerals iv, v, vi...)
    # -------------------------------------------------------------
    print("Creating Section 2 (Front Matter)...")
    sec_front = doc.add_section(docx.enum.section.WD_SECTION.NEW_PAGE)
    set_page_setup(sec_front)
    # Unlink headers/footers from section 1
    sec_front.header.is_linked_to_previous = False
    sec_front.footer.is_linked_to_previous = False
    add_page_number_to_footer(sec_front, fmt="lowerRoman", start_num=2, align=WD_ALIGN_PARAGRAPH.RIGHT)
    
    print("Adding Bonafide Certificate...")
    add_bonafide_certificate(doc)
    
    print("Adding Declaration...")
    add_declaration(doc)
    
    print("Adding Abstract (19 points)...")
    add_abstract_section(doc)
    
    print("Adding Acknowledgement...")
    add_acknowledgement_section(doc)
    
    print("Adding Table of Contents...")
    add_table_of_contents(doc)
    
    print("Adding List of Abbreviations...")
    add_list_of_abbreviations(doc)
    
    print("Adding List of Figures...")
    add_list_of_figures(doc)
    
    print("Adding List of Tables...")
    add_list_of_tables(doc)
    
    # -------------------------------------------------------------
    # SECTION 3: MAIN REPORT BODY (Arabic numerals 1, 2, 3...)
    # -------------------------------------------------------------
    print("Creating Section 3 (Main Report Body)...")
    sec_body = doc.add_section(docx.enum.section.WD_SECTION.NEW_PAGE)
    set_page_setup(sec_body)
    sec_body.header.is_linked_to_previous = False
    sec_body.footer.is_linked_to_previous = False
    add_page_number_to_footer(sec_body, fmt="decimal", start_num=1, align=WD_ALIGN_PARAGRAPH.RIGHT)
    
    print("Building Chapter 1: Introduction...")
    build_chapter_1(doc)
    
    print("Building Chapter 2: Literature Review & Table...")
    build_chapter_2(doc)
    
    print("Building Chapter 3: Problem Definition...")
    build_chapter_3(doc)
    
    print("Building Chapter 4: System Requirements & Software Descriptions...")
    build_chapter_4(doc)
    
    print("Building Chapter 5: System Implementation (Dual Branch, Preprocessing, Fusion)...")
    build_chapter_5(doc)
    
    print("Building Chapter 6: Results and Discussion (Benchmarks, Ablation, SHAP)...")
    build_chapter_6(doc)
    
    print("Building Chapter 7: Conclusion and Future Work...")
    build_chapter_7(doc)
    
    print("Building Chapter 8: Appendices (Coding 8.1 & Output 8.2)...")
    build_chapter_8(doc)
    
    print("Building References (IEEE Format)...")
    build_references(doc)
    
    print("Building GenAI Disclosure...")
    build_genai_disclosure(doc)
    
    print("Saving complete document...")
    doc.save(str(OUTPUT_FILE))
    print(f"SUCCESS: Report saved to {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size:,} bytes")
    print("=" * 70)

if __name__ == "__main__":
    generate_report()
