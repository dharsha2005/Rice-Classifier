"""
styles.py
=========
Typography, styling, table formatting, and XML helpers for the academic project report.
Follows Kongu Engineering College (Autonomous) / Anna University B.Tech thesis format:
- Times New Roman throughout
- 1.5 line spacing for body paragraphs, justified text
- 1.25" left margin (for binding), 1.0" top, bottom, and right margins
- Professional tables with header shading and alternating rows
- Clean code listings in monospaced Consolas/Courier New
- Formal captions for figures and tables
"""

from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

FONT_NAME = "Times New Roman"
FONT_CODE = "Consolas"

COLOR_BLACK = RGBColor(0, 0, 0)
COLOR_NAVY = RGBColor(26, 54, 93)       # Formal academic navy
COLOR_DARK_GRAY = RGBColor(50, 50, 50)  # Charcoal for body text
COLOR_MUTED = RGBColor(110, 110, 110)
COLOR_BORDER = "CCCCCC"
HEX_HEADER_BG = "1F4E79"                # Elegant deep navy for table headers
HEX_ALT_ROW = "F7F9FB"                  # Subtle row shading
HEX_CALLOUT_BG = "F0F4F8"

def set_page_setup(section, is_first_page=False):
    section.page_width = Inches(8.27)    # A4 width
    section.page_height = Inches(11.69)  # A4 height
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.25)   # 1.25" for binding
    section.right_margin = Inches(1.0)
    section.header_distance = Inches(0.5)
    section.footer_distance = Inches(0.5)

def set_cell_shading(cell, color_hex):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def add_page_number_to_footer(section, fmt="decimal", start_num=None, align=WD_ALIGN_PARAGRAPH.RIGHT):
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = align
    
    # Configure page numbering format
    sectPr = section._sectPr
    # Remove existing pgNumType if any
    for child in list(sectPr):
        if child.tag.endswith('pgNumType'):
            sectPr.remove(child)
            
    pgNumType = OxmlElement('w:pgNumType')
    pgNumType.set(qn('w:fmt'), fmt)
    if start_num is not None:
        pgNumType.set(qn('w:start'), str(start_num))
    sectPr.append(pgNumType)
    
    # Add PAGE field
    run = p.add_run()
    run.font.name = FONT_NAME
    run.font.size = Pt(10)
    run.font.color.rgb = COLOR_DARK_GRAY
    r = run._r
    
    fld1 = OxmlElement('w:fldChar')
    fld1.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText')
    instr.set(qn('xml:space'), 'preserve')
    instr.text = " PAGE "
    fld2 = OxmlElement('w:fldChar')
    fld2.set(qn('w:fldCharType'), 'separate')
    fld3 = OxmlElement('w:fldChar')
    fld3.set(qn('w:fldCharType'), 'end')
    
    r.append(fld1)
    r.append(instr)
    r.append(fld2)
    r.append(fld3)

def add_chapter_heading(doc, chapter_num_str, title_str):
    """
    Format:
    CHAPTER 1
    
    INTRODUCTION
    """
    doc.add_page_break()
    p1 = doc.add_paragraph()
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p1.paragraph_format.space_before = Pt(36)
    p1.paragraph_format.space_after = Pt(12)
    p1.paragraph_format.line_spacing = 1.0
    r1 = p1.add_run(chapter_num_str.upper())
    r1.font.name = FONT_NAME
    r1.font.size = Pt(16)
    r1.font.bold = True
    r1.font.color.rgb = COLOR_BLACK

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_before = Pt(0)
    p2.paragraph_format.space_after = Pt(24)
    p2.paragraph_format.line_spacing = 1.15
    r2 = p2.add_run(title_str.upper())
    r2.font.name = FONT_NAME
    r2.font.size = Pt(16)
    r2.font.bold = True
    r2.font.color.rgb = COLOR_BLACK
    return p2

def add_prelim_heading(doc, title_str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(24)
    p.paragraph_format.space_after = Pt(20)
    r = p.add_run(title_str.upper())
    r.font.name = FONT_NAME
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = COLOR_BLACK
    return p

def add_section_heading(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = FONT_NAME
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = COLOR_BLACK
    return p

def add_subsection_heading(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = FONT_NAME
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = COLOR_BLACK
    return p

def add_subsubsection_heading(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = FONT_NAME
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.italic = True
    r.font.color.rgb = COLOR_BLACK
    return p

def add_body_p(doc, text, bold_prefix=None, space_after=6, line_spacing=1.5, indent=0.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if indent > 0:
        p.paragraph_format.first_line_indent = Inches(indent)
        
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.font.name = FONT_NAME
        rb.font.size = Pt(12)
        rb.font.bold = True
        rb.font.color.rgb = COLOR_BLACK
        
    r = p.add_run(text)
    r.font.name = FONT_NAME
    r.font.size = Pt(12)
    r.font.color.rgb = COLOR_BLACK
    return p

def add_bullet_p(doc, text, bold_prefix=None, level=0):
    p = doc.add_paragraph(style='List Bullet')
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.left_indent = Inches(0.25 * (level + 1))
    
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.font.name = FONT_NAME
        rb.font.size = Pt(12)
        rb.font.bold = True
        rb.font.color.rgb = COLOR_BLACK
        
    r = p.add_run(text)
    r.font.name = FONT_NAME
    r.font.size = Pt(12)
    r.font.color.rgb = COLOR_BLACK
    return p

def add_numbered_p(doc, num_str, text, bold_prefix=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.left_indent = Inches(0.35)
    p.paragraph_format.first_line_indent = Inches(-0.35)
    
    rn = p.add_run(f"{num_str}\t")
    rn.font.name = FONT_NAME
    rn.font.size = Pt(12)
    rn.font.bold = True
    rn.font.color.rgb = COLOR_BLACK
    
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.font.name = FONT_NAME
        rb.font.size = Pt(12)
        rb.font.bold = True
        rb.font.color.rgb = COLOR_BLACK
        
    r = p.add_run(text)
    r.font.name = FONT_NAME
    r.font.size = Pt(12)
    r.font.color.rgb = COLOR_BLACK
    return p

def add_callout_box(doc, text, title="CRITICAL METHODOLOGICAL DISTINCTION"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.0)
    set_cell_shading(cell, HEX_CALLOUT_BG)
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="1F4E79"/>'
        f'<w:top w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    r_title = p.add_run(f"[{title}]\n")
    r_title.font.name = FONT_NAME
    r_title.font.size = Pt(11)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_NAVY
    
    r_text = p.add_run(text)
    r_text.font.name = FONT_NAME
    r_text.font.size = Pt(10.5)
    r_text.font.italic = True
    r_text.font.color.rgb = COLOR_DARK_GRAY
    
    # Empty trailing spacing paragraph
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(0)
    p_sp.paragraph_format.space_after = Pt(6)

def add_code_block(doc, code_str, caption=None):
    if caption:
        p_cap = doc.add_paragraph()
        p_cap.paragraph_format.space_before = Pt(10)
        p_cap.paragraph_format.space_after = Pt(4)
        p_cap.paragraph_format.keep_with_next = True
        r_cap = p_cap.add_run(caption)
        r_cap.font.name = FONT_NAME
        r_cap.font.size = Pt(10.5)
        r_cap.font.bold = True
        r_cap.font.color.rgb = COLOR_BLACK

    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.0)
    set_cell_shading(cell, "F4F5F7")
    set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="8" w:space="0" w:color="CCCCCC"/>'
        f'<w:top w:val="single" w:sz="8" w:space="0" w:color="CCCCCC"/>'
        f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="CCCCCC"/>'
        f'<w:right w:val="single" w:sz="8" w:space="0" w:color="CCCCCC"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    
    r = p.add_run(code_str)
    r.font.name = FONT_CODE
    r.font.size = Pt(9.0)
    r.font.color.rgb = RGBColor(30, 30, 30)
    
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(0)
    p_sp.paragraph_format.space_after = Pt(6)

def add_table_data(doc, headers, data, col_widths=None, caption=None):
    if caption:
        p_cap = doc.add_paragraph()
        p_cap.paragraph_format.space_before = Pt(14)
        p_cap.paragraph_format.space_after = Pt(6)
        p_cap.paragraph_format.keep_with_next = True
        r_cap = p_cap.add_run(caption)
        r_cap.font.name = FONT_NAME
        r_cap.font.size = Pt(11)
        r_cap.font.bold = True
        r_cap.font.color.rgb = COLOR_BLACK
        
    num_cols = len(headers)
    table = doc.add_table(rows=len(data) + 1, cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    
    # Format header row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_shading(hdr_cells[i], HEX_HEADER_BG)
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=120, right=120)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = FONT_NAME
            r.font.size = Pt(10)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)
            
    # Set header repeat across pages
    trPr = table.rows[0]._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    
    # Format body rows
    for r_idx, row_values in enumerate(data):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = HEX_ALT_ROW if (r_idx % 2 == 1) else "FFFFFF"
        for c_idx, val in enumerate(row_values):
            row_cells[c_idx].text = str(val)
            set_cell_shading(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=80, bottom=80, left=100, right=100)
            p = row_cells[c_idx].paragraphs[0]
            # Auto-align: numbers centered/right, text left
            val_str = str(val).strip()
            if any(char.isdigit() for char in val_str) and len(val_str) < 15 and not val_str.startswith("0_") and not val_str.startswith("1_"):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                
            for r in p.runs:
                r.font.name = FONT_NAME
                r.font.size = Pt(9.5)
                r.font.color.rgb = COLOR_BLACK
                
    # Apply column widths if provided
    if col_widths and len(col_widths) == num_cols:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)
                
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(0)
    p_sp.paragraph_format.space_after = Pt(10)
    return table

def add_figure(doc, img_path, caption, width_in=5.8):
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(14)
    p_img.paragraph_format.space_after = Pt(4)
    p_img.paragraph_format.keep_with_next = True
    
    img_file = Path(img_path)
    if img_file.exists():
        run_img = p_img.add_run()
        run_img.add_picture(str(img_file), width=Inches(width_in))
    else:
        run_ph = p_img.add_run(f"[FIGURE ASSET NOT FOUND: {img_file.name} - TO BE INSERTED]")
        run_ph.font.name = FONT_NAME
        run_ph.font.size = Pt(11)
        run_ph.font.italic = True
        run_ph.font.color.rgb = RGBColor(180, 0, 0)
        
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(14)
    r_cap = p_cap.add_run(caption)
    r_cap.font.name = FONT_NAME
    r_cap.font.size = Pt(10.5)
    r_cap.font.bold = True
    r_cap.font.color.rgb = COLOR_BLACK
