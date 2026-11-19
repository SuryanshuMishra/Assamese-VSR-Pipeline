"""
Polish report layout in LMR_Pipeline_Report.docx and recompile to PDF:
- Fixes Table of Contents indentation so items 10, 11, 12, 13 use exact tab stops (0.35") matching items 1-9.
- Subsections 3.1-3.8 use clean nested indentation (0.65").
- All main section numbers are bold in theme blue (#1565C0).
- Eliminates all empty dummy paragraphs and random whitespace gaps.
- Removes arbitrary hard pageBreakBefore from sections 3, 6, 7, 8.
- Establishes clean page breaks: Cover Page (p.1) -> TOC (p.2) -> Section 1 (p.3).
- Resizes figures proportionally (max height 2.6 in) so they fit naturally into page flow without causing overflow voids.
- Enforces keep_with_next on all headings and figures so headings/captions never orphan.
- Configures repeating table headers (w:tblHeader) and flexible row splitting (w:cantSplit removed from body rows).
- Tightens cell padding and line spacing across tables and concluding block.
"""
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn

DOCX_PATH = Path("LMR_Pipeline_Report.docx")
PDF_PATH = Path("LMR_Pipeline_Report.pdf")

BLUE_COLOR = RGBColor(0x15, 0x65, 0xC0) # Theme blue
TEXT_COLOR = RGBColor(0x22, 0x22, 0x22)
SUB_COLOR = RGBColor(0x55, 0x55, 0x55)

def polish_doc():
    print(f"[*] Opening {DOCX_PATH} for layout polishing...")
    doc = docx.Document(DOCX_PATH)
    print(f"  Initial paragraph count: {len(doc.paragraphs)}")
    
    # 1. Remove all arbitrary pageBreakBefore
    for p in doc.paragraphs:
        pPr = p._p.pPr
        if pPr is not None:
            pbb = pPr.find(qn('w:pageBreakBefore'))
            if pbb is not None:
                pPr.remove(pbb)
                
    # 2. Format Table of Contents (P19 - P41)
    toc_start = None
    toc_end = None
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == "Table of Contents":
            toc_start = i
        elif p.text.strip() == "1. Project Overview":
            toc_end = i
            break
            
    if toc_start is not None and toc_end is not None:
        print(f"  Formatting Table of Contents between P{toc_start} and P{toc_end}...")
        toc_paras = [doc.paragraphs[i] for i in range(toc_start + 1, toc_end)]
        
        items = [
            ("main", "1.", "Project Overview"),
            ("main", "2.", "System Architecture"),
            ("main", "3.", "Modules Developed"),
            ("sub", "3.1", "Frame Extraction"),
            ("sub", "3.2", "Landmark Detection"),
            ("sub", "3.3", "Normalisation"),
            ("sub", "3.4", "Motion Features"),
            ("sub", "3.5", "Shape Features"),
            ("sub", "3.6", "Optical Flow"),
            ("sub", "3.7", "Quality Metrics"),
            ("sub", "3.8", "Visualisation"),
            ("main", "4.", "Pipeline Orchestrator"),
            ("main", "5.", "Supporting Tools"),
            ("main", "6.", "Google Drive Batch Pipeline"),
            ("main", "7.", "Pipeline Validation Results"),
            ("main", "8.", "Codebase Summary"),
            ("main", "9.", "Technology Stack"),
            ("main", "10.", "Key Design Decisions"),
            ("main", "11.", "Multi-Speaker Vowel vs. Consonant Machine Learning Model"),
            ("main", "12.", "Unseen Video Inference & Pipeline Integration"),
            ("main", "13.", "System Limitations, Failure Modes & Environmental Sensitivity"),
        ]
        
        for idx, (itype, num_str, title_str) in enumerate(items):
            p = toc_paras[idx]
            p.text = "" # Clear previous runs and properties
            
            if itype == "main":
                p.paragraph_format.left_indent = Inches(0.35)
                p.paragraph_format.first_line_indent = -Inches(0.35)
                p.paragraph_format.tab_stops.add_tab_stop(Inches(0.35))
                p.paragraph_format.space_before = Pt(6.0)
                p.paragraph_format.space_after = Pt(2.0)
                p.paragraph_format.line_spacing = 1.15
                p.paragraph_format.keep_with_next = True
                
                r0 = p.add_run(f"{num_str}\t")
                r0.font.name = "Calibri"
                r0.font.size = Pt(10.5)
                r0.font.bold = True
                r0.font.color.rgb = BLUE_COLOR
                
                r1 = p.add_run(title_str)
                r1.font.name = "Calibri"
                r1.font.size = Pt(10.5)
                r1.font.bold = False
                r1.font.color.rgb = TEXT_COLOR
            else:
                p.paragraph_format.left_indent = Inches(0.65)
                p.paragraph_format.first_line_indent = -Inches(0.30)
                p.paragraph_format.tab_stops.add_tab_stop(Inches(0.65))
                p.paragraph_format.space_before = Pt(2.0)
                p.paragraph_format.space_after = Pt(1.0)
                p.paragraph_format.line_spacing = 1.05
                p.paragraph_format.keep_with_next = True
                
                r0 = p.add_run(f"{num_str}\t")
                r0.font.name = "Calibri"
                r0.font.size = Pt(9.5)
                r0.font.bold = False
                r0.font.color.rgb = SUB_COLOR
                
                r1 = p.add_run(title_str)
                r1.font.name = "Calibri"
                r1.font.size = Pt(9.5)
                r1.font.bold = False
                r1.font.color.rgb = SUB_COLOR
                
    # 3. Add clean page break after Table of Contents (before Section 1)
    for p in doc.paragraphs:
        if p.text.strip() == "1. Project Overview":
            p.paragraph_format.page_break_before = True
            break
            
    # 4. Ensure keep_with_next on all body headings (SKIP Table of Contents!)
    for idx, p in enumerate(doc.paragraphs):
        if toc_end is not None and idx < toc_end:
            continue # DO NOT alter Table of Contents!
        if p.style.name.startswith("Heading") or (p.text and p.text[:2] in ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."] and len(p.text) < 70):
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(3)
            
    # 5. Remove completely empty paragraphs (no text, no images, no page breaks)
    to_delete = []
    for idx, p in enumerate(doc.paragraphs):
        if idx < 18:
            continue
        text = p.text.strip()
        has_graphic = any('graphic' in r._r.xml for r in p.runs)
        has_page_break = any('w:type="page"' in r._r.xml for r in p.runs)
        if len(text) == 0 and not has_graphic and not has_page_break:
            to_delete.append(p)
            
    print(f"  Removing {len(to_delete)} empty body paragraphs...")
    for p in to_delete:
        p._element.getparent().remove(p._element)
        
    # 6. Fix image sizing and keep_with_next on image paragraphs
    print("  Adjusting image sizes and setting keep_with_next on figures...")
    for p in doc.paragraphs:
        has_graphic = any('graphic' in r._r.xml for r in p.runs)
        if has_graphic:
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            
    for s in doc.inline_shapes:
        w_in = s.width / 914400.0
        h_in = s.height / 914400.0
        ar = w_in / h_in
        if h_in > 2.7:
            new_h = 2.6
            new_w = new_h * ar
            s.height = int(new_h * 914400)
            s.width = int(new_w * 914400)
        elif w_in > 5.5:
            new_w = 5.4
            new_h = new_w / ar
            s.width = int(new_w * 914400)
            s.height = int(new_h * 914400)
            
    # 7. Set tblHeader on tables and ensure rows can split cleanly
    print("  Updating table headers, row split properties, and cell margins...")
    for t in doc.tables:
        if len(t.rows) > 0:
            trPr0 = t.rows[0]._tr.get_or_add_trPr()
            if trPr0.find(qn('w:tblHeader')) is None:
                trPr0.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
        for r_idx, row in enumerate(t.rows):
            trPr = row._tr.get_or_add_trPr()
            if r_idx > 0:
                cantSplit = trPr.find(qn('w:cantSplit'))
                if cantSplit is not None:
                    trPr.remove(cantSplit)
            for cell in row.cells:
                tcPr = cell._tc.get_or_add_tcPr()
                tcMar = tcPr.find(qn('w:tcMar'))
                if tcMar is None:
                    tcPr.append(parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="60" w:type="dxa"/><w:bottom w:w="60" w:type="dxa"/><w:left w:w="100" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tcMar>'))
                for cp in cell.paragraphs:
                    cp.paragraph_format.space_before = Pt(2)
                    cp.paragraph_format.space_after = Pt(2)
                    cp.paragraph_format.line_spacing = 1.05

    # 8. Section 13 compact spacing to fit cleanly (body paragraphs only!)
    for idx, p in enumerate(doc.paragraphs):
        if toc_end is not None and idx < toc_end:
            continue
        if "13." in p.text:
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.1

    # 9. Ending block: keep together
    doc.paragraphs[-4].paragraph_format.space_before = Pt(6)
    doc.paragraphs[-4].paragraph_format.space_after = Pt(3)
    doc.paragraphs[-4].paragraph_format.keep_with_next = True
    doc.paragraphs[-3].paragraph_format.space_before = Pt(2)
    doc.paragraphs[-3].paragraph_format.space_after = Pt(2)
    doc.paragraphs[-3].paragraph_format.keep_with_next = True
    doc.paragraphs[-2].paragraph_format.space_before = Pt(2)
    doc.paragraphs[-2].paragraph_format.space_after = Pt(2)
    doc.paragraphs[-2].paragraph_format.keep_with_next = True
    doc.paragraphs[-1].paragraph_format.space_before = Pt(2)
    doc.paragraphs[-1].paragraph_format.space_after = Pt(2)
        
    doc.save(DOCX_PATH)
    print(f"[OK] Saved polished {DOCX_PATH} (paragraphs: {len(doc.paragraphs)})")
    
    # Recompile PDF via Word COM
    print(f"[*] Converting {DOCX_PATH} to {PDF_PATH} via Word COM...")
    try:
        import win32com.client
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        abs_docx = str(DOCX_PATH.resolve())
        abs_pdf = str(PDF_PATH.resolve())
        doc_obj = word.Documents.Open(abs_docx)
        doc_obj.SaveAs(abs_pdf, FileFormat=17) # wdFormatPDF
        doc_obj.Close()
        word.Quit()
        print(f"[OK] {PDF_PATH} recompiled successfully!")
    except Exception as e:
        print(f"[!] PDF compilation failed: {e}")

if __name__ == "__main__":
    polish_doc()
