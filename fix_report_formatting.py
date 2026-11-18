"""
Fix Table of Contents Blue Highlighting and Remove VLR references
==================================================================
1. Highlights '11.  ' and '12.  ' in blue color RGB(0x15, 0x65, 0xc0) in Table of Contents.
2. Orders TOC correctly: 10, 11, 12.
3. Renames 'vlr/linguistics.py' and 'vlr' to 'linguistic.py' / 'linguistic' only.
4. Updates Table 19 (Codebase summary) to include linguistic.py and remove vlr/ prefix.
5. Converts to LMR_Pipeline_Report.pdf via Word COM.
"""
from pathlib import Path
import docx
from docx.shared import Pt, RGBColor
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

DOCX_PATH = Path("LMR_Pipeline_Report.docx")
PDF_PATH = Path("LMR_Pipeline_Report.pdf")

BLUE_COLOR = RGBColor(0x15, 0x65, 0xC0) # Theme blue

def fix_report():
    print(f"[*] Opening {DOCX_PATH}...")
    doc = docx.Document(DOCX_PATH)
    
    # --- 1. Fix Table of Contents (Paragraphs 38, 39, 40) ---
    print("[*] Fixing Table of Contents sequence and blue highlighting...")
    p38 = doc.paragraphs[38]
    p39 = doc.paragraphs[39]
    p40 = doc.paragraphs[40]
    
    # Item 10: Key Design Decisions
    p38.text = ""
    r0 = p38.add_run("10.  ")
    r0.font.color.rgb = BLUE_COLOR
    r0.font.name = "Calibri"
    r1 = p38.add_run("Key Design Decisions")
    r1.font.name = "Calibri"
    
    # Item 11: Multi-Speaker Vowel vs. Consonant Machine Learning Model
    p39.text = ""
    r0 = p39.add_run("11.  ")
    r0.font.color.rgb = BLUE_COLOR
    r0.font.name = "Calibri"
    r1 = p39.add_run("Multi-Speaker Vowel vs. Consonant Machine Learning Model")
    r1.font.name = "Calibri"
    
    # Item 12: Unseen Video Inference & Pipeline Integration
    p40.text = ""
    r0 = p40.add_run("12.  ")
    r0.font.color.rgb = BLUE_COLOR
    r0.font.name = "Calibri"
    r1 = p40.add_run("Unseen Video Inference & Pipeline Integration")
    r1.font.name = "Calibri"
    
    # --- 2. Remove 'vlr' references across all paragraphs ---
    print("[*] Removing 'vlr' references in prose paragraphs...")
    for i, p in enumerate(doc.paragraphs):
        if "vlr/linguistics.py" in p.text:
            p.text = p.text.replace("vlr/linguistics.py", "linguistic.py")
            print(f"  P{i}: Replaced vlr/linguistics.py with linguistic.py")
        elif "vlr/linguistics" in p.text:
            p.text = p.text.replace("vlr/linguistics", "linguistic")
            print(f"  P{i}: Replaced vlr/linguistics with linguistic")
        elif "(vlr/)" in p.text:
            p.text = p.text.replace("(vlr/)", "")
            print(f"  P{i}: Removed (vlr/)")
        elif "vlr/" in p.text and not p.text.startswith("http"):
            p.text = p.text.replace("vlr/", "")
            print(f"  P{i}: Removed vlr/ prefix")
            
    # --- 3. Update Table 19 (Codebase Summary) ---
    print("[*] Updating Table 19 (Codebase Summary)...")
    tbl = doc.tables[19]
    for row in tbl.rows:
        for cell in row.cells:
            if "vlr/extraction.py" in cell.text:
                cell.text = "extraction.py"
            elif "vlr/detection.py" in cell.text:
                cell.text = "detection.py"
            elif "vlr/normalization.py" in cell.text:
                cell.text = "normalization.py"
            elif "vlr/motion.py" in cell.text:
                cell.text = "motion.py"
            elif "vlr/shape.py" in cell.text:
                cell.text = "shape.py"
            elif "vlr/optical_flow.py" in cell.text:
                cell.text = "optical_flow.py"
            elif "vlr/quality.py" in cell.text:
                cell.text = "quality.py"
            elif "vlr/visualization.py" in cell.text:
                cell.text = "visualization.py"
            elif "vlr/__init__.py" in cell.text:
                cell.text = "linguistic.py"
                # Update purpose in the same row
                row.cells[0].text = "Linguistic Model"
                row.cells[2].text = "334"
                row.cells[3].text = "Vowel/consonant ML prediction & boundary segmentation"
                
    # Style Table 19 font sizes
    for row in tbl.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.name = "Calibri"
                    r.font.size = Pt(9)
                    
    # Save DOCX
    print(f"[*] Saving updated {DOCX_PATH}...")
    doc.save(DOCX_PATH)
    print(f"[OK] {DOCX_PATH} saved successfully!")
    
    # Convert to PDF via Word COM
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
        print(f"[OK] {PDF_PATH} regenerated successfully!")
    except Exception as e:
        print(f"[!] PDF generation error: {e}")

if __name__ == "__main__":
    fix_report()
