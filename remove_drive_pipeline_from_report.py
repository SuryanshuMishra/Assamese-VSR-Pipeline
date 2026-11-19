"""
Remove legacy Google Drive Batch Pipeline from LMR_Pipeline_Report.docx,
renumber all subsequent sections from 6 onwards, update TOC with symmetric spacing,
and recompile to PDF via Word COM.
"""
from pathlib import Path
import re
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn

DOCX_PATH = Path("LMR_Pipeline_Report.docx")
PDF_PATH = Path("LMR_Pipeline_Report.pdf")

BLUE_COLOR = RGBColor(0x15, 0x65, 0xC0) # Theme blue
TEXT_COLOR = RGBColor(0x22, 0x22, 0x22)
SUB_COLOR = RGBColor(0x55, 0x55, 0x55)

def update_report():
    print(f"[*] Opening {DOCX_PATH}...")
    doc = docx.Document(DOCX_PATH)
    
    # 1. Update Section 1 and Section 2 prose references to Drive
    print("[*] Updating introductory prose references...")
    for p in doc.paragraphs:
        if "cloud-native batch processor (drive_batch_pipeline.py)" in p.text:
            p.text = p.text.replace(
                "and a cloud-native batch processor (drive_batch_pipeline.py) for automated large-scale dataset extraction.",
                "and automated local dataset batch processing scripts for large-scale corpus extraction."
            )
            print("  Updated Section 1 prose.")
        elif "corpus-level batch orchestrator that traverses Google Drive storage" in p.text:
            p.text = p.text.replace(
                "and (2) a corpus-level batch orchestrator that traverses Google Drive storage, manages streaming downloads, coordinates per-sentence processing, and collates outputs into structured global repositories.",
                "and (2) a corpus-level batch workflow that coordinates per-sentence processing across local video directories and collates outputs into structured feature repositories."
            )
            print("  Updated Section 2 prose.")
            
    # 2. Update Validation Results table (row 12) if it mentions Google Drive
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                if "Google Drive API v3" in cell.text:
                    cell.text = "Local File I/O & Memory-Piped Video Streaming"
                    for cp in cell.paragraphs:
                        for r in cp.runs:
                            r.font.name = "Calibri"
                            r.font.size = Pt(9)
                    print("  Updated Validation Table row 12.")

    # 3. Locate and remove Section 6 XML elements (paragraphs and tables)
    print("[*] Removing Section 6 (Google Drive Batch Pipeline) elements...")
    start_sec6 = False
    elements_to_remove = []
    
    for el in doc.element.body:
        tag = el.tag.split('}')[-1]
        text = ""
        if tag == 'p':
            p = docx.text.paragraph.Paragraph(el, doc)
            text = p.text.strip()
        elif tag == 'tbl':
            tbl = docx.table.Table(el, doc)
            text = tbl.rows[0].cells[0].text.strip()
            
        if text.startswith("6. Google Drive Batch Pipeline"):
            start_sec6 = True
            
        if start_sec6:
            if text.startswith("7. Pipeline Validation Results"):
                start_sec6 = False
                break
            elements_to_remove.append(el)
            
    print(f"  Found {len(elements_to_remove)} XML elements to remove from Section 6.")
    for el in elements_to_remove:
        el.getparent().remove(el)
        
    # 4. Renumber headings and subheadings throughout the body
    print("[*] Renumbering subsequent headings in body...")
    renumber_map = [
        ("7. Pipeline Validation Results", "6. Pipeline Validation Results"),
        ("8. Codebase Summary", "7. Codebase Summary"),
        ("9. Technology Stack", "8. Technology Stack"),
        ("10. Key Design Decisions", "9. Key Design Decisions"),
        ("11. Multi-Speaker Vowel vs. Consonant Machine Learning Model", "10. Multi-Speaker Vowel vs. Consonant Machine Learning Model"),
        ("11.1 Biomechanical & Phonetic Articulation Principles", "10.1 Biomechanical & Phonetic Articulation Principles"),
        ("11.2 13-Dimensional Kinematic Feature Representation", "10.2 13-Dimensional Kinematic Feature Representation"),
        ("11.3 Phonetic Ground-Truth Alignment using Transcripts (.lab)", "10.3 Phonetic Ground-Truth Alignment using Transcripts (.lab)"),
        ("11.4 Two-Speaker Cross-Speaker Generalization (S271 vs S275)", "10.4 Two-Speaker Cross-Speaker Generalization (S271 vs S275)"),
        ("11.5 Four-Speaker Leave-One-Speaker-Out (LOSO) Benchmark", "10.5 Four-Speaker Leave-One-Speaker-Out (LOSO) Benchmark"),
        ("11.6 Feature Importance & Kinematic Findings", "10.6 Feature Importance & Kinematic Findings"),
        ("12. Unseen Video Inference & Pipeline Integration", "11. Unseen Video Inference & Pipeline Integration"),
        ("13. System Limitations, Failure Modes & Environmental Sensitivity", "12. System Limitations, Failure Modes & Environmental Sensitivity"),
        ("13.1 Illumination & Photometric Degradation", "12.1 Illumination & Photometric Degradation"),
        ("13.2 Head Pose & Off-Axis Perspective Foreshortening", "12.2 Head Pose & Off-Axis Perspective Foreshortening"),
        ("13.3 Articulatory Motion Blur & Temporal Nyquist Sampling", "12.3 Articulatory Motion Blur & Temporal Nyquist Sampling"),
        ("13.4 Facial Occlusions & Anatomical Variability", "12.4 Facial Occlusions & Anatomical Variability"),
        ("13.5 Proactive Gating Defense via quality.py", "12.5 Proactive Gating Defense via quality.py"),
    ]
    
    for p in doc.paragraphs:
        txt = p.text.strip()
        for old_h, new_h in renumber_map:
            if txt.startswith(old_h):
                p.text = p.text.replace(old_h, new_h)
                print(f"  Renumbered: {old_h} -> {new_h}")
                break

    # 5. Rebuild Table of Contents
    print("[*] Rebuilding Table of Contents with 12 sections and symmetric spacing...")
    toc_start = None
    toc_end = None
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == "Table of Contents":
            toc_start = i
        elif p.text.strip() == "1. Project Overview":
            toc_end = i
            break
            
    # Delete old TOC paragraphs between toc_start and toc_end
    old_toc_paras = [doc.paragraphs[i] for i in range(toc_start + 1, toc_end)]
    for p in old_toc_paras:
        p._element.getparent().remove(p._element)
        
    # Re-insert fresh TOC paragraphs right after toc_start
    p_anchor = doc.paragraphs[toc_start]
    
    new_toc_items = [
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
        ("main", "6.", "Pipeline Validation Results"),
        ("main", "7.", "Codebase Summary"),
        ("main", "8.", "Technology Stack"),
        ("main", "9.", "Key Design Decisions"),
        ("main", "10.", "Multi-Speaker Vowel vs. Consonant Machine Learning Model"),
        ("main", "11.", "Unseen Video Inference & Pipeline Integration"),
        ("main", "12.", "System Limitations, Failure Modes & Environmental Sensitivity"),
    ]
    
    prev_p = p_anchor
    for itype, num_str, title_str in new_toc_items:
        new_p = doc.add_paragraph()
        prev_p._p.addnext(new_p._p)
        prev_p = new_p
        
        if itype == "main":
            new_p.paragraph_format.left_indent = Inches(0.35)
            new_p.paragraph_format.first_line_indent = -Inches(0.35)
            new_p.paragraph_format.tab_stops.add_tab_stop(Inches(0.35))
            new_p.paragraph_format.space_before = Pt(6.0)
            new_p.paragraph_format.space_after = Pt(2.0)
            new_p.paragraph_format.line_spacing = 1.15
            new_p.paragraph_format.keep_with_next = True
            
            r0 = new_p.add_run(f"{num_str}\t")
            r0.font.name = "Calibri"
            r0.font.size = Pt(10.5)
            r0.font.bold = True
            r0.font.color.rgb = BLUE_COLOR
            
            r1 = new_p.add_run(title_str)
            r1.font.name = "Calibri"
            r1.font.size = Pt(10.5)
            r1.font.bold = False
            r1.font.color.rgb = TEXT_COLOR
        else:
            new_p.paragraph_format.left_indent = Inches(0.65)
            new_p.paragraph_format.first_line_indent = -Inches(0.30)
            new_p.paragraph_format.tab_stops.add_tab_stop(Inches(0.65))
            new_p.paragraph_format.space_before = Pt(2.0)
            new_p.paragraph_format.space_after = Pt(1.0)
            new_p.paragraph_format.line_spacing = 1.05
            new_p.paragraph_format.keep_with_next = True
            
            r0 = new_p.add_run(f"{num_str}\t")
            r0.font.name = "Calibri"
            r0.font.size = Pt(9.5)
            r0.font.bold = False
            r0.font.color.rgb = SUB_COLOR
            
            r1 = new_p.add_run(title_str)
            r1.font.name = "Calibri"
            r1.font.size = Pt(9.5)
            r1.font.bold = False
            r1.font.color.rgb = SUB_COLOR
            
    # 6. Ensure clean page break before Section 1
    for p in doc.paragraphs:
        if p.text.strip() == "1. Project Overview":
            p.paragraph_format.page_break_before = True
            break
            
    # Save DOCX
    print(f"[*] Saving updated {DOCX_PATH}...")
    doc.save(DOCX_PATH)
    print(f"[OK] {DOCX_PATH} saved successfully!")
    
    # 7. Convert to PDF via Word COM
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
    update_report()
