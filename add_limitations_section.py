"""
Add Section 13 (System Limitations & Failure Modes) to LMR_Pipeline_Report.docx
=============================================================================
1. Adds TOC item '13.  System Limitations, Failure Modes & Environmental Sensitivity'
   with '13.  ' in theme blue RGB(0x15, 0x65, 0xC0).
2. Appends comprehensive Section 13 with technical sub-sections.
3. Saves LMR_Pipeline_Report.docx and exports to LMR_Pipeline_Report.pdf via Word COM.
"""
from pathlib import Path
import docx
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

DOCX_PATH = Path("LMR_Pipeline_Report.docx")
PDF_PATH = Path("LMR_Pipeline_Report.pdf")

BLUE_COLOR = RGBColor(0x15, 0x65, 0xC0) # Theme blue #1565C0

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
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

def add_styled_table(doc, headers, data, col_widths=None):
    table = doc.add_table(rows=len(data) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    # Header row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], "1F4E79")
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=140, right=140)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.font.size = Pt(9.5)
            run.font.name = "Calibri"
            
    # Data rows
    for r_idx, row_data in enumerate(data):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = "F2F4F7" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=80, bottom=80, left=140, right=140)
            p = row_cells[c_idx].paragraphs[0]
            if c_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                run.font.size = Pt(9)
                run.font.name = "Calibri"
                    
    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)
                
    doc.add_paragraph() # Spacing
    return table

def add_limitations():
    print(f"[*] Opening {DOCX_PATH}...")
    doc = docx.Document(DOCX_PATH)
    
    # 1. Update Table of Contents
    # Paragraph 41 currently is empty, before '1. Project Overview' (P42)
    p41 = doc.paragraphs[41]
    p41.text = ""
    r0 = p41.add_run("13.  ")
    r0.font.color.rgb = BLUE_COLOR
    r0.font.name = "Calibri"
    r1 = p41.add_run("System Limitations, Failure Modes & Environmental Sensitivity")
    r1.font.name = "Calibri"
    
    # Insert an empty line after p41 for spacing before P42
    p_spacing = doc.paragraphs[42].insert_paragraph_before("")
    
    # 2. Add Section 13 at the end of the document
    h1 = doc.add_heading("13. System Limitations, Failure Modes & Environmental Sensitivity", level=1)
    
    doc.add_paragraph(
        "A rigorous scientific evaluation requires establishing the operational boundaries, environmental constraints, "
        "and physical failure modes of the computer vision and machine learning pipeline. "
        "While the system demonstrates high speaker-independent accuracy (72.52% mean LOSO, 96.20% joint) under standard "
        "recording conditions, the underlying landmark tracking and kinematic feature extraction exhibit specific performance "
        "degradations when subjected to adverse photometric, geometric, and temporal conditions."
    )
    
    # 13.1 Illumination & Photometric Degradation
    doc.add_heading("13.1 Illumination & Photometric Degradation", level=2)
    doc.add_paragraph(
        "The primary facial landmark localization relies on dlib's Histogram of Oriented Gradients (HOG) frontal face detector "
        "and an ensemble of regression trees (Kazemi & Sullivan, 2014). This approach requires clear spatial image intensity gradients:\n"
        "• Severe Underexposure (Mean Luma < 40 / 255): In poorly lit environments, contrast between the lip vermilion border "
        "and surrounding perioral skin diminishes significantly. As a consequence, landmarks 48–67 suffer severe spatial jitter "
        "or collapse towards the mouth centroid, causing false 'Consonant' (closure) predictions even when the speaker vocalizes a vowel.\n"
        "• Harsh Directional Shadows: Asymmetric lateral lighting casts dark shadow gradients across one oral commissure (corner), "
        "leading the detector to misestimate horizontal lip width (mouth_width) and skewing the roundness and scale-invariant norm_aperture ratios.\n"
        "• Overexposure & Backlighting: Saturated highlights on the lips wash out the distinct inner-lip boundary, preventing the "
        "Shoelace algorithm from accurately computing the oral cavity area (inner_area)."
    )
    
    # 13.2 Head Pose & Off-Axis Perspective Foreshortening
    doc.add_heading("13.2 Head Pose & Off-Axis Perspective Foreshortening", level=2)
    doc.add_paragraph(
        "The pipeline is designed and calibrated for frontal and near-frontal conversational cameras (yaw/pitch within ±15°):\n"
        "• Out-of-Plane Yaw Rotation (>20°): When a speaker turns their head away from the camera, 2D perspective foreshortening "
        "compresses the apparent horizontal width of the mouth while the far lip corner may become occluded. Because scale-invariant "
        "normalization relies on resting width W₀, yaw angles cause artificial inflation of norm_aperture, triggering false vowel detections.\n"
        "• Pitch Alterations (Looking Up/Down): Upward pitch exposes submental (under-chin) anatomy, shifting the outer lip contour, "
        "while downward pitch causes the upper lip to occlude the inner oral cavity, suppressing measured inner_area."
    )
    
    # 13.3 Articulatory Motion Blur & Temporal Nyquist Gaps
    doc.add_heading("13.3 Articulatory Motion Blur & Temporal Nyquist Sampling", level=2)
    doc.add_paragraph(
        "Standard conversational video is recorded at ~30 frames per second (frame interval Δt ≈ 33.3 ms):\n"
        "• Transient Consonant Release Bursts: Plosive consonants in Assamese (/p, b, t, d, k/) exhibit acoustic voice onset times "
        "and bilabial release phases lasting only 20–40 ms. At 30 fps, a plosive closure may occupy only a single video frame. "
        "If video capture occurs during the transition rather than peak closure, the nadir is blunted, reducing consonant recall.\n"
        "• Motion Blur: Rapid articulatory gestures under slow camera exposure shutter speeds (>1/60s) generate blurred boundaries. "
        "Motion blur smooths out gradient edges, causing landmark localization to underestimate peak opening velocity (d_mouth_opening)."
    )
    
    # 13.4 Facial Occlusions & Anatomical Variability
    doc.add_heading("13.4 Facial Occlusions & Anatomical Variability", level=2)
    doc.add_paragraph(
        "The model assumes an unobstructed view of the vermilion border:\n"
        "• Dense Facial Hair: Prominent moustaches covering the upper vermilion border (landmarks 50–52) force the landmark regressor "
        "to infer rather than detect lip boundaries, leading to constant aperture offsets.\n"
        "• External Obstructions: Microphones, hands resting near the face, or coffee cups entering the lower facial region trigger "
        "dlib face detection dropouts or erratic landmark warping."
    )
    
    # 13.5 Quality Gating Defense in quality.py
    doc.add_heading("13.5 Proactive Gating Defense via quality.py", level=2)
    doc.add_paragraph(
        "To mitigate these physical limitations, the pipeline implements an explicit quality gating module (lmr/quality.py) "
        "that evaluates every frame prior to downstream feature extraction and ML inference:\n"
        "• Laplacian Variance Blur Filter: Measures high-frequency gradient variance; frames falling below the sharpness threshold are flagged.\n"
        "• Luma Brightness Checker: Identifies frames with mean intensity < 40 (underexposed) or > 230 (overexposed).\n"
        "• Crop-Centre Jitter Metric: Flags rapid camera shakes or unstable face tracking.\n"
        "• Boolean Validity Mask: Produces a per-frame validity_mask exported to quality_log.csv, enabling downstream models "
        "to automatically discard corrupt frames rather than producing false linguistic classifications."
    )
    
    # Limitations Summary Matrix
    lim_headers = ["Environmental Factor", "Degradation Mechanism", "Impact on Features", "Mitigation Strategy"]
    lim_data = [
        ["Low Illumination (<40 luma)", "HOG gradient contrast loss", "Aperture underestimation (false closures)", "quality.py luma check; adaptive histogram equalization"],
        ["Yaw Head Pose (>20°)", "Perspective foreshortening", "Mouth width compression; skewed norm_aperture", "Face angle thresholding; 3D head pose estimation"],
        ["Rapid Motion Blur", "Shutter lag during high-velocity speech", "Blunted velocity peaks (d_mouth_opening)", "Higher camera FPS (60+ fps); Laplacian blur gating"],
        ["Upper Lip Occlusion (Moustache/Hand)", "Keypoint regression occlusion", "Artificially reduced inner_area", "Quality validity masking; temporal smoothing (Savitzky-Golay)"]
    ]
    add_styled_table(doc, lim_headers, lim_data)
    
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
        print(f"[OK] {PDF_PATH} regenerated successfully with Section 13!")
    except Exception as e:
        print(f"[!] PDF conversion error: {e}")

if __name__ == "__main__":
    add_limitations()
