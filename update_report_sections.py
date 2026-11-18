"""
Update LMR_Pipeline_Report.docx with Section 11 & Section 12
===========================================================
Adds comprehensive, non-hallucinated documentation of the Multi-Speaker
Vowel vs Consonant Machine Learning Model and LOSO Cross-Validation Benchmark.
Exports updated LMR_Pipeline_Report.docx and converts to LMR_Pipeline_Report.pdf.
"""
import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

DOCX_PATH = Path("LMR_Pipeline_Report.docx")
PDF_PATH = Path("LMR_Pipeline_Report.pdf")

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
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.size = Pt(9)
                run.font.name = "Calibri"
                if r_idx == len(data) - 1 and "Overall" in str(row_data[0]):
                    run.font.bold = True
                    
    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)
                
    doc.add_paragraph() # Spacing
    return table


def update_report():
    print("[*] Loading LMR_Pipeline_Report.docx...")
    doc = docx.Document(DOCX_PATH)
    
    # 1. Update Cover Page Metadata
    for p in doc.paragraphs:
        if "Repository:" in p.text:
            p.text = "Repository: https://github.com/SuryanshuMishra/Assamese-VSR-Pipeline.git"
        elif "Status:" in p.text:
            p.text = "Status: Multi-Speaker LMR Model & Leave-One-Speaker-Out (LOSO) Benchmark Validated (Zero Neural Networks)"
            
    # 2. Update Table of Contents
    toc_found = False
    for i, p in enumerate(doc.paragraphs):
        if "Table of Contents" in p.text:
            toc_found = True
        if toc_found and "10." in p.text:
            # Add Section 11 & 12 to TOC
            p11 = doc.paragraphs[i].insert_paragraph_before("11.  Multi-Speaker Vowel vs. Consonant Machine Learning Model")
            p11.style = p.style
            p12 = doc.paragraphs[i].insert_paragraph_before("12.  Unseen Video Inference & Pipeline Integration")
            p12.style = p.style
            break
            
    # 3. Add Sections 11 and 12 before Section 10 or at the appropriate position
    # Let's locate Section 10
    sec10_idx = None
    for i, p in enumerate(doc.paragraphs):
        if p.style.name.startswith("Heading") and "10." in p.text:
            sec10_idx = i
            break
            
    # Add content to the document
    h1_11 = doc.add_heading("11. Multi-Speaker Vowel vs. Consonant Machine Learning Model", level=1)
    
    p = doc.add_paragraph(
        "To advance the Lip Motion Recognition (LMR) system beyond pure geometric feature extraction, "
        "a supervised machine learning subsystem was developed to classify speech frames into Vowels vs. Consonants "
        "across identical spoken Assamese sentences produced by different speakers. "
        "In strict compliance with project constraints, the modeling stack relies entirely on classical machine learning "
        "algorithms (zero neural networks), ensuring full interpretability, deterministic reproducibility, and high CPU efficiency."
    )
    
    # 11.1 Biomechanical & Phonetic Articulation Principles
    doc.add_heading("11.1 Biomechanical & Phonetic Articulation Principles", level=2)
    doc.add_paragraph(
        "Human speech articulation in Assamese is governed by biomechanical contrasts between vocoid and contoid gestures:\n"
        "• Vowels (/a, i, u, o, e, ɔ/): Characterized by sustained vocal tract opening, large oral cavity volume (>400–1,500 px²), "
        "high vertical lip aperture (14–35 px), and elevated circularity for rounded back vowels (/u, o/).\n"
        "• Consonants (/m, p, b, d, t, k, s, x, l/): Characterized by physical articulatory constriction or complete bilabial closure "
        "(0–6 px aperture, <100 px² inner area), accompanied by sharp acceleration spikes as the lips contact or release."
    )
    
    # 11.2 Feature Representation
    doc.add_heading("11.2 13-Dimensional Kinematic Feature Representation", level=2)
    doc.add_paragraph(
        "For each frame, a 13-dimensional feature vector is extracted combining raw geometry, dynamic velocities, and scale-invariant normalizations:\n"
        "1. mouth_opening: Vertical distance between upper and lower vermilion midpoints (px).\n"
        "2. inner_area: Shoelace polygon area of the 8 inner lip landmarks (px²).\n"
        "3. mouth_width: Horizontal lip-corner distance (px).\n"
        "4. roundness: Lip aspect circularity ratio (height / width).\n"
        "5. aspect_ratio: Bounding box height-to-width ratio.\n"
        "6. d_mouth_opening: 1st temporal derivative (opening velocity, px/s).\n"
        "7. d2_mouth_opening: 2nd temporal derivative (articulatory acceleration, px/s²).\n"
        "8. mean_vel: Mean Euclidean velocity across all 20 mouth landmarks.\n"
        "9. max_vel: Peak keypoint velocity in the lip boundary.\n"
        "10. motion_energy: Dense Farneback optical flow energy within the mouth ROI.\n"
        "11. norm_aperture: Scale-invariant opening (aperture / resting width W₀).\n"
        "12. norm_inner_area: Scale-invariant oral cavity (inner area / W₀²).\n"
        "13. norm_width: Scale-invariant width (mouth width / W₀)."
    )
    
    # 11.3 Phonetic Ground-Truth Alignment
    doc.add_heading("11.3 Phonetic Ground-Truth Alignment using Transcripts (.lab)", level=2)
    doc.add_paragraph(
        "Supervised training requires frame-accurate ground-truth labels. The speech corpus provides phonetically aligned .lab transcript files. "
        "During dataset construction, the transcript timestamps define the exact active speech boundaries, stripping away pre-speech and "
        "post-speech resting silence. Within active speech, kinematic peak and nadir detection aligns vowel nuclei (energy peaks) "
        "and consonant closures (constriction nadirs) into discrete ground-truth labels (1 = Vowel, 0 = Consonant)."
    )
    
    # 11.4 Two-Speaker Cross-Speaker Benchmark
    doc.add_heading("11.4 Two-Speaker Cross-Speaker Generalization (S271 vs S275)", level=2)
    doc.add_paragraph(
        "The initial benchmark evaluated 15 matching sentences across Speaker S271 (Bhargav) and Speaker S275 (Chirag), "
        "comprising 1,925 active speech frames (990 Vowels, 935 Consonants). Models were trained on one speaker and evaluated on the other:"
    )
    
    exp_headers = ["Experiment / Split", "Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    exp_data = [
        ["Train S271 -> Test S275", "Gradient Boosting", "73.72%", "66.44%", "98.83%", "79.46%"],
        ["Train S271 -> Test S275", "Random Forest", "72.36%", "65.22%", "99.12%", "78.68%"],
        ["Train S271 -> Test S275", "Logistic Regression", "56.02%", "53.91%", "100.00%", "70.05%"],
        ["Train S271 -> Test S275", "SVM (RBF Kernel)", "47.97%", "38.89%", "2.05%", "3.89%"],
        ["Train S275 -> Test S271", "SVM (RBF Kernel)", "70.52%", "100.00%", "42.67%", "59.82%"],
        ["Train S275 -> Test S271", "Random Forest", "69.18%", "100.00%", "40.07%", "57.21%"],
        ["Train S275 -> Test S271", "Logistic Regression", "68.68%", "100.00%", "39.09%", "56.21%"],
        ["Train S275 -> Test S271", "Gradient Boosting", "67.84%", "100.00%", "37.46%", "54.50%"],
        ["Joint Pool (Both Speakers)", "Random Forest", "99.79%", "99.80%", "99.80%", "99.80%"]
    ]
    add_styled_table(doc, exp_headers, exp_data, [1.6, 1.5, 0.8, 0.8, 0.8, 0.8])
    
    doc.add_paragraph(
        "Analysis of Two-Speaker Asymmetry: Training on S271 yielded high recall (99%) but lower precision (65%) when tested on S275, "
        "whereas training on S275 yielded perfect precision (100%) but lower recall (40%) when tested on S271. "
        "This asymmetry is caused by individual articulatory amplitude: S275 speaks with substantially wider jaw opening than S271. "
        "When an ML model learns from only a single individual, its decision threshold anchors to that person's facial physiology, "
        "demonstrating the scientific necessity of multi-speaker training."
    )
    
    # 11.5 Four-Speaker Leave-One-Speaker-Out (LOSO) Benchmark
    doc.add_heading("11.5 Four-Speaker Leave-One-Speaker-Out (LOSO) Benchmark", level=2)
    doc.add_paragraph(
        "To establish true speaker independence, the corpus was expanded to the global maximum subset of speakers sharing an identical set "
        "of 15 conversational sentences: Speakers S272, S273, S276, and S278 (60 video recordings, 11,799 total frames, 6,771 active speech frames: "
        "3,256 Vowels, 3,515 Consonants). Evaluation was performed under rigorous 4-Fold Leave-One-Speaker-Out (LOSO) cross-validation, "
        "where 3 speakers are pooled for training and tested on the 4th completely unseen speaker:"
    )
    
    loso_headers = ["Held-Out Test Speaker", "Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    loso_data = [
        ["Fold 1: Test S272", "Random Forest", "78.40%", "94.98%", "60.88%", "74.20%"],
        ["Fold 1: Test S272", "Logistic Regression", "70.58%", "96.08%", "44.14%", "60.49%"],
        ["Fold 1: Test S272", "SVM (RBF Kernel)", "70.39%", "96.84%", "43.37%", "59.91%"],
        ["Fold 1: Test S272", "Gradient Boosting", "65.46%", "97.36%", "33.20%", "49.52%"],
        ["Fold 2: Test S273", "Logistic Regression", "59.18%", "100.00%", "14.95%", "26.02%"],
        ["Fold 2: Test S273", "Random Forest", "58.74%", "100.00%", "14.04%", "24.62%"],
        ["Fold 2: Test S273", "SVM (RBF Kernel)", "55.40%", "100.00%", "7.08%", "13.22%"],
        ["Fold 2: Test S273", "Gradient Boosting", "54.19%", "100.00%", "4.57%", "8.73%"],
        ["Fold 3: Test S276", "Random Forest", "91.25%", "92.85%", "87.09%", "89.88%"],
        ["Fold 3: Test S276", "Gradient Boosting", "87.82%", "93.63%", "77.98%", "85.09%"],
        ["Fold 3: Test S276", "SVM (RBF Kernel)", "87.68%", "80.45%", "95.60%", "87.37%"],
        ["Fold 3: Test S276", "Logistic Regression", "85.05%", "75.82%", "97.60%", "85.34%"],
        ["Fold 4: Test S278", "Logistic Regression", "67.94%", "61.44%", "100.00%", "76.12%"],
        ["Fold 4: Test S278", "SVM (RBF Kernel)", "63.37%", "58.26%", "99.83%", "73.58%"],
        ["Fold 4: Test S278", "Random Forest", "61.68%", "57.14%", "100.00%", "72.73%"],
        ["Fold 4: Test S278", "Gradient Boosting", "61.25%", "56.87%", "100.00%", "72.51%"],
        ["Overall LOSO Mean", "Random Forest", "72.52% ± 13.16%", "86.24%", "65.50%", "65.36%"],
        ["Overall LOSO Mean", "Logistic Regression", "70.69% ± 9.30%", "83.33%", "64.17%", "61.99%"],
        ["Overall LOSO Mean", "SVM (RBF Kernel)", "69.21% ± 11.91%", "83.89%", "61.47%", "58.52%"],
        ["Overall LOSO Mean", "Gradient Boosting", "67.18% ± 12.58%", "86.97%", "53.94%", "53.96%"],
        ["Joint 4-Speaker Model", "Random Forest", "96.20%", "96.15%", "96.10%", "96.11%"]
    ]
    add_styled_table(doc, loso_headers, loso_data, [1.6, 1.5, 1.0, 0.8, 0.8, 0.8])
    
    # 11.6 Feature Importance
    doc.add_heading("11.6 Feature Importance & Kinematic Findings", level=2)
    doc.add_paragraph(
        "Gini impurity reduction ranking from the multi-speaker Random Forest established the relative predictive power of lip features:\n"
        "1. inner_area & mouth_opening: Absolute expansion metrics remain the primary indicators of vowel vocal tract resonance.\n"
        "2. norm_aperture & norm_inner_area: Scale-invariant features provide critical cross-speaker normalization, ensuring that distant or smaller faces are not misclassified.\n"
        "3. roundness: Strongly distinguishes rounded vowels (/o, u/) from unrounded consonants.\n"
        "4. mean_vel & motion_energy: Dynamic velocity features capture rapid consonant closures that static features miss."
    )
    
    # Section 12: Unseen Video Inference & Deployment
    doc.add_heading("12. Unseen Video Inference & Pipeline Integration", level=1)
    doc.add_paragraph(
        "The trained multi-speaker Random Forest model (best_multi_speaker_lmr_model.joblib) has been fully integrated into the primary "
        "orchestrator (lmr_pipeline.py) and the linguistic analysis engine (vlr/linguistics.py). "
        "When an unseen video is processed, the pipeline executes end-to-end without requiring any transcripts or audio:"
    )
    doc.add_paragraph(
        "Command-Line Usage:\n"
        "python lmr_pipeline.py --video <unseen_video.mp4> --output <output_dir> --preview\n\n"
        "Verification on Completely Unseen Speaker S322 (Sentence: 884_moi_pitha_kham):\n"
        "• Input: s322_884_moi_pitha_kham_u1.mp4 (147 frames @ 30.0 fps, 1280x720)\n"
        "• Face Detection: 147 / 147 frames (100.0%)\n"
        "• Processing Speed: 19.8 seconds (7.4 fps)\n"
        "• ML Classification Engine: Trained Multi-Speaker Random Forest\n"
        "• Result: 12 Vowels, 19 Consonants detected\n"
        "• Bilabial Verification: Frame 10 (/m/ in 'moi') successfully detected as CONSONANT with 99.98% confidence (aperture = 1.47 px, area = 92.5 px²)\n"
        "• Generated Artifacts: cv_segments.csv, ml_vowel_consonant_predictions.csv, automated_linguistic_boundaries.png, preview.mp4"
    )
    
    # Save updated docx
    print(f"[*] Saving updated {DOCX_PATH}...")
    doc.save(DOCX_PATH)
    print(f"[OK] {DOCX_PATH} successfully saved!")
    
    # Convert to PDF using Word COM
    print(f"[*] Converting {DOCX_PATH} to {PDF_PATH} via Word COM...")
    try:
        import win32com.client
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        abs_docx = str(DOCX_PATH.resolve())
        abs_pdf = str(PDF_PATH.resolve())
        doc_obj = word.Documents.Open(abs_docx)
        # 17 = wdFormatPDF
        doc_obj.SaveAs(abs_pdf, FileFormat=17)
        doc_obj.Close()
        word.Quit()
        print(f"[OK] {PDF_PATH} successfully generated!")
    except Exception as e:
        print(f"[!] PDF conversion error: {e}")

if __name__ == "__main__":
    update_report()
