"""
Embed all diagnostic/evaluation graphs and unseen video output tables into LMR_Pipeline_Report.docx
and recompile to PDF via Word COM.
"""
from pathlib import Path
import shutil
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

DOCX_PATH = Path("LMR_Pipeline_Report.docx")
BACKUP_PATH = Path("LMR_Pipeline_Report_backup.docx")
PDF_PATH = Path("LMR_Pipeline_Report.pdf")

def set_cell_background(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{hex_color}"/>')
    tcPr.append(shd)

def create_figure(doc, img_path, fig_num, caption_title, caption_desc, width_inches=5.8):
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(3)
    run_img = p_img.add_run()
    run_img.add_picture(str(Path(img_path).resolve()), width=Inches(width_inches))
    
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(12)
    
    r_label = p_cap.add_run(f"Figure {fig_num}: ")
    r_label.font.name = "Calibri"
    r_label.font.size = Pt(9)
    r_label.font.bold = True
    r_label.font.color.rgb = RGBColor(0x0D, 0x47, 0xA1) # Theme blue
    
    r_title = p_cap.add_run(caption_title + " ")
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(9)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    
    r_desc = p_cap.add_run(caption_desc)
    r_desc.font.name = "Calibri"
    r_desc.font.size = Pt(9)
    r_desc.font.italic = True
    r_desc.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    
    return p_img, p_cap

def create_unseen_output_table(doc):
    table_data = [
        ("Input Unseen Video File", "s322_884_moi_pitha_kham_u1.mp4 (Speaker S322, Sentence 884)"),
        ("Spoken Sentence Meaning", "Assamese: 'মই পিঠা খাম' (English: 'I will eat pitha')"),
        ("Video Specifications", "147 frames @ 30.0 fps, 1280 × 720 px, Total Duration: 4.87 seconds"),
        ("Face Tracking Success Rate", "147 / 147 frames (100.0% landmark detection, 0 dropped frames)"),
        ("End-to-End Execution Speed", "19.8 seconds total runtime (~7.4 fps on commodity CPU)"),
        ("Detected Vowel Segments", "7 continuous vowel segments (Mean duration: 228.6 ms, Mean aperture: 12.79 px)"),
        ("Detected Consonant Segments", "14 consonant segments (Mean duration: 157.1 ms, Mean aperture: 7.18 px)"),
        ("Peak Articulatory Aperture", "27.36 px (observed at frame 62 during open central vowel /a/)"),
        ("Peak Inner Oral Cavity Area", "1,808.5 px² (inner vermilion border Shoelace polygon integration)"),
        ("Bilabial Closure Verification", "Frame 10 (/m/ onset in 'moi') classified as CONSONANT (99.98% confidence, aperture = 1.47 px)"),
        ("Generated Output Artifacts", "cv_segments.csv, ml_vowel_consonant_predictions.csv, automated_linguistic_boundaries.png, preview.mp4")
    ]
    
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_title.paragraph_format.space_before = Pt(8)
    p_title.paragraph_format.space_after = Pt(3)
    r_tbl_label = p_title.add_run("Table 24: ")
    r_tbl_label.font.name = "Calibri"
    r_tbl_label.font.size = Pt(9.5)
    r_tbl_label.font.bold = True
    r_tbl_label.font.color.rgb = RGBColor(0x0D, 0x47, 0xA1)
    
    r_tbl_title = p_title.add_run("Quantitative Output Metrics from Live Inference on Unseen Speaker S322")
    r_tbl_title.font.name = "Calibri"
    r_tbl_title.font.size = Pt(9.5)
    r_tbl_title.font.bold = True
    r_tbl_title.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    table = doc.add_table(rows=len(table_data) + 1, cols=2)
    table.style = "Table Grid"
    
    col_widths = [Inches(2.3), Inches(4.2)]
    
    # Headers
    headers = ["Metric / Parameter", "Observed Quantitative Output"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.width = col_widths[j]
        set_cell_background(cell, "0D47A1")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(h)
        r.font.name = "Calibri"
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        
    # Data rows
    for i, (m, val) in enumerate(table_data):
        row_idx = i + 1
        bg = "F5F7FA" if (i % 2 == 1) else "FFFFFF"
        for j, text_val in enumerate([m, val]):
            cell = table.cell(row_idx, j)
            cell.width = col_widths[j]
            if bg != "FFFFFF":
                set_cell_background(cell, bg)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            r = p.add_run(text_val)
            r.font.name = "Calibri"
            r.font.size = Pt(8.5)
            if j == 0:
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
            else:
                r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
                
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(4)
    p_after.paragraph_format.space_after = Pt(6)
    
    return p_title, table, p_after

def main():
    print(f"[*] Backing up {DOCX_PATH} to {BACKUP_PATH}...")
    shutil.copyfile(DOCX_PATH, BACKUP_PATH)
    
    print(f"[*] Opening {DOCX_PATH}...")
    doc = docx.Document(DOCX_PATH)
    
    # 1. Figure 1: Raw vs Normalized (Section 3.3)
    print("[*] Inserting Figure 1 (raw vs normalized) in Section 3.3...")
    for p in doc.paragraphs:
        if "normalize_frame() -- orchestrates the above" in p.text:
            p_img, p_cap = create_figure(
                doc,
                "reports/figures/fig1_raw_vs_normalized.png",
                1,
                "Raw Landmark Coordinates vs. Scale-Normalized Kinematics.",
                "Mouth-centering and inter-ocular/resting-width scale normalization isolate genuine articulatory kinematics from rigid head motion."
            )
            p._p.addnext(p_img._p)
            p_img._p.addnext(p_cap._p)
            break
            
    # 2. Figure 2: Preview Frame (Section 3.8)
    print("[*] Inserting Figure 2 (preview frame) in Section 3.8...")
    for i, t in enumerate(doc.tables):
        if len(t.rows) == 7 and t.rows[0].cells[0].text.strip() == "Plot File":
            p_img, p_cap = create_figure(
                doc,
                "reports/figures/fig2_preview_frame.png",
                2,
                "Real-Time Multi-Modal Preview Frame Output.",
                "Synchronized 4-panel diagnostic view displaying face bounding box, 68 Dlib landmarks, cropped mouth ROI, and dense Gunnar Farnebäck optical flow vectors."
            )
            t._element.addnext(p_img._p)
            p_img._p.addnext(p_cap._p)
            break
            
    # 3. Figure 3 & 4: Displacement Heatmap & Mouth Opening (Section 7)
    print("[*] Inserting Figure 3 & 4 (heatmap & mouth opening) in Section 7...")
    for t in doc.tables:
        if len(t.rows) == 10 and t.rows[0].cells[0].text.strip() == "Output" and t.rows[0].cells[1].text.strip() == "File":
            p_img4, p_cap4 = create_figure(
                doc,
                "reports/figures/fig4_mouth_opening.png",
                4,
                "Articulatory Vertical Aperture Waveform, Velocity, and Acceleration Dynamics.",
                "Temporal kinematic aperture trace demonstrating clear velocity extrema that map directly onto phonetic transition boundaries."
            )
            p_img3, p_cap3 = create_figure(
                doc,
                "reports/figures/fig3_displacement_heatmap.png",
                3,
                "Spatiotemporal Landmark Displacement Heatmap Across 20 Lip Keypoints.",
                "Color-coded displacement intensities revealing differential kinematic activity between upper vermilion, lower vermilion, and inner oral boundaries."
            )
            t._element.addnext(p_img3._p)
            p_img3._p.addnext(p_cap3._p)
            p_cap3._p.addnext(p_img4._p)
            p_img4._p.addnext(p_cap4._p)
            break
            
    # 4. Figure 5: Two-Speaker Generalization (Section 11.4)
    print("[*] Inserting Figure 5 (two-speaker cross-validation) in Section 11.4...")
    for t in doc.tables:
        if len(t.rows) == 10 and "Experiment / Split" in t.rows[0].cells[0].text:
            p_img5, p_cap5 = create_figure(
                doc,
                "reports/figures/fig5_cross_speaker_generalization.png",
                5,
                "Two-Speaker Cross-Speaker Generalization Benchmark (S271 vs S275).",
                "Comparison of within-speaker baseline vs across-speaker generalization across Random Forest, SVM RBF, and HistGradientBoosting."
            )
            t._element.addnext(p_img5._p)
            p_img5._p.addnext(p_cap5._p)
            break
            
    # 5. Figure 6: Multi-Speaker LOSO & Feature Importance (Section 11.6)
    print("[*] Inserting Figure 6 (multi-speaker LOSO & feature importance) in Section 11.6...")
    for idx, p in enumerate(doc.paragraphs):
        if "11.6 Feature Importance & Kinematic Findings" in p.text:
            target_p = p
            for next_p in doc.paragraphs[idx+1:idx+10]:
                if "inner_area" in next_p.text or "Gini" in next_p.text:
                    target_p = next_p
                    break
            p_img6, p_cap6 = create_figure(
                doc,
                "reports/figures/fig6_multi_speaker_loso_feature_importance.png",
                6,
                "4-Fold Leave-One-Speaker-Out (LOSO) Cross-Validation Accuracy and Articulatory Feature Importance.",
                "Random Forest Gini importance rankings highlighting inner area, mouth opening, and normalized aperture dominance alongside speaker-independent test fold accuracies."
            )
            target_p._p.addnext(p_img6._p)
            p_img6._p.addnext(p_cap6._p)
            break
            
    # 6. Section 12: Output Table + Figure 7 + Figure 8
    print("[*] Inserting Table 24, Figure 7, and Figure 8 in Section 12...")
    for p in doc.paragraphs:
        if "Verification on Completely Unseen Speaker S322" in p.text:
            p_tbl_title, tbl_24, p_tbl_after = create_unseen_output_table(doc)
            
            p_img7, p_cap7 = create_figure(
                doc,
                "reports/figures/fig7_s322_prediction_timeline.png",
                7,
                "Automated Machine Learning Linguistic Prediction Timeline on Unseen Speaker S322.",
                "Continuous class-probability trajectories and discrete vowel/consonant classification decisions across 147 video frames without transcripts or audio."
            )
            
            p_img8, p_cap8 = create_figure(
                doc,
                "reports/figures/fig8_automated_linguistic_boundaries.png",
                8,
                "Automated Consonant-Vowel (C-V) Boundary Demarcation and Kinematic Segment Decomposition.",
                "Temporal alignment of kinematic aperture, velocity extrema, and automated segment demarcations on completely unseen speech."
            )
            
            # Place in order: target_p -> p_tbl_title -> tbl_24 -> p_tbl_after -> p_img7 -> p_cap7 -> p_img8 -> p_cap8
            p._p.addnext(p_tbl_title._p)
            p_tbl_title._p.addnext(tbl_24._tbl)
            tbl_24._tbl.addnext(p_tbl_after._p)
            p_tbl_after._p.addnext(p_img7._p)
            p_img7._p.addnext(p_cap7._p)
            p_cap7._p.addnext(p_img8._p)
            p_img8._p.addnext(p_cap8._p)
            break
            
    print(f"[*] Saving {DOCX_PATH}...")
    doc.save(DOCX_PATH)
    print(f"[OK] {DOCX_PATH} successfully saved!")
    
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
    main()
