"""
Data Preparation Script for Multi-Speaker LMR Model Training
============================================================
Extracts 15 matching sentences across the MAXIMUM possible number of speakers
(4 speakers: S272, S273, S276, S278) -> 60 videos + 60 .lab annotations.
"""
import os
import zipfile
from pathlib import Path

base_dir = Path("q:/IITg intern")
out_dir = base_dir / "cross_speaker_data"
videos_dir = out_dir / "videos"
annotations_dir = out_dir / "annotations"

videos_dir.mkdir(parents=True, exist_ok=True)
annotations_dir.mkdir(parents=True, exist_ok=True)

# 15 Sentences common across all 4 speakers
sentences = [
    '1444_muk_khong_uthise',
    '1437_boroxun_ahi_ase',
    '1443_moi_sah_khalu',
    '1435_moi_football_khelu',
    '1447_azi_boroxun_ahibo',
    '1455_moi_bozar_zam',
    '1429_teu_kandi_ase',
    '1428_tai_gos_katise',
    '1445_moi_badminton_khelu',
    '1441_bola_phuri_ahu',
    '1414_dadaye_football_khelile',
    '1420_tai_porikkha_dile',
    '1453_ma_ghorot_ase',
    '1423_xi_ludu_khele',
    '1460_godhuli_phon_korilu',
]

speaker_zips = {
    's272': 'Bhargav Bharadwaj Kataky-20261116T174244Z-1-001.zip',
    's273': 'Bhargav Bharadwaj Kataky-20261116T174244Z-1-001.zip',
    's276': 'Chirag Dey-20261116T174242Z-1-001.zip',
    's278': 'Chirag Dey-20261116T174242Z-1-001.zip',
}

extracted_pairs = []

# Group speakers by zip to open each zip only once
zip_to_spks = {}
for spk, zname in speaker_zips.items():
    zip_to_spks.setdefault(zname, []).append(spk)

for zname, spk_list in zip_to_spks.items():
    zpath = base_dir / zname
    print(f"[*] Opening {zname} for speakers {spk_list}...")
    with zipfile.ZipFile(zpath) as z:
        namelist = z.namelist()
        for spk in spk_list:
            for s in sentences:
                # Find video
                v_matches = [
                    f for f in namelist 
                    if s in f and f.lower().endswith('.mp4') and f"{spk}_" in Path(f).name.lower()
                ]
                if not v_matches:
                    print(f"  [!] Video not found for {spk} - {s}")
                    continue
                v_file = v_matches[0]
                v_name = Path(v_file).name
                
                # Find matching lab
                l_matches = [
                    f for f in namelist 
                    if s in f and f.lower().endswith('.lab') and f"{spk}_" in Path(f).name.lower()
                ]
                if not l_matches:
                    print(f"  [!] Lab not found for {spk} - {s}")
                    continue
                l_file = l_matches[0]
                l_name = Path(l_file).name
                
                # Extract video
                v_dest = videos_dir / v_name
                if not v_dest.exists():
                    with open(v_dest, "wb") as f_out:
                        f_out.write(z.read(v_file))
                        
                # Extract lab
                l_dest = annotations_dir / l_name
                if not l_dest.exists():
                    with open(l_dest, "wb") as f_out:
                        f_out.write(z.read(l_file))
                        
                extracted_pairs.append({
                    "speaker": spk,
                    "sentence": s,
                    "video": str(v_dest),
                    "lab": str(l_dest)
                })

print(f"\n[OK] Target extraction complete!")
print(f"     Extracted items for this 4-speaker benchmark: {len(extracted_pairs)}")
print(f"     Total Videos in {videos_dir}: {len(list(videos_dir.glob('*.mp4')))}")
print(f"     Total Annotations in {annotations_dir}: {len(list(annotations_dir.glob('*.lab')))}")
