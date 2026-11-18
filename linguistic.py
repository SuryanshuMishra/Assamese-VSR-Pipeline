"""
Linguistic Analysis & Machine Learning Classification Engine
============================================================
Provides vowel vs consonant prediction, viseme classification,
and syllable boundary detection for Lip Motion Recognition (LMR).
"""
import sys
from pathlib import Path

# Ensure root and lmr package are accessible
_pkg_root = Path(__file__).resolve().parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from lmr.linguistics import LinguisticSegmenter, DEFAULT_MODEL_PATH

__all__ = ["LinguisticSegmenter", "DEFAULT_MODEL_PATH"]
