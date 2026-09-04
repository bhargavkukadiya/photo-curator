"""Backward-compatibility test runner for photo_curator.

Re-exports the test suites from tests/ so that legacy invocations of
`pytest test_album_selector.py` continue to run all unit tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on sys.path
_SRC_DIR = Path(__file__).resolve().parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tests.test_cli import TestCLIValidation
from tests.test_scoring import (
    TestAestheticScore,
    TestCalculateTotalScore,
    TestEmotionScore,
    TestIsImageFile,
    TestLoadImage,
    TestTechnicalScore,
)
from tests.test_selection import TestFilterNearDuplicates, TestScoredImage
from tests.test_storage import TestCopyTopImages, TestExportScoresCSV

__all__ = [
    "TestIsImageFile",
    "TestTechnicalScore",
    "TestAestheticScore",
    "TestEmotionScore",
    "TestCalculateTotalScore",
    "TestLoadImage",
    "TestScoredImage",
    "TestFilterNearDuplicates",
    "TestExportScoresCSV",
    "TestCopyTopImages",
    "TestCLIValidation",
]
