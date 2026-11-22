"""Make the pipeline modules (pipeline/) importable from the tests, and run from the repository root."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))
os.chdir(ROOT)
