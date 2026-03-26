"""
pytest configuration for ADS_project test suite.
Adds the project root to sys.path so package imports resolve correctly.
"""

import sys
from pathlib import Path

# Insert project root so `backend.*` imports work from any working directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
