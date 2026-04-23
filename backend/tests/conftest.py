"""
Shared pytest configuration.

Sets a temp data directory and a small minimum audio file count before the
app is imported so tests run without touching user state.
"""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HELIUM_DATA_DIR", tempfile.mkdtemp())
os.environ.setdefault("HELIUM_MIN_AUDIO_FILES", "2")
