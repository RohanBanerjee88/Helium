"""
Shared pytest configuration.

Sets HELIUM_DATA_DIR and HELIUM_MIN_IMAGES before any app module is
imported so all tests run against a temp directory and accept 2+ images
instead of requiring the full 8-image minimum.
"""

import os
import tempfile

os.environ.setdefault("HELIUM_DATA_DIR", tempfile.mkdtemp())
os.environ.setdefault("HELIUM_MIN_IMAGES", "2")
