#!/usr/bin/env python3
"""Create the material studio with homogeneous fog, glass, and finite-aperture camera."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from create_blender_fixture import create
from blender_export import arguments

create(arguments(), fog=True)
