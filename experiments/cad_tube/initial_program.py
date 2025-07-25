# EVOLVE-BLOCK-START
"""Function that builds 3D figure by text description"""

import cadquery as cq


def build_3d_figure() -> cq.Workplane:
    return cq.Workplane().sphere(0.1)


# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)

import os
from pathlib import Path

if __name__ == "__main__":
    figure = build_3d_figure()
    # Export the figure to an STL file
    dirname = os.path.dirname(__file__)
    figure.export(str(Path(os.path.join(dirname, "./figure.stl")).absolute()))
