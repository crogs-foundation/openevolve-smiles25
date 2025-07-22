# EVOLVE-BLOCK-START
"""Function that builds some figure which should match a reference 3D model in stl-format"""

import cadquery as cq

# Create base block dimensions (larger than hole diameter)
block_length = 30
block_width = 30
block_height = 15

# Create block and add centered through hole
r = (
    cq.Workplane("XY")
    .box(block_length, block_width, block_height)
    .faces(">Z")
    .workplane()
    .hole(22)
)  # 22mm diameter hole through entire thickness


# EVOLVE-BLOCK-END
