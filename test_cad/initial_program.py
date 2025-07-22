# EVOLVE-BLOCK-START
"""Function that builds some figure which should match a reference 3D model in stl-format"""

import cadquery as cq


def create_cad_figure():
    length = 10
    height = 10
    thickness = 10

    r = cq.Workplane("XY").box(length, height, thickness)
    r = cq.Workplane("XY").rect(40, 20).extrude(5)
    return r


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
import datetime


def run_build():
    r = create_cad_figure()
    return r


def save_evolve_block(
    source_file: str = __file__, output_file: str = "evolved_block.py"
):
    start_marker = "# EVOLVE-BLOCK-START"
    end_marker = "# EVOLVE-BLOCK-END"
    inside_block = False
    block_lines = []

    with open(source_file, "r", encoding="utf-8") as f:
        for line in f:
            if start_marker in line:
                inside_block = True
                block_lines.append(line)
                continue
            if end_marker in line:
                block_lines.append(line)
                break
            if inside_block:
                block_lines.append(line)

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(block_lines)

    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + output_file


if __name__ == "__main__":
    r = create_cad_figure()
    r.export("result.stl")
