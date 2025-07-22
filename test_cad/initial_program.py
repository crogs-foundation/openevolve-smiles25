# EVOLVE-BLOCK-START
import cadquery as cq

length = 10
height = 10
thickness = 10

r = cq.Workplane("XY").box(length, height, thickness).faces(">Z").workplane().hole(0)
# EVOLVE-BLOCK-END


from datetime import datetime
from pathlib import Path

# This part remains fixed (not evolved)


def save_evolve_block(
    source_file: str = __file__, output_file: Path = Path("./test_cad/evolved_block.py")
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_output_filename = f"{output_file.stem}_{timestamp}{output_file.suffix}"

    with open(new_output_filename, "w", encoding="utf-8") as f:
        f.writelines(block_lines)

    return str(new_output_filename)


if __name__ == "__main__":
    r.export("./test_cad/result.stl")
