import cadquery as cq


w0 = cq.Workplane("ZX", origin=(-100, 89, 0))
r = (
    w0.sketch()
    .face(
        w0.sketch()
        .segment((0, 0), (-59, 19))
        .segment((-59, 19), (-95, 69))
        .segment((-95, 69), (-95, 131))
        .segment((-95, 131), (-59, 181))
        .segment((-59, 181), (0, 200))
        .segment((0, 200), (59, 181))
        .segment((59, 181), (95, 131))
        .segment((95, 131), (95, 100))
        .segment((95, 100), (95, 69))
        .segment((95, 69), (59, 19))
        .segment((59, 19), (0, 0))
        .assemble()
    )
    .finalize()
    .extrude(-178)
)
