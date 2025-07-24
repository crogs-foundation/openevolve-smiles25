import cadquery as cq

w0 = cq.Workplane("XY", origin=(-100, -41, -32))
w1 = cq.Workplane("XY", origin=(-94, -35, -32))
r = (
    w0.sketch()
    .face(
        w0.sketch()
        .segment((0, 0), (200, 0))
        .segment((200, 0), (200, 82))
        .segment((200, 82), (0, 82))
        .segment((0, 82), (0, 0))
        .assemble()
    )
    .face(
        w0.sketch()
        .segment((6, 6), (171, 6))
        .segment((171, 6), (171, 76))
        .segment((171, 76), (6, 76))
        .segment((6, 76), (6, 6))
        .assemble(),
        mode="s",
    )
    .finalize()
    .extrude(64)
    .union(
        w1.sketch()
        .face(
            w1.sketch()
            .segment((0, 0), (165, 0))
            .segment((165, 0), (165, 71))
            .segment((165, 71), (0, 71))
            .segment((0, 71), (0, 0))
            .assemble()
        )
        .finalize()
        .extrude(3)
    )
)

# Create a rectangular box with dimensions 0.75 × 0.3088 × 0.2404 units, featuring a rectangular
#    hole, and then add a rectangular plate on top with dimensions 0.6176 × 0.2647 × 0.011 units, both
#    aligned with the same orientation and position as described.
