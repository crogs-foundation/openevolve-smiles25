import cadquery as cq

w0 = cq.Workplane("ZX", origin=(-58, 0, 0))
w1 = cq.Workplane("ZX", origin=(-33, -100, 0))
r = (
    w0.sketch()
    .face(w0.sketch().push([(0, 58)]).circle(58))
    .finalize()
    .extrude(-100, both=True)
    .cut(
        w1.sketch().face(w1.sketch().push([(0, 33)]).circle(33)).finalize().extrude(200)
    )
)
