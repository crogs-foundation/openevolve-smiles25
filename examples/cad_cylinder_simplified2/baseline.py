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

# Start by constructing a cylindrical object with a flat circular top and bottom. First, create a new coordinate system with Euler angles set to 0, 0, -90 degrees and a translation vector of 0, 0.375, 0. Next, draw a 2-dimensional sketch on a new face. Within this face, draw a single loop and within the loop, draw a circle centered at (0.2188, 0.2188) with a radius of 0.2188. Apply a scale factor of 0.4375 to the sketch. Rotate and translate the scaled sketch using the previously defined coordinate system. Extrude the sketch 0.1875 units in both the normal and opposite directions to create a solid body. The dimensions of this first part are 0.4375 in length, 0.4375 in width, and 0.375 in height.

# For the second part, construct a hollow center in the cylindrical object. Begin by creating another new coordinate system with Euler angles set to 0, 0, -90 degrees and a translation vector of 0.0937, 0, 0.0937. Draw a 2-dimensional sketch on a new face. Within this face, draw a single loop and within the loop, draw a circle centered at (0.125, 0.125) with a radius of 0.125. Apply a scale factor of 0.25 to the sketch. Rotate and translate the scaled sketch using the previously defined coordinate system. Extrude the sketch 0.75 units in the opposite direction of the normal to cut material from the existing body, creating a hollow center. The dimensions of this second part are 0.25 in length, 0.25 in width, and 0.75 in height.
