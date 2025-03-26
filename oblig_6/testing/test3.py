import pyglet as pg
import numpy as np

"""
We perform one circle-circle-collision.
 This is thought as a "get you started" script. There are many places where just
 hard-coded solutions get used and we use simplifications in the model like
   - same mass
   - same size
   - collision is in direction of motion etc.
 (But we cannot give away too much :-)
"""

WIDTH = 500
HEIGHT = WIDTH
window = pg.window.Window(WIDTH, HEIGHT)

c1 = np.array([WIDTH / 4, HEIGHT / 4], dtype=float)
vabs = 15
v1 = np.array([vabs, vabs], dtype=float)
c2 = np.array([3 * WIDTH / 4, 3 * HEIGHT / 4], dtype=float)
v2 = -v1.copy()
r1 = 30
e_rest = 0.6  # coefficient of restitution

circle1 = pg.shapes.Circle(c1[0], c1[1], r1, color=(255, 255, 0))
circle2 = pg.shapes.Circle(c2[0], c2[1], r1, color=(255, 0, 0))


@window.event
def on_draw():
    window.clear()

    circle1.x, circle1.y = c1[0], c1[1]
    circle2.x, circle2.y = c2[0], c2[1]

    circle1.draw()
    circle2.draw()


def update(dt):
    global c1, c2, v1, v2, collision
    dt = 0.1
    # update position
    c1 += dt * v1
    c2 += dt * v2

    # Check for collision
    # a) distance of the circle centers:
    circle_distance = np.linalg.norm(c1 - c2)
    # b) actual check
    collision = circle_distance < 2 * r1

    # Where a collision happens: Compute normal and tangent direction
    if collision:
        normal_vec = v2.copy()  # hard coded solution for this one special case(!)
        normal_vec /= np.linalg.norm(normal_vec)
        # tangent_vec = np.array([normal_vec[1], - normal_vec[0]]) # you need this later in the v1, v2-update
        #                             For this simplified model, we do not need it
        # perform 1d collision in the normal direction
        u1_normal = -np.linalg.norm(v1)
        u2_normal = np.linalg.norm(v2)
        v1_normal, v2_normal = one_d_collision(u1_normal, u2_normal)
        v1 = v1_normal * normal_vec  # Again:
        v2 = v2_normal * normal_vec  # simplified situation :-)
        # move objects out of collision
        c1 += (r1 - circle_distance / 2) / circle_distance * (c1 - c2)
        c2 += (r1 - circle_distance / 2) / circle_distance * (c2 - c1)


def one_d_collision(u1, u2):
    # This should take masses (and maybe some edge cases) into account ...
    return 0.5 * (u1 + u2 + e_rest * (u2 - u1)), 0.5 * (u1 + u2 + e_rest * (u1 - u2))


pg.clock.schedule_interval(update, 1 / 60)
pg.app.run()
